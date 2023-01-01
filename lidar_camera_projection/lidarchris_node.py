import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

from vision_msgs.msg import Detection3D, BoundingBox3D, BoundingBox2D #, BoundingBox2DArray
from builtin_interfaces.msg import Duration
from visualization_msgs.msg import Marker
from sensor_msgs.msg import PointCloud2, PointField, Image
from message_filters import ApproximateTimeSynchronizer, TimeSynchronizer, Subscriber
from sensor_msgs.msg import CameraInfo
from sensor_msgs_py import point_cloud2 as pc2
import numpy as np
import cv2
# from matplotlib import pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
from cv_bridge import CvBridge, CvBridgeError
# import ros2_numpy as rp
from ros2_numpy.point_cloud2 import array_to_pointcloud2, pointcloud2_to_array, get_xyz_points
import os
import time
import math
from numpy.linalg import inv
import open3d as o3d
from .utils_lidar import boxes_to_matirx
print(os.getcwd())

class Lidar2Cam(Node):
    def __init__(self):
        print("Initialized Lidar2Cam Node!")
        super().__init__('lidar_to_cam_node')
        self.bridge = CvBridge()
        self.point_cloud_msg = None
        self.bbox_msg = None # Initialize an empty array
        self.img_msg = None

        # ---------------------------------------------------------------------------- #
        #         All the Hard Coded Matrices Applies to Front Left Camera ONLY        #
        # ---------------------------------------------------------------------------- #
        self.camera_info = np.array([[1732.571708, 0.000000, 549.797164], 
                                     [0.000000, 1731.274561, 295.484988], 
                                     [0.000000, 0.000000, 1.000000*2]])*0.5
        print(self.camera_info)

        # ---------------------------------------------------------------------------- #
        #                      Calibrated Transformation Matrices                      #
        # ---------------------------------------------------------------------------- #
        self.translation_luminar_front2_flc = np.array([ 0.017, -0.016, 0.156])
        self.RotMat_luminar_front2_flc = np.array([[ 0.02135093, -0.99976672, -0.00326259],
                                                    [ 0.05990699,  0.00453683, -0.99819365],
                                                    [ 0.9979756,   0.02111691,  0.05998988]])

        # ---------------------------------------------------------------------------- #
        #                                  QOS Profile                                 #
        # ---------------------------------------------------------------------------- #
        self.qos_profile =  QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # ---------------------------------------------------------------------------- #
        #                                 Subscriptions                                #
        # ---------------------------------------------------------------------------- #
        
        # LIDAR PointCloud ----------------------------- #
        self.pointcloud_sub = self.create_subscription(
            msg_type    = PointCloud2,
            topic       = '/luminar_front_points',
            callback    = self.point_cloud_callback,
            qos_profile = self.qos_profile
        )
        self.pointcloud_sub  # prevent unused variable warning

        # Bounding Box ---------------------------------- #
        self.image_sub = self.create_subscription(
            msg_type    = BoundingBox2D, 
            topic       = 'vimba_front_left_center/out/objects',
            callback    = self.bbox_callback,
            qos_profile = self.qos_profile
        )
        self.image_sub # prevent unused variable warning

        # YOLO BBOX Image ---------------------------------- #
        self.image_sub = self.create_subscription(
            msg_type    = Image,
            topic       = 'vimba_front_left_center/out/image',
            callback    = self.img_callback,
            qos_profile = self.qos_profile
        )
        self.image_sub # prevent unused variable warning

        # ---------------------------------------------------------------------------- #
        #                                  Publishers                                  #
        # ---------------------------------------------------------------------------- #
        # self.marker_pub = self.create_publisher(Marker, '/Lidar_car_marker', rclpy.qos.qos_profile_sensor_data)
        self.image_pub = self.create_publisher(Image , "/Lidar_filtered_label", rclpy.qos.qos_profile_sensor_data)


    # ---------------------------------------------------------------------------- #
    #                     Callback Functions for Subscriptions                     #
    # ---------------------------------------------------------------------------- #
    def img_callback(self, msg):
        # print("I")
        self.img_msg = msg

    def bbox_callback(self, msg):
        # print("B")
        self.bbox_msg = msg
        if self.point_cloud_msg is not None:
            # print("+")
            self.bbox_ptc_callback()

    def point_cloud_callback(self, msg):
        # print("P")
        self.point_cloud_msg = msg
        if self.bbox_msg is not None:
            # print("+")
            self.bbox_ptc_callback()

    # ---------------------------------------------------------------------------- #
    #         Helper Functions for Projection Calculation and Visualization        #
    # ---------------------------------------------------------------------------- #
    def img_tocv2(self, message):
        try:
            # Convert the ROS2 Image message to a NumPy array
            image = self.bridge.imgmsg_to_cv2(message, "bgr8")
        except CvBridgeError as e:
            print(e)
        return image

    # ---------------------------------------------------------------------------- #
    #                Inverse Lidar camera projection right here                    #
    # ---------------------------------------------------------------------------- #
    def bbox_ptc_callback(self):
        '''
        Plan to do the new projection:
        1. Apply inverse K matrix onto bbox corners
        2. Transform this into Lidar Frame
        3. Normalize all Lidar Points on their Z-axis
        4. capture all points within the bounding box
        5. Convert bounding box corners into lidar frame 
        '''
        
        #1
        K_inv = inv(self.camera_info)
        # print(self.bbox_msg)
        image = self.img_tocv2(self.img_msg)
        min_x,min_y,w,h=cv2.selectROI('roi',image,False,False)
        cv2.waitKey(0)
        bbox_matrix=np.array([[min_x,min_x+w],[min_y,min_y+h],[1,1]])
        # camera_corners_cam = K_inv @ boxes_to_matirx([self.bbox_msg])
        camera_corners_cam = K_inv @ bbox_matrix
        print(camera_corners_cam.shape)
        # camera_corners_cam=camera_corners_cam[:,[1,3]]
        # print(camera_corners_cam.shape)
        
        #2
        # Apply Inverse rotation matrix
        R_inv = inv(self.RotMat_luminar_front2_flc) 
        numboxes = 1
        # translation_stacked = np.tile(self.translation_luminar_front2_flc.reshape((3, 1)), 4*numboxes)
        camera_corners_cam = camera_corners_cam - self.translation_luminar_front2_flc[:,np.newaxis]
        camera_corners_lid = R_inv @ camera_corners_cam # This row will make the bottom row not necessarily zero
        # print(camera_corners_lid)
        camera_corners_lid_z = camera_corners_lid[0, :]
        camera_corners_lid_normed = (camera_corners_lid[1:]/camera_corners_lid_z).T
        print(camera_corners_lid_normed)


        #3
        # Normalize all points on their Z-axis
        ptc_numpy_record = pointcloud2_to_array(self.point_cloud_msg)
        ptc_xyz_lidar = get_xyz_points(ptc_numpy_record) # (N * 3 matrix)
        ptc_z_camera = ptc_xyz_lidar[:, 0].reshape((-1, 1))
        ptc_xyz_lidar_normed = ptc_xyz_lidar[:,1:]/ptc_z_camera
        print(ptc_xyz_lidar_normed.shape,np.min(ptc_xyz_lidar_normed,0),np.max(ptc_xyz_lidar_normed,0))

        # Save the array to a file
        np.savetxt('LidarUsedToMask.txt', ptc_xyz_lidar_normed, fmt='%.4f')
        # Save the array to a file
        np.savetxt('CameraCorners.txt', camera_corners_lid_normed, fmt='%.4f')

        #4
        # Capture all points within the bounding box
        mask = np.full((ptc_xyz_lidar_normed.shape[0],), False)
        for i in range(1):#camera_corners_lid.shape[1]): #(Columns as size)
            offset = 3*i
            mask = (mask | ((ptc_xyz_lidar_normed[:,0]>=camera_corners_lid_normed[0,0]) & # x>=left
                            (ptc_xyz_lidar_normed[:,0]<=camera_corners_lid_normed[1,0]) & # x<=right
                            (ptc_xyz_lidar_normed[:,1]>=camera_corners_lid_normed[1,1]) & #y>=top
                            (ptc_xyz_lidar_normed[:,1]<=camera_corners_lid_normed[0,1]))) #y<=bottom
                            # Space for Optimization here
            
        # ptc_xyz_lidar_filtered = ptc_xyz_lidar[mask]
        print(np.sum(mask))


        # ---------------------------------------------------------------------------- #
        #                  Reflecting the points on the labelled image                 #
        # ---------------------------------------------------------------------------- #
        image = self.img_tocv2(self.img_msg) 
        translation_stacked = np.tile(self.translation_luminar_front2_flc.reshape((-1, 1)), ptc_xyz_lidar.shape[0])
        ptc_xyz_camera_filtered = self.RotMat_luminar_front2_flc @ ptc_xyz_lidar.T + translation_stacked
        ptc_xyz_camera_filtered = self.camera_info @ ptc_xyz_camera_filtered
        ptc_xyz_camera_filtered = ptc_xyz_camera_filtered.T
        ptc_z_camera = ptc_xyz_camera_filtered[:, 2].reshape((-1, 1))
        ptc_xyz_camera_filtered = ptc_xyz_camera_filtered/(ptc_z_camera)
        ptc_xyz_camera_filtered = ptc_xyz_camera_filtered[mask]
        # print("mask: ", np.any( mask))
        
        border_size=300
        image_undistorted=cv2.copyMakeBorder(image,border_size,border_size,border_size,border_size,cv2.BORDER_CONSTANT,None,0)

        z_min=np.min(ptc_z_camera)
        z_range=np.max(ptc_z_camera)-z_min
        ptc_z_camera=(ptc_z_camera-z_min)*255/z_range
        ptc_z_camera=ptc_z_camera.astype(np.uint8)
        color=cv2.applyColorMap(ptc_z_camera[:,np.newaxis],cv2.COLORMAP_HSV)
        r=ptc_xyz_camera_filtered.shape[0]
        print(r)
        for j in range(r):
            i=ptc_xyz_camera_filtered[j]
            c=color[np.newaxis,np.newaxis,j,0]
            a = int(np.floor(i[0]) + border_size)
            b = int(np.floor(i[1]) + border_size)
            if a>0 and b>0:
                try:
                    image_undistorted[b-1:b+2,a-1:a+2] = c
                except:
                    continue

        # for i in ptc_xyz_camera[mask]:
        #     image_undistorted = cv2.circle(image_undistorted, (int(i[0]+border_size), int(i[1])+border_size), 1, (0, 0, 255), 2)
        #     # image_undistorted = cv2.circle(image_undistorted, (int(i[0]), int(i[1])), 1, (0, 0, 255), 1)

        # Publishing the Image and PointCloud
        self.image_pub.publish(self.bridge.cv2_to_imgmsg(image_undistorted))
        # # ---------------------------------------------------------------------------- #
        # #                                   Custering                                  #
        # # ---------------------------------------------------------------------------- #
        # ptc_xyz_camera_list=ptc_xyz_lidar_filtered
        # median_xyz_camera=[]
        # o3d_pcd: o3d.geometry.PointCloud = o3d.geometry.PointCloud(
        #     o3d.utility.Vector3dVector(ptc_xyz_camera_list)
        # )
        # labels = np.array(
        #     o3d_pcd.cluster_dbscan(
        #         eps=2, min_points=40, print_progress=False
        #     )
        # )
        # max_label = labels.max()
        # print(max_label)
        # for i in range(max_label+1):
        #     print(np.where(labels==i)[0].shape)
        #     cluster=o3d_pcd.select_by_index(list(np.where(labels==i)[0]))

        #     cluster_bbox=cluster.get_axis_aligned_bounding_box()
        #     print(len(cluster_bbox.get_point_indices_within_bounding_box(o3d.utility.Vector3dVector(ptc_xyz_camera_list))))
        #     median_xyz_camera.append(cluster_bbox)
        #     print(cluster_bbox)
 

        # # ---------------------------------------------------------------------------- #
        # #                   Creating a Marker Object to be published                   #
        # # ---------------------------------------------------------------------------- #

        # for cluster_bbox in median_xyz_camera:
        #     center_max=cluster_bbox.get_max_bound()
        #     center_min=cluster_bbox.get_min_bound()
        #     center = cluster_bbox.get_center()
        #     extent=cluster_bbox.get_extent()
        #     print(center,extent)
        #     center=[center[0],center[1]+extent[1]*1.25,center[2]]

        #     marker_msg = Marker()
        #     marker_msg.header.frame_id = "luminar_front"
        #     marker_msg.header.stamp = self.point_cloud_msg.header.stamp
        #     marker_msg.ns = "Lidar_detection"
        #     marker_msg.id = 0
        #     marker_msg.type = Marker().CUBE
        #     marker_msg.action = 0
        #     marker_msg.pose.position.x = center[0]
        #     marker_msg.pose.position.y = -center[1]
        #     marker_msg.pose.position.z = center[2]
        #     # marker_msg.pose.orientation.x = rotation[0]
        #     # marker_msg.pose.orientation.y = rotation[1]
        #     # marker_msg.pose.orientation.z = rotation[2]
        #     # marker_msg.pose.orientation.w = rotation[3]
        #     marker_msg.scale.x = -extent[0]
        #     marker_msg.scale.y = extent[1]
        #     marker_msg.scale.z = -extent[2]
        #     marker_msg.color.a = 0.5
        #     marker_msg.color.r = 1.0
        #     marker_msg.lifetime = Duration(sec=0, nanosec=400000000)

        #     self.marker_pub.publish(marker_msg)

        # ---------------------------------------------------------------------------- #
        #    Setting the buffers to None to wait for the next image-pointcloud pair    #
        # ---------------------------------------------------------------------------- #
        self.image_msg = None
        self.point_cloud_msg = None 

def main(args=None):
    rclpy.init(args=args)

    minimal_subscriber = Lidar2Cam()
    print("going to spin")
    rclpy.spin(minimal_subscriber)
    print("destroying node")
    # Destroy the node explicitly
    minimal_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
