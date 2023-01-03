import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

from vision_msgs.msg import Detection2DArray, Detection2D #BoundingBox3D, BoundingBox2D, BoundingBox2DArray
from builtin_interfaces.msg import Duration
from visualization_msgs.msg import Marker
from sensor_msgs.msg import PointCloud2, PointField, Image
from message_filters import ApproximateTimeSynchronizer, TimeSynchronizer, Subscriber
from sensor_msgs.msg import CameraInfo
from sensor_msgs_py import point_cloud2 as pc2
import numpy as np
import cv2
from cv_bridge import CvBridge, CvBridgeError
from ros2_numpy.point_cloud2 import array_to_pointcloud2, pointcloud2_to_array, get_xyz_points
import os
import time
import math
from numpy.linalg import inv
import open3d as o3d
from .utils_lidar import boxes_to_matirx
from scipy.spatial.transform import Rotation
print(os.getcwd())

class Lidar2Cam(Node):
    def __init__(self):
        print("Initialized Lidar2Cam Node!")
        super().__init__('lidar_to_cam_node')
        self.bridge = CvBridge()

        # Point Clouds
        self.front_cloud_msg = None
        self.left_cloud_msg  = None
        self.right_cloud_msg = None
        
        # Bbox Array
        self.bboxes_array_msg = None
        # self.img_msg = None
        # self.roi=None

        # ---------------------------------------------------------------------------- #
        #         No more Camera Matrix Because the YOLO Node will Deal With it        #
        # ---------------------------------------------------------------------------- #
        '''
        For this current attempt of synchronization, 
        We first Synchronize the 3 Lidar Point Clouds:
            1. Fill out the three Lidar buffers.
            2. Then, we will get the latest Point Cloud timestame and compare that
                with the earliest timestamp in the Detection2DArray.
            3. If the timestamp is within 10ms, we will proceed output the detection.
            4. If not, we will output an error (for the tracker) and wait for the next detection.
        '''
        # ---------------------------------------------------------------------------- #
        #                      Calibrated Transformation Matrices                      #
        # ---------------------------------------------------------------------------- #
        self.translation_luminar_front2_flc = np.array([ 0.017, -0.016, 0.156]) # Front Lidar to Front Left Center Camera
        self.translation_luminar_left2_flc  = np.array([ 0.017, -0.016, 0.156]) # Front Lidar to Front Right Center Camera
        self.translation_luminar_right2_flc = np.array([ 0.017, -0.016, 0.156]) # Front Lidar to Front Left Camera
        self.translation_luminar_front2_flc = np.array([ 0.017, -0.016, 0.156]) # Front Lidar to Front Right Camera
        self.translation_luminar_front2_flc = np.array([ 0.017, -0.016, 0.156]) # Left Lidar to Front Left Camera
        self.translation_luminar_front2_flc = np.array([ 0.017, -0.016, 0.156]) # Left Lidar to Rear Left Camera
        self.translation_luminar_front2_flc = np.array([ 0.017, -0.016, 0.156]) # Right Lidar to Front Right Camera
        self.translation_luminar_front2_flc = np.array([ 0.017, -0.016, 0.156]) # Right Lidar to Rear Right Camera
        self.RotMat_luminar_front2_flc = np.array([[ 0.02135093, -0.99976672, -0.00326259],
                                                    [ 0.05990699,  0.00453683, -0.99819365],
                                                    [ 0.9979756,   0.02111691,  0.05998988]])
        self.RotMat_luminar_front2_flc = np.array([[ 0.02135093, -0.99976672, -0.00326259],
                                                    [ 0.05990699,  0.00453683, -0.99819365],
                                                    [ 0.9979756,   0.02111691,  0.05998988]])                                            
        self.RotMat_luminar_front2_flc = np.array([[ 0.02135093, -0.99976672, -0.00326259],
                                                    [ 0.05990699,  0.00453683, -0.99819365],
                                                    [ 0.9979756,   0.02111691,  0.05998988]])
        self.RotMat_luminar_front2_flc = np.array([[ 0.02135093, -0.99976672, -0.00326259],
                                                    [ 0.05990699,  0.00453683, -0.99819365],
                                                    [ 0.9979756,   0.02111691,  0.05998988]])         
        self.RotMat_luminar_front2_flc = np.array([[ 0.02135093, -0.99976672, -0.00326259],
                                                    [ 0.05990699,  0.00453683, -0.99819365],
                                                    [ 0.9979756,   0.02111691,  0.05998988]])
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
        
        # Front LIDAR PointCloud ----------------------------- #
        self.front_pointcloud_sub = self.create_subscription(
            msg_type    = PointCloud2,
            topic       = '/luminar_front_points',
            callback    = self.front_cloud_callback,
            qos_profile = self.qos_profile
        )
        self.front_pointcloud_sub  # prevent unused variable warning

        # Left LIDAR PointCloud ----------------------------- #
        self.left_pointcloud_sub = self.create_subscription(
            msg_type    = PointCloud2,
            topic       = '/luminar_left_points',
            callback    = self.left_cloud_callback,
            qos_profile = self.qos_profile
        )
        self.left_pointcloud_sub  # prevent unused variable warning

        # Right LIDAR PointCloud ----------------------------- #
        self.right_pointcloud_sub = self.create_subscription(
            msg_type    = PointCloud2,
            topic       = '/luminar_right_points',
            callback    = self.right_cloud_callback,
            qos_profile = self.qos_profile
        )
        self.right_pointcloud_sub  # prevent unused variable warning

        # YOLO Detection2DArray Subscription ---------------------------------- #
        self.image_sub = self.create_subscription(
            msg_type    = Detection2DArray, 
            topic       = 'vimba_front_left_center/out/objects',
            callback    = self.YOLO_callback,
            qos_profile = self.qos_profile
        )
        self.image_sub # prevent unused variable warning

        # # YOLO BBOX Image ---------------------------------- #
        # self.image_sub = self.create_subscription(
        #     msg_type    = Image,
        #     topic       = 'vimba_front_left_center/out/image',
        #     callback    = self.img_callback,
        #     qos_profile = self.qos_profile
        # )
        # self.image_sub # prevent unused variable warning

        # ---------------------------------------------------------------------------- #
        #                                  Publishers                                  #
        # ---------------------------------------------------------------------------- #
        self.marker_pub = self.create_publisher(Marker, '/Lidar_car_marker', rclpy.qos.qos_profile_sensor_data)
        # self.image_pub = self.create_publisher(Image , "/Lidar_filtered_label", rclpy.qos.qos_profile_sensor_data)


    # ---------------------------------------------------------------------------- #
    #                     Callback Functions for Subscriptions                     #
    # ---------------------------------------------------------------------------- #
    # ----------- If all of the buffers are filled, execute_projection. ---------- #
    # ------------------------------ YOLO Detection ------------------------------ #
    def YOLO_callback(self, msg):
        # print("B")
        self.bboxes_array_msg = msg
        if np.all([self.front_cloud_msg, self.left_cloud_msg, self.right_cloud_msg]) is not None:
            # print("+")
            self.execute_projection()

    # -------------------------------- PointClouds ------------------------------- #
    def front_cloud_callback(self, msg):
        # print("P")
        self.front_cloud_msg = msg
        if np.all([self.bboxes_array_msg, self.left_cloud_msg, self.right_cloud_msg]) is not None:
            # print("+")
            self.execute_projection()

    def left_cloud_callback(self, msg):
        # print("P")
        self.left_cloud_msg = msg
        if np.all([self.front_cloud_msg, self.bboxes_array_msg, self.right_cloud_msg]) is not None:
            # print("+")
            self.execute_projection()

    def right_cloud_callback(self, msg):
        # print("P")
        self.right_cloud_msg = msg
        if np.all([self.front_cloud_msg, self.left_cloud_msg, self.bboxes_array_msg]) is not None:
            # print("+")
            self.execute_projection()

    # # ---------------------------------------------------------------------------- #
    # #         Helper Functions for Projection Calculation and Visualization        #
    # # ---------------------------------------------------------------------------- #
    # def img_tocv2(self, message):
    #     try:
    #         # Convert the ROS2 Image message to a NumPy array
    #         image = self.bridge.imgmsg_to_cv2(message, "bgr8")
    #     except CvBridgeError as e:
    #         print(e)
    #     return image

    # ---------------------------------------------------------------------------- #
    #                Inverse Lidar camera projection right here                    #
    # ---------------------------------------------------------------------------- #
    def execute_projection(self):
        '''
        1. Check if the timestamps are in sync. It not, return some sort of error

        Plan to do the new projection:
        2. Transform this into Lidar Frame
        3. Normalize all Lidar Points on their Z-axis
        4. capture all points within the bounding box
        5. Convert bounding box corners into lidar frame 
        '''
        
        # #1
        # K_inv = inv(self.camera_info)
        # camera_corners_cam = K_inv @ boxes_to_matirx([self.bbox_msg],0)
        #1
        if 
        
        #2
        # Apply Inverse rotation matrix
        R_inv = inv(self.RotMat_luminar_front2_flc) 
        camera_corners_lid = R_inv @ camera_corners_cam # This operation will make the bottom row not necessarily zero
        camera_corners_lid_z = camera_corners_lid[:,0:1]
        camera_corners_lid_normed = camera_corners_lid[:,1:]/camera_corners_lid_z
        camera_corners_lid_normed=camera_corners_lid_normed.T


        #3
        # Normalize all points on their Z-axis
        trans=R_inv@self.translation_luminar_front2_flc
        ptc_numpy_record = pointcloud2_to_array(self.point_cloud_msg)
        ptc_xyz_lidar = get_xyz_points(ptc_numpy_record) # (N * 3 matrix)
        ptc_xyz_lidar-=trans[np.newaxis,:]
        ptc_z_camera = ptc_xyz_lidar[:, 0].reshape((-1, 1))
        ptc_xyz_lidar_normed = ptc_xyz_lidar[:,1:]/ptc_z_camera


        #4
        # Capture all points within the bounding box
        mask = np.full((ptc_xyz_lidar_normed.shape[0],), False)
        for i in range(camera_corners_lid_normed.shape[2]):#camera_corners_lid.shape[1]): #(Columns as size)
            mask = (mask | ((ptc_xyz_lidar_normed[:,0]>=camera_corners_lid_normed[0,0,i]) & # x>=left
                            (ptc_xyz_lidar_normed[:,0]<=camera_corners_lid_normed[1,0,i]) & # x<=right
                            (ptc_xyz_lidar_normed[:,1]>=camera_corners_lid_normed[0,1,i]) & #y>=top
                            (ptc_xyz_lidar_normed[:,1]<=camera_corners_lid_normed[1,1,i]))) #y<=bottom
                            # Space for Optimization here
            
        ptc_xyz_lidar_filtered = ptc_xyz_lidar[mask]
        num_lidar=np.sum(mask)
        print('num lidar in bbox:',num_lidar)
        if num_lidar==0:
            self.image_pub.publish(self.img_msg)
            self.image_msg = None
            self.point_cloud_msg = None 
            self.bbox_msg=None
            return



        # ---------------------------------------------------------------------------- #
        #                                   Custering                                  #
        # ---------------------------------------------------------------------------- #
        ptc_xyz_camera_list=ptc_xyz_lidar_filtered
        o3d_pcd: o3d.geometry.PointCloud = o3d.geometry.PointCloud(
            o3d.utility.Vector3dVector(ptc_xyz_camera_list)
        )
        labels = np.array(
            o3d_pcd.cluster_dbscan(
                eps=1.8, min_points=max(int(num_lidar/4),10), print_progress=False
            )
        )
        max_label = labels.max()
        print('label num:',max_label+1)

        # ---------------------------------------------------------------------------- #
        #                                Calculate 3D BBox                             #
        # ---------------------------------------------------------------------------- #
        median_xyz_camera=[]
        bbox_rotate_mode=True
        for i in range(max_label+1):
            print(f'label {i} num:',np.where(labels==i)[0].shape)
            cluster=o3d_pcd.select_by_index(list(np.where(labels==i)[0]))
            if bbox_rotate_mode:
                cluster_bbox=cluster.get_oriented_bounding_box()
            else:
                cluster_bbox=cluster.get_axis_aligned_bounding_box()
            median_xyz_camera.append([cluster_bbox,abs(cluster_bbox.get_center()[0]),i])
        median_xyz_camera.sort(key=lambda x:x[1])
        if max_label>-1:
            print('label choosed:',median_xyz_camera[0][2],'distance:',median_xyz_camera[0][1])

        # # ---------------------------------------------------------------------------- #
        # #                   Creating a Marker Object to be published                   #
        # # ---------------------------------------------------------------------------- #

        for cluster_bbox,_,_ in median_xyz_camera:
            if bbox_rotate_mode:
                center = cluster_bbox.get_center()
                extent=cluster_bbox.extent
                rotation=cluster_bbox.R.copy()
                rotation[2]*=-1
                rotation=Rotation.from_matrix(rotation).as_quat()
                center=[center[0],-center[1],center[2]]
            else:
                center = cluster_bbox.get_center()
                extent=cluster_bbox.get_extent()
                center=[center[0],-center[1],center[2]]

            marker_msg = Marker()
            marker_msg.header.frame_id = "luminar_front"
            marker_msg.header.stamp = self.point_cloud_msg.header.stamp
            marker_msg.ns = "Lidar_detection"
            marker_msg.id = 0
            marker_msg.type = Marker().CUBE
            marker_msg.action = 0
            marker_msg.pose.position.x = center[0]
            marker_msg.pose.position.y = -center[1]
            marker_msg.pose.position.z = center[2]
            if bbox_rotate_mode:
                marker_msg.pose.orientation.x = rotation[0]
                marker_msg.pose.orientation.y = rotation[1]
                marker_msg.pose.orientation.z = rotation[2]
                marker_msg.pose.orientation.w = rotation[3]
            marker_msg.scale.x = extent[0]
            marker_msg.scale.y = extent[1]
            marker_msg.scale.z = extent[2]
            marker_msg.color.a = 0.5
            marker_msg.color.r = 1.0
            marker_msg.lifetime = Duration(sec=0, nanosec=400000000)

            self.marker_pub.publish(marker_msg)
            break

        # ---------------------------------------------------------------------------- #
        #                  Reflecting the points on the labelled image                 #
        # ---------------------------------------------------------------------------- #
        if self.img_msg  is not None:
            if max_label>=0:
                image = self.img_tocv2(self.img_msg) 
                ptc_numpy_record = pointcloud2_to_array(self.point_cloud_msg)
                ptc_xyz_lidar = get_xyz_points(ptc_numpy_record) # (N * 3 matrix)
                ptc_xyz_camera_filtered = self.RotMat_luminar_front2_flc @ ptc_xyz_lidar.T + self.translation_luminar_front2_flc[:,np.newaxis]
                ptc_xyz_camera_filtered = self.camera_info @ ptc_xyz_camera_filtered
                ptc_xyz_camera_filtered = ptc_xyz_camera_filtered.T
                ptc_xyz_camera_filtered = ptc_xyz_camera_filtered[mask][np.where(labels==median_xyz_camera[0][2])]
                ptc_z_camera = ptc_xyz_camera_filtered[:, 2].reshape((-1, 1))
                ptc_xyz_camera_filtered = ptc_xyz_camera_filtered/(ptc_z_camera)
                
                
                border_size=0
                image_undistorted=cv2.copyMakeBorder(image,border_size,border_size,border_size,border_size,cv2.BORDER_CONSTANT,None,0)

                z_min=np.min(ptc_z_camera)
                z_range=np.max(ptc_z_camera)-z_min
                ptc_z_camera=(ptc_z_camera-z_min)*255/z_range
                ptc_z_camera=ptc_z_camera.astype(np.uint8)
                color=cv2.applyColorMap(ptc_z_camera[:,np.newaxis],cv2.COLORMAP_HSV)
                r=ptc_xyz_camera_filtered.shape[0]
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

                # Publishing the Image and PointCloud
                self.image_pub.publish(self.bridge.cv2_to_imgmsg(image_undistorted))
            else:
                self.image_pub.publish(self.img_msg)

        # ---------------------------------------------------------------------------- #
        #    Setting the buffers to None to wait for the next image-pointcloud pair    #
        # ---------------------------------------------------------------------------- #
        self.image_msg = None
        self.point_cloud_msg = None 
        self.bbox_msg=None

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
