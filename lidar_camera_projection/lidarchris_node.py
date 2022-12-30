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
        self.marker_pub = self.create_publisher(Marker, '/Lidar_car_marker', rclpy.qos.qos_profile_sensor_data)



    # ---------------------------------------------------------------------------- #
    #                     Callback Functions for Subscriptions                     #
    # ---------------------------------------------------------------------------- #
    def img_callback(self, msg):
        print("I")
        self.img_msg = msg

    def bbox_callback(self, msg):
        print("B")
        self.bbox_msg = msg
        if self.point_cloud_msg is not None:
            print("+")
            self.bbox_ptc_callback()

    def point_cloud_callback(self, msg):
        print("P")
        self.point_cloud_msg = msg
        if self.bbox_msg is not None:
            print("+")
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


    # ------ Converts pointclouds(in camera frame) to spherical Coordinates ------ # (Spherical coordinates following Wikipedia definition here: https://en.wikipedia.org/wiki/Spherical_coordinate_system)
    # Output points are in degrees
    def xyz2spherical(self, ptc_arr):
        def cartesian_to_spherical(xyz):
            x, y, z = xyz # in camera frame
            rho   = math.sqrt(x**2 + y**2 + z**2)
            r     = math.sqrt(x**2 + y**2)
            
            # Determine theta
            if z>0:
                theta = math.atan(r/z)
            elif z<0:
                theta = math.pi + math.atan(r/z)
            elif z==0 and x!=0 and y!=0:
                theta = math.pi/2
            elif x==0 and y==0 and z==0:
                theta = None
            else:
                theta = math.acos(z/rho)
            
            # Determine Phi
            if x>0:
                phi = math.atan(y/x)
            elif x<0 and y>=0:
                phi = math.atan(y/x)
            elif x<0 and y<0:
                phi = math.atan(y/x) - math.pi
            elif x==0 and y>0:
                phi = math.pi
            elif x<0 and y<0:
                phi = math.pi
            elif x == 0 and y == 0:
                phi = None
            else:
                phi = (0 if y==0 else 1 if x > 0 else -1)*math.acos(x/r)

            return [phi, theta, rho]

        # Apply Spherical Conversion on each point
        return np.array(list(map(lambda x: cartesian_to_spherical(x), ptc_arr)))

    # --------------- Converts Spherical Coordinates to Pointcloud in Camera Frame -------------- #
    def spherical2xyz(self, spr_arr):
        def spherical_to_cartesian(spr):
            phi, theta, rho = spr
            x = rho * math.sin(theta) * math.cos(phi)
            y = rho * math.sin(theta) * math.sin(phi)
            z = rho * math.cos(theta)
            return [x, y, z]
        # Apply Cartesian Conversion on each point
        return np.array(list(map(lambda x: spherical_to_cartesian(x), spr_arr)))
    

    # ---- Converts bbox coordinate format from (centerpoint, size_x, size_y) ---- #
    # ------------------- to (top_y, bottom_y, left_x, right_x) ------------------ #
    def box_to_corners(self, cx, cy, width, height):
        half_width = width / 2
        half_height = height / 2

        y1 = cy - half_height # top
        y2 = cy + half_height # bottom
        x1 = cx - half_width # left
        x2 = cx + half_width # right

        return (y1, y2, x1, x2) #(up, down, left, right)


    # # ---------------------------------------------------------------------------- #
    # #                      Lidar camera projection right here                      #
    # # ---------------------------------------------------------------------------- #
    # def bbox_ptc_callback(self):

    #     ptc_numpy_record = pointcloud2_to_array(self.point_cloud_msg)
    #     ptc_xyz_lidar = get_xyz_points(ptc_numpy_record)
    #     numpoints = ptc_xyz_lidar.shape[0]
    #     assert(ptc_xyz_lidar.shape[1] == 3), "PointCloud_lidar is not N x 3"
   

    #     # --------------------------- Applying the Rotation -------------------------- # Alt + x
    #     translation_luminar_front2_flc = np.tile(self.translation_luminar_front2_flc.reshape((3, 1)), numpoints)
    #     assert(translation_luminar_front2_flc.shape == (3, numpoints)), "Translation is not 3 x N"
    #     ptc_xyz_camera_real = self.RotMat_luminar_front2_flc @ ptc_xyz_lidar.T + translation_luminar_front2_flc # This is correct
    #     ptc_xyz_camera_real = ptc_xyz_camera_real.T
    #     assert(ptc_xyz_camera_real.shape == (numpoints, 3)), "PointCloud_camera is not N x 3"


    #     # ------------------------- Applying the Camera Info ------------------------- #
    #     ptc_xyz_camera_real      = ptc_xyz_camera_real.T
    #     ptc_xyz_camera_real_px = self.camera_info @ ptc_xyz_camera_real
    #     ptc_xyz_camera_real_px = ptc_xyz_camera_real_px.T
    #     ptc_xyz_camera_real      = ptc_xyz_camera_real.T


    #     # ---------------------- Applying division on the Z-axis --------------------- #
    #     ptc_z_camera = ptc_xyz_camera_real_px[:, 2]
    #     ptc_xyz_camera_normed = np.divide(ptc_xyz_camera_real_px, ptc_z_camera.reshape(ptc_xyz_camera_real_px.shape[0], 1))


    #     # ---------------------------------------------------------------------------- #
    #     #                  PointCloud Filtering to get points in BBOX                  #
    #     # ---------------------------------------------------------------------------- #

    #     # ------------------- Convert the bbox msg to (top, bottom, left, right) bounds ------------------ #
    #     y1, y2, x1, x2 = self.box_to_corners(self.bbox_msg.center.x, self.bbox_msg.center.y, self.bbox_msg.size_x, self.bbox_msg.size_y)
        
    #     # ----- Applying detection strategy to cut off top 40% of the bbox ----- #
    #     y1 = y1 + int(0.4 * self.bbox_msg.size_y)
    #     mask=(ptc_xyz_camera_normed[:,0]>=x1)&(ptc_xyz_camera_normed[:,0]<=x2)&(ptc_xyz_camera_normed[:,1]>=y1)&(ptc_xyz_camera_normed[:,1]<=y2)
        
    #     # ----- Applying median filter naively on the points in the bbox after convertint them into Spherical Coordinates ------- #
    #     ptc_xyz_camera_real_filtered = ptc_xyz_camera_real[mask] # Filtering the points
    #     print("Selected PointCloud:, \n", ptc_xyz_camera_real_px)
    #     print("Seleced Poitncloud ptc_xyz_camera_real:\n", ptc_xyz_camera_real)
    #     ptc_sph_camera_real_filtered = self.xyz2spherical(ptc_xyz_camera_real_filtered)

    #     # Find the indices of the rows where the element with the median value occurs
    #     print("Selected PointCloud in spherical:\n", ptc_sph_camera_real_filtered)
    #     median_r_idx = np.argsort(ptc_sph_camera_real_filtered[:, 2])[len(ptc_sph_camera_real_filtered)//2]
    #     # print(median_r_idx)
    #     median_sph_point = ptc_sph_camera_real_filtered[median_r_idx]
    #     # print('ptc_sph_camera: ', ptc_sph_camera.shape)
    #     # median_r = np.median(ptc_sph_camera[:, 2])
    #     # print('median_r: ', median_r)
    #     # indices = np.argwhere(ptc_sph_camera[:, 2] == median_r)
    #     # print('indices: ', indices)
    #     # median_sph_point = ptc_sph_camera[indices]
    #     # print('median_sph_point: ', median_sph_point)

    #     # Converting the point into xyz_cam_frame again
    #     print("median_sph_point: ", median_sph_point)
    #     median_xyz_camera = self.spherical2xyz([median_sph_point])[0]
    #     print("median_xyz_camera: ", median_xyz_camera)

    #     # ---------------------------------------------------------------------------- #
    #     #                   Creating a Marker Object to be published                   #
    #     # ---------------------------------------------------------------------------- #
    #     # marker = Marker()
    #     # # marker.ns = "Cylinder" # unique ID
    #     # marker.header.stamp = self.get_clock().now().to_msg()
    #     # marker.action = Marker().ADD
    #     # marker.type = Marker().CYLINDER
    #     # marker.header.frame_id = 'vimba_front_left_center'
    #     # marker.lifetime.sec = 1
    #     # marker.id = 100
    #     # marker.scale.x = 0.5  # diameter of cylinder
    #     # marker.scale.y = 0.5  # diameter of cylinder
    #     # marker.scale.z = 10.0  # length of cylinder
    #     # marker.pose.position.x = 1.0#median_xyz_camera[0]
    #     # marker.pose.position.y = 1.0#median_xyz_camera[1]
    #     # marker.pose.position.z = 0.0#median_xyz_camera[2]
    #     # marker.color.a = 1.0 
    #     # marker.color.r = 0.0
    #     # marker.color.g = 1.0
    #     # marker.color.b = 0.0

    #     marker_msg = Marker()
    #     marker_msg.header.frame_id = "vimba_front_left_center"
    #     marker_msg.header.stamp = self.point_cloud_msg.header.stamp
    #     marker_msg.ns = "Lidar_detection"
    #     marker_msg.id = 0
    #     marker_msg.type = Marker().CYLINDER
    #     marker_msg.action = 0
    #     marker_msg.pose.position.x = median_xyz_camera[0]
    #     marker_msg.pose.position.y = median_xyz_camera[1]
    #     marker_msg.pose.position.z = median_xyz_camera[2]
    #     marker_msg.pose.orientation.x = -1.0
    #     # marker_msg.pose.orientation.w = 0.0
    #     marker_msg.scale.x = 1.0
    #     marker_msg.scale.y = 1.0
    #     marker_msg.scale.z = 10.0
    #     marker_msg.color.a = 0.5
    #     marker_msg.color.g = 1.0
    #     marker_msg.lifetime = Duration(sec=0, nanosec=400000000)

    #     self.marker_pub.publish(marker_msg)

    #     # ---------------------------------------------------------------------------- #
    #     #    Setting the buffers to None to wait for the next image-pointcloud pair    #
    #     # ---------------------------------------------------------------------------- #
    #     self.image_msg = None
    #     self.point_cloud_msg = None 
        
    
    # Helper function that convert list of boxes to a matrix
    def boxes_to_matirx(self, boxes): 
        '''
        Helper function that convert list of boxes to a matrix
        '''
        mat_return = np.empty((3, 0))
        for box_msg in boxes:
            # print("Infunc: ", box_msg)
            top, bot, left, right = self.box_to_corners(box_msg.center.x, box_msg.center.y, box_msg.size_x, box_msg.size_y)
            mat = np.array([[left, right, right, left], 
                            [top,  top,   bot,   bot], 
                            [1 , 1 , 1 , 1 ]]) # (4x3 camera corner matrix)
            # print("Infunc: ", mat)
            mat_return = np.hstack((mat_return, mat))
            
        return mat_return

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
        print(self.bbox_msg)
        camera_corners_cam = K_inv @ self.boxes_to_matirx([self.bbox_msg])
        print(camera_corners_cam)
        
        #2
        # Apply Inverse rotation matrix
        R_inv = inv(self.RotMat_luminar_front2_flc) 
        numboxes = 1
        translation_stacked = np.tile(self.translation_luminar_front2_flc.reshape((3, 1)), 4*numboxes)
        camera_corners_lid = R_inv @ camera_corners_cam - translation_stacked

        #3
        # Normalize all points on their Z-axis
        ptc_numpy_record = pointcloud2_to_array(self.point_cloud_msg)
        ptc_xyz_lidar = get_xyz_points(ptc_numpy_record)
        ptc_xyz_lidar_normed = ptc_xyz_lidar / ptc_xyz_lidar[:, 2].reshape(-1, 1)
        # ptc_z_camera = ptc_xyz_camera_real_px[:, 2]
        # np.divide(ptc_xyz_camera_real_px, ptc_z_camera.reshape(ptc_xyz_camera_real_px.shape[0], 1))

        #4
        # Capture all points within the bounding box
        for i in camera_corners_lid.shape[1]: #(Columns as size)
            offset = 3*i
            mask = (mask | ((ptc_xyz_lidar_normed[:,0]>=camera_corners_lid[0,offset]) & # x>=left
                            (ptc_xyz_lidar_normed[:,0]<=camera_corners_lid[0,offset + 1]) & # x<=right
                            (ptc_xyz_lidar_normed[:,1]>=camera_corners_lid[1,offset + 1]) & #y>=top
                            (ptc_xyz_lidar_normed[:,1]<=camera_corners_lid[1,offset + 2]))) #y<=bottom
                            # Space for Optimization here
            
        # mask=(ptc_xyz_camera_normed[:,0]>=x1)&(ptc_xyz_camera_normed[:,0]<=x2)&(ptc_xyz_camera_normed[:,1]>=y1)&(ptc_xyz_camera_normed[:,1]<=y2)
        ptc_xyz_lidar_filtered = ptc_xyz_lidar[mask]

        # ---------------------------------------------------------------------------- #
        #                                   Custering                                  #
        # ---------------------------------------------------------------------------- #
        ptc_xyz_camera_list=ptc_xyz_lidar_filtered
        median_xyz_camera=[]
        o3d_pcd: o3d.geometry.PointCloud = o3d.geometry.PointCloud(
            o3d.utility.Vector3dVector(ptc_xyz_camera_list)
        )
        labels = np.array(
            o3d_pcd.cluster_dbscan(
                eps=2, min_points=40, print_progress=False
            )
        )
        max_label = labels.max()
        print(max_label)
        for i in range(max_label+1):
            print(np.where(labels==i)[0].shape)
            cluster=o3d_pcd.select_by_index(list(np.where(labels==i)[0]))

            cluster_bbox=cluster.get_axis_aligned_bounding_box()
            print(len(cluster_bbox.get_point_indices_within_bounding_box(o3d.utility.Vector3dVector(ptc_xyz_camera_list))))
            median_xyz_camera.append(cluster_bbox)
            print(cluster_bbox)
 

        # ---------------------------------------------------------------------------- #
        #                   Creating a Marker Object to be published                   #
        # ---------------------------------------------------------------------------- #

        for cluster_bbox in median_xyz_camera:
            center_max=cluster_bbox.get_max_bound()
            center_min=cluster_bbox.get_min_bound()
            center = cluster_bbox.get_center()
            extent=cluster_bbox.get_extent()
            print(center,extent)
            center=[center[0],center[1]+extent[1]*1.25,center[2]]

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
            # marker_msg.pose.orientation.x = rotation[0]
            # marker_msg.pose.orientation.y = rotation[1]
            # marker_msg.pose.orientation.z = rotation[2]
            # marker_msg.pose.orientation.w = rotation[3]
            marker_msg.scale.x = -extent[0]
            marker_msg.scale.y = extent[1]
            marker_msg.scale.z = -extent[2]
            marker_msg.color.a = 0.5
            marker_msg.color.r = 1.0
            marker_msg.lifetime = Duration(sec=0, nanosec=400000000)

            self.marker_pub.publish(marker_msg)

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
