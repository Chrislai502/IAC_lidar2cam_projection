import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

from vision_msgs.msg import Detection3D, BoundingBox3D, BoundingBox2D
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
from ai import cs
print(os.getcwd())

class Lidar2Cam(Node):
    def __init__(self):
        print("Initialized Lidar2Cam Node!")
        super().__init__('lidar_to_cam_node')
        self.bridge = CvBridge()
        self.point_cloud_msg = None
        self.bbox_msg = None # Initialize an empty array
        self.img_msg = None

        # Front Left Camera only
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

        # # Multiply the rotation Matrices by K
        # self.RotMat_K      = self.camera_info @ self.RotMat
        # self.translation_K = self.camera_info @ self.translation

        # -------------------------------- QOS Profile ------------------------------- #
        self.qos_profile =  QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # ------------------------------- Subscriptions ------------------------------ #
        # ----------------------------------- LIDAR ---------------------------------- #
        self.pointcloud_sub = self.create_subscription(
            msg_type    = PointCloud2,
            topic       = '/luminar_front_points',
            callback    = self.point_cloud_callback,
            qos_profile = self.qos_profile
        )
        self.pointcloud_sub  # prevent unused variable warning

        # ------------------------------  Bounding Box ---------------------------------- #
        self.image_sub = self.create_subscription(
            msg_type    = BoundingBox2D, 
            topic       = 'vimba_front_left_center/out/objects',
            callback    = self.bbox_callback,
            qos_profile = self.qos_profile
        )
        self.image_sub # prevent unused variable warning
        # ------------------------------  Image ---------------------------------- #
        self.image_sub = self.create_subscription(
            msg_type    = Image,
            topic       = 'vimba_front_left_center/out/image',
            callback    = self.img_callback,
            qos_profile = self.qos_profile
        )
        self.image_sub # prevent unused variable warning

        # ------------------------------- Publishers ------------------------------ #
        self.image_pub = self.create_publisher(Image , "/Lidar_filtered_label", rclpy.qos.qos_profile_sensor_data)
        self.cloud_pub = self.create_publisher(PointCloud2, "/filteredPointcloud", rclpy.qos.qos_profile_sensor_data)
        self.marker_pub = self.create_publisher(Marker, '/Lidar_car_marker', rclpy.qos.qos_profile_sensor_data)

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

    '''Helper Function'''
    def img_tocv2(self, message):
        try:
            # Convert the ROS2 Image message to a NumPy array
            image = self.bridge.imgmsg_to_cv2(message, "bgr8")
        except CvBridgeError as e:
            print(e)
        return image

    # ---------------------------------------------------------------------------- #
    #       Func converts pointclouds to spherical Coordinates and vice-versa      #
    # ---------------------------------------------------------------------------- #
    # Output points are in degrees
    def xyz2spherical(self, ptc_arr):
        # Transformation of cartesian point cloud array to spherical point cloud array
        spherical_pcd_points = np.zeros(ptc_arr.shape)
        points_out_of_cam_perspective = []
        for i in range (0, ptc_arr.shape[0]):
            # Cartesian --> spherical (theta_x, theta_y, r), where theta_x and theta_y are in degrees and are in the frame of OpenCV?(likely)
            spherical_pcd_points[i, 0], spherical_pcd_points[i, 1], spherical_pcd_points[i, 2] = cs.cart2sp(ptc_arr[i,0],ptc_arr[i,1], ptc_arr[i,2])  #xyz --> r theta phi
        return spherical_pcd_points # Spherical or XYZ?

    def spherical2xyz(self, spr_arr):
        # Transformation of cartesian point cloud array to spherical point cloud array
        dimensions = spr_arr.shape
        cartesian_pcd_points_cam_perspective = np.zeros(spr_arr.shape)
        for i in range (0, dimensions[0]):
            # Cartesian --> spherical
            cartesian_pcd_points_cam_perspective[i, 0], cartesian_pcd_points_cam_perspective[i, 1], cartesian_pcd_points_cam_perspective[i, 2] = \
            cs.sp2cart(spr_arr[i,0],spr_arr[i,1], spr_arr[i,2])  #xyz --> r theta phi
        return cartesian_pcd_points_cam_perspective
    
    def box_to_corners(self, cx, cy, width, height):
        half_width = width / 2
        half_height = height / 2

        y1 = cy - half_height # top
        y2 = cy + half_height # bottom
        x1 = cx - half_width # left
        x2 = cx + half_width # right

        return (y1, y2, x1, x2) #(up, down, left, right)

    ''' 
    In the middle of debugging the LIDAR
    '''
    def bbox_ptc_callback(self):

        # -------------------- Lidar camera projection right here -------------------- #
        ptc_numpy_record = pointcloud2_to_array(self.point_cloud_msg)
        ptc_xyz_lidar = get_xyz_points(ptc_numpy_record)
        numpoints = ptc_xyz_lidar.shape[0]
        assert(ptc_xyz_lidar.shape[1] == 3), "PointCloud_lidar is not N x 3"
   

        # --------------------------- Applying the Rotation -------------------------- # Alt + x
        # ---------------------------------------------------------------------------- # Alt + Y
        # ---------------------------------------------------------------------------- # Alt + shift + x
        #                                      sdf                                     #
        # ---------------------------------------------------------------------------- #
        translation_luminar_front2_flc = np.tile(self.translation_luminar_front2_flc.reshape((3, 1)), numpoints)
        assert(translation_luminar_front2_flc.shape == (3, numpoints)), "Translation is not 3 x N"
        ptc_xyz_camera = self.RotMat_luminar_front2_flc @ ptc_xyz_lidar.T + translation_luminar_front2_flc # This is correct
        ptc_xyz_camera = ptc_xyz_camera.T
        assert(ptc_xyz_camera.shape == (numpoints, 3)), "PointCloud_camera is not N x 3"


        # ------------------------- Applying the Camera Info ------------------------- #
        ptc_xyz_camera = ptc_xyz_camera.T
        ptc_xyz_camera = self.camera_info @ ptc_xyz_camera
        ptc_xyz_camera = ptc_xyz_camera.T


        # ------------------------------ Do the Division ----------------------------- #
        ptc_z_camera = ptc_xyz_camera[:, 2]
        ptc_xyz_camera = np.divide(ptc_xyz_camera, ptc_z_camera.reshape(ptc_xyz_camera.shape[0], 1))


        # ---------------------------------------------------------------------------- #
        #                    Point Comparison with BBox happens here                   #
        # ---------------------------------------------------------------------------- #
        
        # ------------------- Convert the bbox msg to (top, bottom, left, right) bounds ------------------ #
        y1, y2, x1, x2 = self.box_to_corners(self.bbox_msg.center.x, self.bbox_msg.center.y, self.bbox_msg.size_x, self.bbox_msg.size_y)
        
        # ----- Applying detection strategy to cut off top 40% of the bbox ----- #
        y1 = y1 + int(0.4 * self.bbox_msg.size_y)
        mask=(ptc_xyz_camera[:,0]>=x1)&(ptc_xyz_camera[:,0]<=x2)&(ptc_xyz_camera[:,1]>=y1)&(ptc_xyz_camera[:,1]<=y2)
        
        # -------- Applying median filter naively on the points in the bbox after convertint them into Spherical Coordinates ------- #
        ptc_xyz_camera = ptc_xyz_camera[mask] # Filtering the points
        ptc_sph_camera = self.xyz2spherical(ptc_xyz_camera)
        # Find the indices of the rows where the element with the median value occurs
        print("ptc_sph_camera: ", ptc_sph_camera)
        median_r = np.median(ptc_sph_camera[:, 2])
        indices = np.argwhere(ptc_sph_camera[:, 2] == median_r)
        median_sph_point = ptc_sph_camera[indices]
        # Converting the point into xyz_cam_frame again
        median_xyz_camera = self.spherical2xyz(median_sph_point)

        # ---------------------------------------------------------------------------- #
        #                   Creating a Marker Object to be published                   #
        # ---------------------------------------------------------------------------- #
        marker = Marker()
        marker.header.frame_id = 'camera_front_left_center'
        marker.type = Marker.CYLINDER
        # marker.scale.x = 0.1  # diameter of cylinder
        # marker.scale.y = 0.1  # diameter of cylinder
        # marker.scale.z = 0.2  # length of cylinder
        marker.pose.position.x = median_xyz_camera[0]
        marker.pose.position.y = median_xyz_camera[1]
        marker.pose.position.z = median_xyz_camera[2]

        self.publisher_.publish(marker)

        # # ------ Applying 2nd strategy to remove outliers, and select the median within 10~30th percentile----- #
        # # ------- Idea: Get the correct distance of the car from our car first, ------ #
        # # ---- in spherical coordinates and then decide where to place the marker ---- #
        # # ------- (in terms of theta/ x, y,z) What shape of the object needed? ------- #
        # ptc_xyz_camera = ptc_xyz_camera[mask] # Filtering the points
        
        # # Converting the Markers from XYZ Coordinates to Spherical coordinares

        # # Select the last column
        # values = ptc_xyz_camera[:, 2]

        # # Calculate the 10th and 40th percentiles
        # p10 = np.percentile(values, 10)
        # p30 = np.percentile(values, 30)

        # # Use a boolean mask to filter the points that lie outside the 10th to 40th percentile
        # mask = (values < p10) | (values > p30)
        # filtered_points = ptc_xyz_camera[mask]

        # # Calculate the median of the filtered points
        # median = np.median(filtered_points[:, 2])

        # # Find the indices of the rows where the element with the median value occurs
        # indices = np.argwhere(filtered_points[:, 2] == median)
        # print(indices)

        # # Index into the original array using the indices and return the corresponding rows
        # median_rows = filtered_points[indices]

        # # ---- Applying 3rd strategy, take 10th to 40th percentile's median value ---- #
        # ptc_numpy_record_filtered = ptc_numpy_record[mask]
        # print("ptc_numpy_record_filtered: ", ptc_numpy_record_filtered.shape)
        # ptc_numpy_record_filtered_msg = array_to_pointcloud2(ptc_numpy_record_filtered, stamp=None, frame_id="luminar_front")

        # -------------------- Plotting everything into the Image -------------------- #
        image = self.img_tocv2(self.img_msg) 
        # print("image_width, height:", image.shape)
        # image = cv2.undistort(image, self.camera_info, np.array([-0.272455, 0.268395, -0.005054, 0.000391, 0.000000]))
        ptc_xyz_camera = ptc_xyz_camera[mask]
        
        # Doing Simple Data Processing to get the location where Marker will be placed

        
        border_size=300
        image_undistorted=cv2.copyMakeBorder(image,border_size,border_size,border_size,border_size,cv2.BORDER_CONSTANT,None,0)

        z_min=np.min(ptc_z_camera)
        z_range=np.max(ptc_z_camera)-z_min
        ptc_z_camera=(ptc_z_camera-z_min)*255/z_range
        ptc_z_camera=ptc_z_camera.astype(np.uint8)
        color=cv2.applyColorMap(ptc_z_camera[:,np.newaxis],cv2.COLORMAP_HSV)
        r=ptc_xyz_camera.shape[0]
        print(r)
        for j in range(r):
            i=ptc_xyz_camera[j]
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
        self.cloud_pub.publish(ptc_numpy_record_filtered_msg)

        # # Pause the code
        # try:
        #     # Start an infinite loop
        #     while True:
        #         # Sleep for one second
        #         time.sleep(1)
        # except KeyboardInterrupt:
        #     # Handle the KeyboardInterrupt exception
        #     print("Exiting loop")

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
