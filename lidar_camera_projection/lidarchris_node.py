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
print(os.getcwd())

class Lidar2Cam(Node):
    def __init__(self):
        print("Initialized Lidar2Cam Node!")
        super().__init__('lidar_to_cam_node')
        self.bridge = CvBridge()
        self.point_cloud_msg = None
        self.bbox_msg = None # Initialize an empty array

        # Front Left Camera only
        self.camera_info = np.array([[1732.571708, 0.000000, 549.797164], 
                                     [0.000000, 1731.274561, 295.484988], 
                                     [0.000000, 0.000000, 1.000000*2]])*0.5
        print(self.camera_info)

        # self.RotMat = np.array([[0.0, -1.0,  0.0],
        #                         [0.0,  0.0, -1.0],
        #                         [1.0,  0.0,  0.0]])
        # self.translation = np.array([0.007, -0.121, 0.026])*-1

        # ---------------------------------------------------------------------------- #
        #                      Calibrated Transformation Matrices                      #
        # ---------------------------------------------------------------------------- #
        self.translation = np.array([ 0.01585027 -0.15477238  0.02598614])
        self.RotMat = np.array([[ 0.02135093 -0.99976672 -0.00326259]
                                [ 0.05990699  0.00453683 -0.99819365]
                                [ 0.9979756   0.02111691  0.05998988]])

        # Multiply the rotation Matrices by K
        self.RotMat_K      = self.camera_info @ self.RotMat
        self.translation_K = self.camera_info @ self.translation

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

        # ------------------------------- Publishers ------------------------------ #
        self.image_pub = self.create_publisher(Image , "/vimba_front_left_center/image_lidar_label", rclpy.qos.qos_profile_sensor_data)
        self.cloud_pub = self.create_publisher(PointCloud2, "/luminar_front_points/points_filtered", rclpy.qos.qos_profile_sensor_data)

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
        print("Min Values:", np.min(ptc_xyz_lidar, axis=0))
        print("Max Values: ", np.max(ptc_xyz_lidar, axis=0))

        # --------------------------- Applying the Rotation -------------------------- # Alt + x
        # ---------------------------------------------------------------------------- # Alt + Y
        # ---------------------------------------------------------------------------- # Alt + shift + x
        #                                      sdf                                     #
        # ---------------------------------------------------------------------------- #
        RotMat_luminar_front2_flc = np.array([[0.0, -1.0,  0.0],
                                            [0.0,  0.0, -1.0],
                                            [1.0,  0.0,  0.0]])
        translation_luminar_front2_flc = np.array([0.121, -0.026, 0.007])
        translation_luminar_front2_flc = np.tile(translation_luminar_front2_flc.reshape((3, 1)), numpoints)
        assert(translation_luminar_front2_flc.shape == (3, numpoints)), "Translation is not 3 x N"
        ptc_xyz_camera = RotMat_luminar_front2_flc @ ptc_xyz_lidar.T + translation_luminar_front2_flc # This is correct
        ptc_xyz_camera = ptc_xyz_camera.T
        assert(ptc_xyz_camera.shape == (numpoints, 3)), "PointCloud_camera is not N x 3"
        print("\t")
        print("R_Min Vals:", np.min(ptc_xyz_camera, axis=0))
        print("R_Max Vals: ", np.max(ptc_xyz_camera, axis=0))


        # ------------------------- Applying the Camera Info ------------------------- #
        camera_info = np.array([[1732.571708, 0.000000, 549.797164], 
                                [0.000000, 1731.274561, 295.484988], 
                                [0.000000, 0.000000, 1.000000]])
        camera_info = np.array([[1732.571708*0.5, 0.000000, 549.797164*0.5], 
                            [0.000000, 1731.274561*0.5, 295.484988*0.5], 
                            [0.000000, 0.000000, 1.000000]])
        ptc_xyz_camera = ptc_xyz_camera.T
        ptc_xyz_camera = camera_info @ ptc_xyz_camera
        ptc_xyz_camera = ptc_xyz_camera.T
        print("\t")
        print("First Point: ", ptc_xyz_camera[0])
        print("K_NORM_RT_Min Vals:", np.min(ptc_xyz_camera, axis=0))
        print("K_NORM_RT_Max Vals: ", np.max(ptc_xyz_camera, axis=0))


        # ------------------------------ Do the Division ----------------------------- #
        ptc_z_camera = ptc_xyz_camera[:, 2]
        ptc_xyz_camera = np.divide(ptc_xyz_camera, ptc_z_camera.reshape(ptc_xyz_camera.shape[0], 1))
        print("\t")
        print("First Point: ", ptc_xyz_camera[0])
        print("NORM_RT_Min Vals:", np.min(ptc_xyz_camera, axis=0))
        print("NORM_RT_Max Vals: ", np.max(ptc_xyz_camera, axis=0))

        # ---------------------------------------------------------------------------- #
        #                    Point Comparison with BBox happens here                   #
        # ---------------------------------------------------------------------------- #
        y1, y2, x1, x2 = self.box_to_corners(self.bbox_msg.center.x, self.bbox_msg.center.y, self.bbox_msg.size_x, self.bbox_msg.size_y)
        mask=ptc_xyz_camera[:,0]>=x1&ptc_xyz_camera[:,0]<=x2&ptc_xyz_camera[:,1]>=y1&ptc_xyz_camera[:,1]<=y2
        ptc_numpy_record_filtered = ptc_numpy_record[mask]
        ptc_numpy_record_filtered_msg = array_to_pointcloud2(ptc_numpy_record_filtered, stamp=None, frame_id="luminar_front")

        # -------------------- Plotting everything into the Image -------------------- #
        # image = self.img_tocv2(self.image_msg) 
        # print("image_width, height:", image.shape)
        # image_undistorted = cv2.undistort(image, camera_info, np.array([-0.272455, 0.268395, -0.005054, 0.000391, 0.000000]))
        # border_size=300
        # image_undistorted=cv2.copyMakeBorder(image_undistorted,border_size,border_size,border_size,border_size,cv2.BORDER_CONSTANT,None,0)
        # for i in ptc_xyz_camera:
        #     image_undistorted = cv2.circle(image_undistorted, (int(i[0]+border_size), int(i[1])+border_size), 1, (0, 0, 255), 2)
            # image_undistorted = cv2.circle(image_undistorted, (int(i[0]), int(i[1])), 1, (0, 0, 255), 1)

        # Publishing the Image and PointCloud
        # self.image_pub.publish(self.bridge.cv2_to_imgmsg(image_undistorted))
        # self.image_pub.publish(self.bridge.cv2_to_imgmsg(image[:2*border_size,:2*border_size]))
        self.cloud_pub.publish(filteredptc_numpy_record_filtered_msg_msg)

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
