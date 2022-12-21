import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

from vision_msgs.msg import Detection3D, BoundingBox3D, BoundingBox2D 
from builtin_interfaces.msg import Duration
from visualization_msgs.msg import Marker
from sensor_msg.msgs import PointCloud2
from sensor_msg.msgs import Image
from message_filters import ApproximateTimeSynchronizer, TimeSynchronizer, Subscriber
from sensor_msg.msgs import CameraInfo
from sensor_msgs_py import pc2

import cv2

class Lidar2Cam(Node):
    def __init__(self):
        super().__init__('lidar_to_cam_node')


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
            topic       = '/luminar_front_points/points_raw',
            callback    = self.pc2_callback,
            qos_profile = self.qos_profile
        )
        self.pointcloud_sub  # prevent unused variable warning

        # ---------------------------------- CAMERA ---------------------------------- #

        self.marker_pub = self.create_publisher(PointCloud2, "/lidar2cam_ptc", rclpy.qos.qos_profile_sensor_data)
        
        # self.ptc_sub    = Subscriber(self, PointCloud2, "/luminar_front_points/points_raw/ground_filtered", self.qos_profile)
        self.ptc_raw_sub    = Subscriber(self, PointCloud2, "/luminar_front_points/points_raw", self.callback, self.qos_profile)
        

        self.rate = self.create_rate(1) #2Hz
        self.counter = 0


    def callback(self, ptc):
        while(True):
            if (self.counter == 20):
                print("Published!")
                ptc_numpy = pc2.read_points_numpy(ptc)
                print(ptc_numpy)
                # self.marker_pub.publish(ptc_numpy)
                self.counter += 1
            self.rate.sleep()




# class Lidar2Cam(Node):
#     def __init__(self):
#         super().__init__('Lidar2Cam_node')

#         # -------------------------------- QOS Profile ------------------------------- #
#         self.qos_profile =  QoSProfile(
#             reliability=QoSReliabilityPolicy.BEST_EFFORT,
#             history=QoSHistoryPolicy.KEEP_LAST,
#             depth=1,
#         )

#         # --------------------------- Sensor Subscriptions --------------------------- #
#         # ----------------------------------- LIDAR ---------------------------------- #
#         '''
#         Topics: that use: sensor_msgs/msg/PointCloud2
#         /luminar_front_points/points_raw/ground_filtered
#         /luminar_left_points
#         /debug/clusters
#         /luminar_front_points/filtered
#         /luminar_left_points/filtered
#         /luminar_right_points/filtered
#         /luminar_front_points
#         /luminar_left_points
#         /luminar_right_points
#         /points_raw/ego_cropped
#         /points_raw/cropped
#         /points_raw/outside_poly_removed
#         /points_raw/inside_poly_removed
#         /luminar_front_points/points_raw/ground_filtered
#         /luminar_left_points/points_raw/ground_filtered
#         /luminar_right_points/points_raw/ground_filtered
#         /points_raw/concatenated

#         Topics: that use: sensor_msgs/msg/Image
#         /vimba_front_left_center/image
#         /vimba_front_right_center/image

#         Topics: that use sensor_msgs/msg/CameraInfo
#         /vimba_front_left_center/camera_info
#         /vimba_front_right_center/camera_info
#         '''

#         # ------------------------- Sensor Subscriptions Start ------------------------- #
#         # ------------------------- Synchronizer code start ------------------------- #
#         self.image_left_sub  = Subscriber(self, Image, "/vimba_front_left_center/image", self.qos_profile)
#         self.image_right_sub = Subscriber(self, Image, "/vimba_front_right_center/image", self.qos_profile)
#         self.ptc_sub         = Subscriber(self, PointCloud2, "/luminar_front_points/points_raw/ground_filtered", self.qos_profile)
        
#         queue_size = 30

#         # you can use ApproximateTimeSynchronizer if msgs dont have exactly the same timestamp
#         self.ts = ApproximateTimeSynchronizer(
#             [self.image_left_sub, self.image_right_sub , self.ptc_sub],
#             queue_size,
#             0.05,  # defines the delay (in seconds) with which messages can be synchronized
#         )

#         # ------------------------- Publisher Start ------------------------- #
#         self.marker_pub = self.create_publisher(PointCloud2, "lidar2cam_detection_marker", rclpy.qos.qos_profile_sensor_data)
#         # ------------------------- Publisher End ------------------------- #

#         # ------------------------- Preprocessing things Start ------------------------- #
#         self.w  = 516
#         self.h = 384
#         # Get the frame of the LIDAR and cameras

#         # TODO
#         # ros2 run tf2_tools view_frames.py
#         # ros2 run tf2_ros tf2_echo [reference_frame] [target_frame]

#         # Calculate the g matrices
#         self.g_l
#         self.g_r
#         # Create Camera object
#         newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx,dist,(self.w,self.h),1,(self.w,self.h))


#     def lidar_callback(self, image_left, image_right, ptc):
        
#         # Extract the point cloud
#         pt_clouds = ptc.read_points() #read_points_numpy()

#         # Convert {X_s, Y_s, Z_s}_LIDAR into {X_s, Y_s, Z_s}_[L_cam, R_cam] 
#         X_cl = self.g_l @ pt_clouds
#         X_cr = self.g_r @ pt_clouds
#         # ros2 bag play <name of bag>
#         # https://github.com/Box-Robotics/ros2_numpy
#         # msg = ros2_numpy.msgify(PointCloud2, data)

#         # Do the transformation from {X_s, Y_s, Z_s}_[L_cam, R_cam] into {x_c, y_c}_[L_cam, R_cam]
#         dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
        
#         # Determine if it is in the bounding box

#             # If it is, add it to a point cloud object 
            

# class RadarToCamConverter(Node):
 
#     def __init__(self):
#         super().__init__('radar_to_cam_node')

#         # -------------------------------- QOS Profile ------------------------------- #
#         self.qos_profile =  QoSProfile(
#             reliability=QoSReliabilityPolicy.BEST_EFFORT,
#             history=QoSHistoryPolicy.KEEP_LAST,
#             depth=1,
#         )

#         # --------------------------- Sensor Subscriptions --------------------------- #
#         # ----------------------------------- RADAR ---------------------------------- #
#         self.radar_sub = self.create_subscription(
#             msg_type    = EsrTrack,
#             topic       = '/radar_front/esr_track',
#             callback    = self.radar_callback,
#             qos_profile = self.qos_profile
#         )
#         self.radar_sub  # prevent unused variable warning

#         # ---------------------------------- GPS VEL --------------------------------- #
#         self.gps_vel_sub    = self.create_subscription(
#             msg_type        = BESTVEL,
#             topic           = '/novatel_top/bestgnssvel',
#             callback        = self.gps_callback,
#             qos_profile     = self.qos_profile
#         )
#         self.gps_vel_sub # prevent unused variable warning

#         # ------------------------- Sensor Subscriptions END ------------------------- #

#         # ----------------------------- Marker Publisher ----------------------------- #
#         self.marker_pub = self.create_publisher(Marker, "radar2cam_detection_marker", rclpy.qos.qos_profile_sensor_data)
#         # Autoware Detected Object Publisher
#         self.det_pub    = self.create_publisher(DetectedObject, "radar2cam_detected_object", rclpy.qos.qos_profile_sensor_data)

#         # ---------------------- Initialize Required Attributes ---------------------- #
#         self.ego_vel    = None
#         self.pre_pos = None

#     def gps_callback(self, msg):
#         self.ego_vel = np.array([
#             # Horizontal speed over ground, in meters per second
#             msg.hor_speed,
#             # Vertical speed, in meters per second, where positive values indicate 
#             # increasing altitude (up) and negative values indicate decreasing altitude (down)
#             msg.ver_speed, 
#         ])

#         detection = RadarDetection(msg)
#         if detection.get_valid_preproc() and self.ego_vel is not None:
#             # -------------------------- Get Cartesian Position -------------------------- #
#             cartesian_pos = detection.get_cartesian_pos()
#             # ------------------------------ Get Kinematics ------------------------------ #
#             det_speed_wrt_ego   = detection.get_speed() # Unused here but used internally by detection

#             # -------------------------- Check if static object -------------------------- #
#             is_static           = detection.get_static(np.abs(self.ego_vel[0])) # Unused here but used internally by detection
#             if detection.get_valid_postproc():
#                 # if self.pre_pos:
#                 #     time_diff = abs(msg.header.stamp.sec + msg.header.stamp.nanosec * 10**(-9) - self.pre_pos[0])
#                 #     moved_dist = (self.pre_pos[1] - cartesian_pos[0])**2 + (self.pre_pos[2] - cartesian_pos[1])**2
#                 #     print(self.pre_pos[1], self.pre_pos[2], cartesian_pos[:2], moved_dist)
#                 #     print(time_diff, moved_dist)
#                 # if self.pre_pos and time_diff < 0.3 and moved_dist >= 64 :
#                 #     #print(self.pre_pos[1], self.pre_po time_diff < 0.5 and moved_dist >= 100 :
#                 #     #print(self.pre_pos[1], self.pre_ps[2], cartesian_pos[:2], moved_dist)
#                 #     print(f"same time but too far {moved_dist}")
#                 #     self.pre_pos = (msg.header.stamp.sec + msg.header.stamp.nanosec * 10**(-9), cartesian_pos[0], cartesian_pos[1])
#                 #     del detection
#                 #     return 
#                 # ---------------------- Prepare and publish marker msg ---------------------- #
#                 marker_msg = Marker()
#                 marker_msg.header.frame_id = "radar_front"
#                 marker_msg.header.stamp = msg.header.stamp
#                 marker_msg.ns = "radar_detection"
#                 marker_msg.id = 0
#                 marker_msg.type = 1
#                 marker_msg.action = 0
#                 marker_msg.pose.position.x = cartesian_pos[0]
#                 marker_msg.pose.position.y = cartesian_pos[1]
#                 marker_msg.pose.position.z = 0.0
#                 marker_msg.pose.orientation.w = 1.0
#                 marker_msg.scale.x = 2.9
#                 marker_msg.scale.y = 1.6
#                 marker_msg.scale.z = 1.0
#                 marker_msg.color.a = 1.0
#                 marker_msg.color.g = 1.0
#                 marker_msg.lifetime = Duration(sec=0, nanosec=400000000)
#                 self.marker_pub.publish(marker_msg)
#                 self.pre_pos = (msg.header.stamp.sec + msg.header.stamp.nanosec * 10**(-9), cartesian_pos[0], cartesian_pos[1])

#                 q = tf_transformations.quaternion_from_euler(0.0, 0.0, np.arctan(detection.cartesian_vel[1]/ (detection.cartesian_vel[0] + detection.epsilon) ))
#                 detection_msg = DetectedObject()
#                 detection_msg.header                = msg.header
#                 detection_msg.label                 = "CAR"
#                 detection_msg.valid                 = True
#                 detection_msg.score                 = 1.0
#                 detection_msg.pose.position.x       = cartesian_pos[0]
#                 detection_msg.pose.position.y       = cartesian_pos[1]
#                 detection_msg.pose.position.z       = 0.0
#                 detection_msg.pose.orientation      = Quaternion(*q)
#                 detection_msg.dimensions.x          = 2.9
#                 detection_msg.dimensions.y          = 1.6
#                 detection_msg.dimensions.z          = 1.2
#                 detection_msg.velocity.linear.x     = detection.cartesian_vel[0]
#                 detection_msg.velocity.linear.y     = detection.cartesian_vel[1]
#                 detection_msg.velocity.linear.z     = 0.0
#                 detection_msg.acceleration.linear.x = detection.range_acc * detection.cos_angle
#                 detection_msg.acceleration.linear.y = detection.range_acc * detection.sin_angle
#                 detection_msg.acceleration.linear.z = 0.0

#                 self.det_pub.publish(detection_msg)

#         # Cleanup
#         del detection
#         return       
        


def main(args=None):
    rclpy.init(args=args)

    minimal_subscriber = Lidar2Cam()

    rclpy.spin(minimal_subscriber)

    # Destroy the node explicitly
    minimal_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
