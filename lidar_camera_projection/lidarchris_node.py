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
        self.image_msg = None # Initialize an empty array

        # Front Left Camera only
        self.camera_info = np.array([[1.732571708, 0.000000, 0.549797164], 
                                     [0.000000, 1.731274561, 0.295484988], 
                                     [0.000000, 0.000000, 1.000000]])
        self.RotMat = np.array([[0.0, -1.0,  0.0],
                                [0.0,  0.0, -1.0],
                                [1.0,  0.0,  0.0]])
        self.translation = np.array([0.007, -0.121, 0.026])*-1

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

        # ------------------------------  Unlabelled Image ---------------------------------- #
        self.image_sub = self.create_subscription(
            msg_type    = Image, 
            topic       = '/vimba_front_left_center/image',
            callback    = self.image_callback,
            qos_profile = self.qos_profile
        )
        self.image_sub # prevent unused variable warning

        # ------------------------------- Publishers ------------------------------ #
        self.image_pub = self.create_publisher(Image , "/vimba_front_left_center/image_lidar_label", rclpy.qos.qos_profile_sensor_data)
        self.image_pub2 = self.create_publisher(Image , "/vimba_front_left_center/image_lidar_label2", rclpy.qos.qos_profile_sensor_data)
        # self.cloud_pub = self.create_publisher(PointCloud2, "/luminar_front_points/points_filtered", rclpy.qos.qos_profile_sensor_data)

    def image_callback(self, msg):
        print("I")
        self.image_msg = msg
        if self.point_cloud_msg is not None:
            print("+")
            self.image_ptc_callback()

    def point_cloud_callback(self, msg):
        print("P")
        self.point_cloud_msg = msg
        if self.image_msg is not None:
            print("+")
            self.image_ptc_callback()

    '''Helper Function'''
    def img_tocv2(self, message):
        try:
            # Convert the ROS2 Image message to a NumPy array
            image = self.bridge.imgmsg_to_cv2(message, "bgr8")
        except CvBridgeError as e:
            print(e)
        return image

    ''' 
    In the middle of debugging the LIDAR
    '''
    def image_ptc_callback(self):


        # -------------------- Lidar camera projection right here -------------------- #
        ptc_numpy_record = pointcloud2_to_array(self.point_cloud_msg)
        ptc_xyz_lidar = get_xyz_points(ptc_numpy_record)
        numpoints = ptc_xyz_lidar.shape[0]
        assert(ptc_xyz_lidar.shape[1] == 3), "PointCloud_lidar is not N x 3"
        # np.save(os.path.join(os.getcwd(), "pointcloud_raw.txt"), ptc_numpy)
        # np.savetxt(os.path.join(os.getcwd(), "pointcloud_raw.txt"), ptc_numpy, fmt="%d", delimiter=",")
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
        translation_luminar_front2_flc2 = np.array([0.121, -0.026, 0.007])*2
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

        # -------------------- Plotting everything into the Image -------------------- #
        image = self.img_tocv2(self.image_msg) 
        image_undistorted = cv2.undistort(image, camera_info, np.array([-0.272455, 0.268395, -0.005054, 0.000391, 0.000000]))
        border_size=300
        image_undistorted=cv2.copyMakeBorder(image_undistorted,border_size,border_size,border_size,border_size,cv2.BORDER_CONSTANT,None,0)
        for i in ptc_xyz_camera:
            image_undistorted = cv2.circle(image_undistorted, (int(i[0]+border_size), int(i[1])+border_size), 1, (0, 0, 255), 2)
            # image_undistorted = cv2.circle(image_undistorted, (int(i[0]), int(i[1])), 1, (0, 0, 255), 1)

        # Publishing the Image and PointCloud
        self.image_pub.publish(self.bridge.cv2_to_imgmsg(image_undistorted))
        self.image_pub2.publish(self.bridge.cv2_to_imgmsg(image_undistorted))
        # self.image_pub.publish(self.bridge.cv2_to_imgmsg(image[:2*border_size,:2*border_size]))
        # self.cloud_pub.publish(filtered_msg)

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

        # ptc_numpy[:,[0,1,2]]*=-1
        # Applying Rotation 
        # ptc_numpy=ptc_numpy[:,[1,2,0]] 
        # ptc_numpy[:, 0:2] = -ptc_numpy[:, 0:2]
        
        # ptc_numpy_c = self.camera_info @ ptc_numpy.T + np.tile(self.translation_K.reshape((3, 1)), ptc_numpy.shape[0]) # could we incorprate the camera matrix into the Rotation transformation
        # # ptc_numpy_c = np.divide(ptc_numpy_c[:, :2], ptc_numpy_c[:, 2:3])
        # ptc_numpy_c = ptc_numpy_c.T
        # # Redo this
        # # ptc_numpy_c=ptc_numpy_c[(ptc_numpy_c[:,0]>=0)&(ptc_numpy_c[:,0]<=384)&(ptc_numpy_c[:,1]>=0)&(ptc_numpy_c[:,1]<=516)]
        # # print(ptc_numpy_c.shape,np.min(ptc_numpy_c,0),np.max(ptc_numpy_c,0))
        # ptc_numpy_l = ptc_numpy_rec # Should already be in the right frame
        # filtered_msg = array_to_pointcloud2(ptc_numpy_l, frame_id="luminar_front")

        # # Labelling the Image for verification Purposes
        # ptc_numpy_l_xyz = ptc_numpy[mask]
        # # print(ptc_numpy_l_xyz)
        # image = self.img_tocv2(self.image_msg) #(384, 516, 3)
        # border_size=300
        # image=cv2.copyMakeBorder(image,border_size,border_size,border_size,border_size,cv2.BORDER_CONSTANT,None,0)
        # # print(type(image))
        # # ptc_numpy_c=ptc_numpy_c[:,[0,2,1]]
        # for i in ptc_numpy_c:
        #     # print(i)

        #     image = cv2.circle(image, (int(i[0]+border_size), int(i[1])+border_size), 1, (0, 0, 255), 1)
        # # cv2.imshow(image)
        
        # # Publishing the Image and PointCloud
        # self.image_pub.publish(self.bridge.cv2_to_imgmsg(image[:2*border_size,:2*border_size]))
        # self.cloud_pub.publish(filtered_msg)

        # self.box_msg = None
        # self.point_cloud_msg = None 


# class Lidar2Cam(Node):
#     def __init__(self):
#         print("Initialized Lidar2Cam Node!")
#         super().__init__('lidar_to_cam_node')
#         self.bridge = CvBridge()
#         self.point_cloud_msg = None
#         self.image_msg = None # Initialize an empty array
#         self.box_msg = None

#         # Front Left Camera only
#         # self.camera_info     = np.array([[494.186152, 0.000000, 521.646677],
#         #                                 [0.000000, 494.988972, 389.133925],
#         #                                 [0.000000, 0.000000, 1.000000]])  
#         # self.camera_info     = np.array([[494.186152, 0.000000, 52.1*1],
#         #                                 [0.000000, 494.988972, 38.9*1],
#         #                                 [0.000000, 0.000000, 1]]) 
#         # self.camera_info     = np.array([[494.186152, 0.000000, 0],
#         #                                 [0.000000, 494.186152, 0],
#         #                                 [0.000000, 0.000000, 1]]) 
#         self.camera_info = np.array([[1.732571708, 0.000000, 0.549797164], 
#                                      [0.000000, 1.731274561, 0.295484988], 
#                                      [0.000000, 0.000000, 1.000000]])
#         # self.camera_info_inv = np.linalg.inv(np.array([[494.186152, 0.000000, 521.646677],
#         #                                                 [0.000000, 494.988972, 389.133925],
#         #                                                 [0.000000, 0.000000, 1.000000]]))
#         # self.RotMat = np.array([[  0.5732029, 0.8194135,  0.0000000],
#         #                         [-0.8194135,  0.5732029,  0.0000000],
#         #                         [0.0000000,  0.0000000,  1.0000000] ])
#         self.RotMat = np.array([[0.0, -1.0,  0.0],
#                                 [0.0,  0.0, -1.0],
#                                 [1.0,  0.0,  0.0]])
#         # self.RotMat=np.identity(3)
#         self.translation = np.array([0.007, -0.121, 0.026])*-1
#         # self.translation = np.array([0, 0, 0])

#         self.RotMat_K      = self.camera_info @ self.RotMat
#         # self.RotMat_K=np.identity(3)
#         self.translation_K = self.camera_info @ self.translation

#         # -------------------------------- QOS Profile ------------------------------- #
#         self.qos_profile =  QoSProfile(
#             reliability=QoSReliabilityPolicy.BEST_EFFORT,
#             history=QoSHistoryPolicy.KEEP_LAST,
#             depth=1,
#         )

#         # ------------------------------- Subscriptions ------------------------------ #
#         # ----------------------------------- LIDAR ---------------------------------- #
#         self.pointcloud_sub = self.create_subscription(
#             msg_type    = PointCloud2,
#             topic       = '/luminar_front_points',
#             callback    = self.point_cloud_callback,
#             qos_profile = self.qos_profile
#         )
#         self.pointcloud_sub  # prevent unused variable warning

#         # # ------------------------------ Bounding Box Object---------------------------------- #
#         # self.box_sub = self.create_subscription(
#         #     msg_type    = BoundingBox2D, # contain 4 coordinates and the centerpoint
#         #     topic       = 'vimba_front_left_center/out/objects',
#         #     callback    = self.box_callback,
#         #     qos_profile = self.qos_profile
#         # )
#         # self.box_sub # prevent unused variable warning

#         # # ------------------------------  Labelled Image ---------------------------------- #
#         # self.image_sub = self.create_subscription(
#         #     msg_type    = Image, 
#         #     topic       = '/vimba_front_left_center/out/image',
#         #     callback    = self.image_callback,
#         #     qos_profile = self.qos_profile
#         # )
#         # self.image_sub # prevent unused variable warning

#         # ------------------------------  Unlabelled Image ---------------------------------- #
#         self.image_sub = self.create_subscription(
#             msg_type    = Image, 
#             topic       = '/vimba_front_left_center/image',
#             callback    = self.image_callback,
#             qos_profile = self.qos_profile
#         )
#         self.image_sub # prevent unused variable warning

#         # ------------------------------- Publishers ------------------------------ #
#         self.image_pub = self.create_publisher(Image , "/vimba_front_left_center/out/image_lidar_label", rclpy.qos.qos_profile_sensor_data)
#         self.cloud_pub = self.create_publisher(PointCloud2, "/luminar_front_points/points_filtered", rclpy.qos.qos_profile_sensor_data)

#     def image_callback(self, msg):
#         print("I")
#         self.point_cloud_msg = msg
#         if self.box_msg is not None:
#             print("+")
#             self.image_ptc_callback()

#     def point_cloud_callback(self, msg):
#         print("P")
#         self.point_cloud_msg = msg
#         if self.box_msg is not None:
#             print("+")
#             self.image_ptc_callback()

#     '''Helper Function'''
#     def img_tocv2(self, message):
#         try:
#             # Convert the ROS2 Image message to a NumPy array
#             image = self.bridge.imgmsg_to_cv2(message, "bgr8")
#         except CvBridgeError as e:
#             print(e)
#         return image

#     ''' 
#     In the middle of debugging the LIDAR
#     '''
#     def image_ptc_callback(self):
#         # Lidar camera projection right here
#         ptc_numpy_rec = pointcloud2_to_array(self.point_cloud_msg)
#         ptc_numpy = get_xyz_points(ptc_numpy_rec)
#         # ptc_numpy[:,[0,1,2]]*=-1
#         # Applying Rotation 
#         # ptc_numpy=ptc_numpy[:,[1,2,0]] 
#         # ptc_numpy[:, 0:2] = -ptc_numpy[:, 0:2]
        
#         # ptc_numpy = np.array([p for p in pc2.read_points(self.point_cloud_msg, field_names = ("x", "y", "z"), skip_nans=False, uvs = [])])
#         # print(type(np.asarray(ptc_numpy)[0]))
#         # print(ptc_numpy.shape)
#         # print(self.RotMat_K)
#         ptc_numpy_c = self.camera_info @ ptc_numpy.T + np.tile(self.translation_K.reshape((3, 1)), ptc_numpy.shape[0]) # could we incorprate the camera matrix into the Rotation transformation
#         # ptc_numpy_c = np.divide(ptc_numpy_c[:, :2], ptc_numpy_c[:, 2:3])
#         # print(ptc_numpy_c.shape)
#         ptc_numpy_c = ptc_numpy_c.T
#         # ptc_numpy_c*=[50,50,50]
#         # ptc_numpy_c+=[400,400,400]
#         # Comparison with BBox
#         # Redo this
#         # print(ptc_numpy.shape,np.min(ptc_numpy,0),np.max(ptc_numpy,0))
#         # print(ptc_numpy_c.shape,np.min(ptc_numpy_c,0),np.max(ptc_numpy_c,0))
#         # ptc_numpy_c = np.divide(ptc_numpy_c[:, :2], ptc_numpy_c[:, 2:3])
#         # print(x1, x2, y1, y2)
#         # print(self.image_msg.width, self.image_msg.height)
#         # print(np.all((ptc_numpy_c[:,0]<=self.image_msg.width)&(ptc_numpy_c[:,1]<=self.image_msg.height)))
#         # print(size(np.where(ptc_numpy_c[:,0]>=self.image_msg.width)))
#         # print(image.shape)
#         # print(ptc_numpy_c[:,0])
#         # print(ptc_numpy_c.shape,np.min(ptc_numpy_c,0),np.max(ptc_numpy_c,0))
#         mask=(ptc_numpy_c[:,0]>=x1)&(ptc_numpy_c[:,0]<=x2)&(ptc_numpy_c[:,1]>=y1)&(ptc_numpy_c[:,1]<=y2)
#         # ptc_numpy_c=ptc_numpy_c[(ptc_numpy_c[:,0]>=0)&(ptc_numpy_c[:,0]<=384)&(ptc_numpy_c[:,1]>=0)&(ptc_numpy_c[:,1]<=516)]
#         # print(ptc_numpy_c.shape,np.min(ptc_numpy_c,0),np.max(ptc_numpy_c,0))
#         ptc_numpy_l = ptc_numpy_rec # Should already be in the right frame
#         # print(np.count_nonzero(mask))
#         # print(ptc_numpy_rec)
#         filtered_msg = array_to_pointcloud2(ptc_numpy_l, frame_id="luminar_front")

#         # Labelling the Image for verification Purposes
#         ptc_numpy_l_xyz = ptc_numpy[mask]
#         # print(ptc_numpy_l_xyz)
#         image = self.img_tocv2(self.image_msg) #(384, 516, 3)
#         border_size=300
#         image=cv2.copyMakeBorder(image,border_size,border_size,border_size,border_size,cv2.BORDER_CONSTANT,None,0)
#         # print(type(image))
#         # ptc_numpy_c=ptc_numpy_c[:,[0,2,1]]
#         for i in ptc_numpy_c:
#             # print(i)

#             image = cv2.circle(image, (int(i[0]+border_size), int(i[1])+border_size), 1, (0, 0, 255), 1)
#         # cv2.imshow(image)
        
#         # Publishing the Image and PointCloud
#         self.image_pub.publish(self.bridge.cv2_to_imgmsg(image[:2*border_size,:2*border_size]))
#         self.cloud_pub.publish(filtered_msg)

#         self.box_msg = None
#         self.point_cloud_msg = None 

    # def image_callback(self, msg):
    #     print("I")
    #     self.image_msg = msg

    # def box_callback(self, msg):
    #     print("B")
    #     self.box_msg = msg
    #     if self.point_cloud_msg is not None:
    #         print("+")
    #         self.pair_box_point_cloud()

    # def point_cloud_callback(self, msg):
    #     print("P")
    #     self.point_cloud_msg = msg
    #     if self.box_msg is not None:
    #         print("+")
    #         self.pair_box_point_cloud()

    # '''Helper functions'''
    # def box_to_corners(self, cx, cy, width, height):
    #     half_width = width / 2
    #     half_height = height / 2

    #     y1 = cy - half_height # top
    #     y2 = cy + half_height # bottom
    #     x1 = cx - half_width # left
    #     x2 = cx + half_width # right

    #     return (y1, y2, x1, x2) #(up, down, left, right)

    # def img_tocv2(self, message):
    #     try:
    #         # Convert the ROS2 Image message to a NumPy array
    #         image = self.bridge.imgmsg_to_cv2(message, "bgr8")
    #     except CvBridgeError as e:
    #         print(e)
    #     return image




    '''
    Transformation from front_left_camera to luminar_front
    Translation: [-0.107, -0.146, 0.026]
    Rotation: in Quaternion [0.000, 0.000, -0.462, 0.887]
    Rotation Matrix = [  0.5732029,  0.8194135,  0.0000000;
                        -0.8194135,  0.5732029,  0.0000000;
                        0.0000000,  0.0000000,  1.0000000 ]
    Transformation from  luminar_front to camera
    - Translation: [-0.058, 0.171, -0.026]
    - Rotation: in Quaternion [0.000, 0.000, 0.462, 0.887]
    - Rotation Matrix = [  0.5732029, -0.8194135,  0.0000000;
                        0.8194135,  0.5732029,  0.0000000;
                        0.0000000,  0.0000000,  1.0000000 ]
    '''

    # def pair_box_point_cloud(self):
    #     # Box to homogeneous coords
    #     y1, y2, x1, x2 =  self.box_to_corners(self.box_msg.center.x, self.box_msg.center.y, self.box_msg.size_x, self.box_msg.size_y)

    #     # Lidar camera projection right here
    #     ptc_numpy_rec = pointcloud2_to_array(self.point_cloud_msg)
    #     ptc_numpy = get_xyz_points(ptc_numpy_rec)
    #     # ptc_numpy[:,[0,1,2]]*=-1
    #     # Applying Rotation 
    #     # ptc_numpy=ptc_numpy[:,[1,2,0]] 
    #     # ptc_numpy[:, 0:2] = -ptc_numpy[:, 0:2]
        
    #     # ptc_numpy = np.array([p for p in pc2.read_points(self.point_cloud_msg, field_names = ("x", "y", "z"), skip_nans=False, uvs = [])])
    #     # print(type(np.asarray(ptc_numpy)[0]))
    #     # print(ptc_numpy.shape)
    #     # print(self.RotMat_K)
    #     ptc_numpy_c = self.camera_info @ ptc_numpy.T + np.tile(self.translation_K.reshape((3, 1)), ptc_numpy.shape[0]) # could we incorprate the camera matrix into the Rotation transformation
    #     # ptc_numpy_c = np.divide(ptc_numpy_c[:, :2], ptc_numpy_c[:, 2:3])
    #     # print(ptc_numpy_c.shape)
    #     ptc_numpy_c = ptc_numpy_c.T
    #     # ptc_numpy_c*=[50,50,50]
    #     # ptc_numpy_c+=[400,400,400]
    #     # Comparison with BBox
    #     # Redo this
    #     # print(ptc_numpy.shape,np.min(ptc_numpy,0),np.max(ptc_numpy,0))
    #     # print(ptc_numpy_c.shape,np.min(ptc_numpy_c,0),np.max(ptc_numpy_c,0))
    #     # ptc_numpy_c = np.divide(ptc_numpy_c[:, :2], ptc_numpy_c[:, 2:3])
    #     # print(x1, x2, y1, y2)
    #     # print(self.image_msg.width, self.image_msg.height)
    #     # print(np.all((ptc_numpy_c[:,0]<=self.image_msg.width)&(ptc_numpy_c[:,1]<=self.image_msg.height)))
    #     # print(size(np.where(ptc_numpy_c[:,0]>=self.image_msg.width)))
    #     # print(image.shape)
    #     # print(ptc_numpy_c[:,0])
    #     # print(ptc_numpy_c.shape,np.min(ptc_numpy_c,0),np.max(ptc_numpy_c,0))
    #     mask=(ptc_numpy_c[:,0]>=x1)&(ptc_numpy_c[:,0]<=x2)&(ptc_numpy_c[:,1]>=y1)&(ptc_numpy_c[:,1]<=y2)
    #     # ptc_numpy_c=ptc_numpy_c[(ptc_numpy_c[:,0]>=0)&(ptc_numpy_c[:,0]<=384)&(ptc_numpy_c[:,1]>=0)&(ptc_numpy_c[:,1]<=516)]
    #     # print(ptc_numpy_c.shape,np.min(ptc_numpy_c,0),np.max(ptc_numpy_c,0))
    #     ptc_numpy_l = ptc_numpy_rec # Should already be in the right frame
    #     # print(np.count_nonzero(mask))
    #     # print(ptc_numpy_rec)
    #     filtered_msg = array_to_pointcloud2(ptc_numpy_l, frame_id="luminar_front")

    #     # Labelling the Image for verification Purposes
    #     ptc_numpy_l_xyz = ptc_numpy[mask]
    #     # print(ptc_numpy_l_xyz)
    #     image = self.img_tocv2(self.image_msg) #(384, 516, 3)
    #     border_size=300
    #     image=cv2.copyMakeBorder(image,border_size,border_size,border_size,border_size,cv2.BORDER_CONSTANT,None,0)
    #     # print(type(image))
    #     # ptc_numpy_c=ptc_numpy_c[:,[0,2,1]]
    #     for i in ptc_numpy_c:
    #         # print(i)

    #         image = cv2.circle(image, (int(i[0]+border_size), int(i[1])+border_size), 1, (0, 0, 255), 1)
    #     # cv2.imshow(image)
        
    #     # Publishing the Image and PointCloud
    #     self.image_pub.publish(self.bridge.cv2_to_imgmsg(image[:2*border_size,:2*border_size]))
    #     self.cloud_pub.publish(filtered_msg)

    #     self.box_msg = None
    #     self.point_cloud_msg = None
            


        

# class Lidar2Cam(Node):
#     def __init__(self):
#         super().__init__('lidar_to_cam_node')
#         print('Node Initialized!')

#         # -------------------------------- QOS Profile ------------------------------- #
#         self.qos_profile =  QoSProfile(
#             reliability=QoSReliabilityPolicy.BEST_EFFORT,
#             history=QoSHistoryPolicy.KEEP_LAST,
#             depth=1,
#         )
#         print('QOS Init Done')

#         # ------------------------------- Subscriptions ------------------------------ #
#         # ----------------------------------- LIDAR ---------------------------------- #
#         self.pointcloud_sub = self.create_subscription(
#             msg_type    = PointCloud2,
#             topic       = '/luminar_front_points',
#             # topic       = '/luminar_front_points/points_raw/ground_filtered', 
#             callback    = self.callback,
#             qos_profile = self.qos_profile
#         )
#         # self.cam_info_sub = self.create_subscription(
#         #     msg_type    = CameraInfo,
#         #     topic       = '/vimba_front_left/camera_info',
#         #     callback    = self.cam_info_callback,
#         #     qos_profile = rclpy.qos.QoSProfile(depth=1) 
#         # )

#         self.pointcloud_sub  # prevent unused variable warning

#         self.marker_pub = self.create_publisher(PointCloud2, "/lidar2cam_ptc", rclpy.qos.qos_profile_sensor_data)
#         # self.marker_pub = self.create_publisher(PointCloud2, "/lidar2cam_ptc", rclpy.qos.qos_profile_sensor_data)
        
#         # self.ptc_sub    = Subscriber(self, PointCloud2, "/luminar_front_points/points_raw/ground_filtered", self.qos_profile)
#         # self.ptc_raw_sub    = Subscriber(PointCloud2, "/luminar_front_points/points_raw", self.callback, self.qos_profile)
        

#         self.rate = self.create_rate(0.01) #2Hz

#     # def cam_info_callback(self, msg):
#     #     print('received ')


#     def callback(self, ptc):
#         self.marker_pub.publish(ptc)
#         # while(True):
#         ptc_numpy = np.array([p for p in pc2.read_points(ptc, field_names = ("x", "y", "z"), skip_nans=False, uvs = [])]) # uvs: only give the points in the list
#         camera_info = np.array([[251.935177, 0.000000, 260.887279], 
#                                 [0.000000, 252.003440, 196.606218], 
#                                 [0.000000, 0.000000, 1.000000]])
        
        
        
#         # ptc_gen = pc2.read_points(ptc, field_names = ("x", "y", "z"), skip_nans=True)
#         # ptc_numpy = np.array(next(ptc_gen).append(0)) # append the order stamp
#         # ptc = []
#         # for i, p in enumerate(ptc_gen):
#         #     ptc = ptc.append(list(p)
#         # ptc_numpy = np.array(ptc)
#         # ptc_numpy = ptc_numpy[::4]
#         print(ptc_numpy.shape)
#         # print(ptc_numpy.tolist())
#         # print(ptc_numpy)
#         # print("Published!")
#         # print(ptc_numpy.shape)
#         # print(ptc_numpy[:, :2].shape)
#         # print(ptc_numpy[:, :2])
#         # print(ptc_numpy[:1000, :1].size)
#         # print(ptc_numpy[:1000, 1:2].size)
#         # count = 0
#         # prev = 0
#         # for i in range(ptc_numpy.shape[0]):
#         #     if ptc_numpy[i,0] < prev:
#         #         print(i, count)
#         #         count += 1
#         #     prev = ptc_numpy[i,0]
#         fig = plt.figure()
#         # ax = fig.add_subplot(projection='3d')
#         # ax = fig.gca(projection='3d')
#         # ax.plot(ptc_numpy[:, 1], ptc_numpy[:, 2], ptc_numpy[:, 0], linewidth=0.5, marker = ".")
#         # ax.scatter(ptc_numpy[:, 0], ptc_numpy[:, 1], ptc_numpy[:, 2], linewidth=1, marker = ".")
#         time = np.arange(len(ptc_numpy))[::-1].reshape(-1, 1) # reversed array
#         reduction_factor = 2
#         ptc_numpy = np.hstack((ptc_numpy,time))[::-1]
#         ptc_numpys = ptc_numpy[::reduction_factor]
#         # print(ptc_numpy.tolist())
#         close_up = 5000 # get the closest 5000 points by order starting from the eyes of the lidar
#         close_ups = int(close_up/reduction_factor)
#         print(close_ups)
#         # ax.scatter(-ptc_numpy[:close_up, 1], ptc_numpy[:close_up, 2], ptc_numpy[:close_up, 3], linewidth=1, marker = ".")
#         # ax.plot(-ptc_numpy[:close_up, 1], ptc_numpy[:close_up, 2], ptc_numpy[:close_up, 3], linewidth=1, marker = ".")
#         plt.scatter(-ptc_numpy[:close_up, 1], ptc_numpy[:close_up, 3], linewidth=1, marker = ".")
#         plt.plot(-ptc_numpys[:close_ups, 1], ptc_numpys[:close_ups, 3], linewidth=1, marker = ".", color = 'red')
#         # ax.set_xlabel('Y Label')
#         # ax.set_ylabel('Z Label')
#         # ax.set_zlabel('Time')

#         # plt.figure()
#         # for i in range(num_points):
#         #     # print(ptc_numpy[i])
#         #     plt.text(-ptc_numpy[i, 1:2],ptc_numpy[i, 2:],str(i))
#         #     # plt.plot(-ptc_numpy[i, 1:2],ptc_numpy[i, 2:])
#             # plt.scatter(-ptc_numpy[i, 0],ptc_numpy[i, 1])
#         # for i in range(num_points):
#         #     plt.annotate(i, (-ptc_numpy[i, 1], ptc_numpy[i, 2]))
#         # for i in range(num_points):
#         #     ax.scatter(ptc_numpy[i, 0], ptc_numpy[i, 1], ptc_numpy[i, 2], marker='o')
#         # ax.plot(ptc_numpy[:num_points, 0], ptc_numpy[:num_points, 1], ptc_numpy[:num_points, 2], marker='o')

        
#             # plt.annotate(i, (-ptc_numpy[i, 1], ptc_numpy[i, 2]))
#         # plt.plot(-ptc_numpy[:num_points, 0],ptc_numpy[:num_points, 1])
#         plt.show()
#         self.rate.sleep()



#         # self.marker_pub.publish(ptc_numpy)


#     # def ptc2numpy(self, cloud_msg):
#     #     # Create a point cloud message
#     #     cloud_msg = PointCloud2()

#     #     # Use the point_step and row_step fields to access the point data as a bytes object
#     #     point_data = cloud_msg.data[cloud_msg.header.data_offset:cloud_msg.height * cloud_msg.width * cloud_msg.point_step]

#     #     # Convert the data to a NumPy array
#     #     points = np.frombuffer(point_data, dtype=np.float32).reshape(-1, cloud_msg.fields[0].count)

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
    print("going to spin")
    rclpy.spin(minimal_subscriber)
    print("destroying node")
    # Destroy the node explicitly
    minimal_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
