import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

from vision_msgs.msg import Detection2DArray, Detection2D  # BoundingBox3D, BoundingBox2D, BoundingBox2DArray
from builtin_interfaces.msg import Duration
from visualization_msgs.msg import Marker
from sensor_msgs.msg import PointCloud2, PointField, Image
from message_filters import ApproximateTimeSynchronizer, TimeSynchronizer, Subscriber
# from sensor_msgs.msg import CameraInfo
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

        # ---------------------------------------------------------------------------- #
        #                                  Parameters                                  #
        # ---------------------------------------------------------------------------- #
        self.time_threshold = 0.01  # 10ms
        # ------------------------------ End Parameters ------------------------------ #

        # Bbox Array (Only the ones with Detections)
        self.bboxes_array_msg = None

        self.cam_map = {
            "vimba_front_left":         1,
            "vimba_front_left_center":  2,
            "vimba_front_right_center": 3,
            "vimba_front_right":        4,
            "vimba_rear_right":         5,
            "vimba_rear_left":          6,
        }

        self.lidar_map = {
            "front": 1,
            "left":  2,
            "right": 3
        }

        self.cam2lidar = {
            1: [1, 2],
            2: [1],
            3: [1],
            4: [1, 3],
            5: [3],
            6: [2],
        }

        self.lidar_msg = {
            1: None,
            2: None,
            3: None,
        }
        self.cam_matrix = {}
        self.translation = {}
        self.RotMat = {}
        self.LidarTranslation = {}
        self.LidarRotMat = {}
        # ---------------------------------------------------------------------------- #
        #         No more Camera Matrix Because the YOLO Node will Deal With it        #
        # ---------------------------------------------------------------------------- #
        self.cam_matrix[1] = np.array(
            [[196.296974, -0.307402, 256.633528], [0.000000, 196.203937, 191.920790], [0.000000, 0.000000, 1.000000]])
        self.cam_matrix[2] = np.array(
            [[1731.375645, 0.000000, 564.055153], [0.000000, 1728.054397, 290.619860], [0.000000, 0.000000, 1.000000]])
        self.cam_matrix[3] = np.array(
            [[1760.474682, 0.000000, 619.440720], [0.000000, 1759.762046, 394.231557], [0.000000, 0.000000, 1.000000]])
        self.cam_matrix[4] = np.array(
            [[162.964445, 2.217544, 255.811038], [0.000000, 163.123969, 192.409497], [0.000000, 0.000000, 1.000000]])
        self.cam_matrix[5] = np.array(
            [[189.189981, -14.738140, 257.995696], [0.000000, 191.503315, 174.894545], [0.000000, 0.000000, 1.000000]])
        self.cam_matrix[6] = np.array(
            [[172.786842, 1.411124, 255.612286], [0.000000, 170.205329, 195.844222], [0.000000, 0.000000, 1.000000]])

        # ---------------------------------------------------------------------------- #
        #                        Original Transformation Matrices                      #
        # ---------------------------------------------------------------------------- #
        self.translation[(2, 1)] = np.array([0.121, -0.026, 0.007])  # Front Lidar to Front Left Center Camera
        self.translation[(3, 1)] = np.array([-0.121, -0.026, 0.007])  # Front Lidar to Front Right Center Camera
        self.translation[(1, 1)] = np.array([0.146, -0.026, -0.107])  # Front Lidar to Front Left Camera
        self.translation[(4, 1)] = np.array([-0.146, -0.026, -0.107])  # Front Lidar to Front Right Camera
        self.translation[(1, 2)] = np.array([-0.575, -0.121, -0.286])  # Left Lidar to Front Left Camera
        self.translation[(6, 2)] = np.array([0.140, -0.000, 0.048])  # Left Lidar to Rear Left Camera
        self.translation[(4, 3)] = np.array([0.575, -0.121, -0.286])  # Right Lidar to Front Right Camera
        self.translation[(5, 3)] = np.array([-0.140, -0.000, 0.048])  # Right Lidar to Rear Right Camera
        self.RotMat[(2, 1)] = Rotation.from_quat([0.496, -0.496, 0.504, 0.504]).as_matrix()
        self.RotMat[(3, 1)] = Rotation.from_quat([0.496, -0.496, 0.504, 0.504]).as_matrix()
        self.RotMat[(1, 1)] = Rotation.from_quat([0.672, -0.207, 0.219, 0.676]).as_matrix()
        self.RotMat[(4, 1)] = Rotation.from_quat([0.207, -0.673, 0.676, 0.219]).as_matrix()
        self.RotMat[(1, 2)] = Rotation.from_quat([-0.153, 0.690, -0.690, -0.153]).as_matrix()
        self.RotMat[(6, 2)] = Rotation.from_quat([0.542, -0.455, 0.455, 0.542]).as_matrix()
        self.RotMat[(4, 3)] = Rotation.from_quat([0.689, -0.160, 0.146, 0.692]).as_matrix()
        self.RotMat[(5, 3)] = Rotation.from_quat([-0.457, 0.548, -0.535, -0.452]).as_matrix()

        # ---------------------------------------------------------------------------- #
        #                      Calibrated Transformation Matrices                      #
        # ---------------------------------------------------------------------------- #
        self.translation[(2, 1)] += np.array([0.0000, 0.0000, -0.0480])  # Front Lidar to Front Left Center Camera
        self.translation[(3, 1)] += np.array([-0.0200, 0.1430, 0.0070])  # Front Lidar to Front Right Center Camera
        self.translation[(1, 1)] += np.array([0.0000, 0.0000, -0.1500])  # Front Lidar to Front Left Camera
        self.translation[(4, 1)] += np.array([0.0000, 0.0000, 0.0000])  # Front Lidar to Front Right Camera
        # self.translation[(1, 2)] = np.array([-0.575, -0.121, -0.286])  # Left Lidar to Front Left Camera
        self.translation[(6, 2)] += np.array([0.0000, 0.0000, 0.1430])  # Left Lidar to Rear Left Camera
        # self.translation[(4, 3)] = np.array([0.575, -0.121, -0.286])  # Right Lidar to Front Right Camera
        self.translation[(5, 3)] += np.array([0.0270, -0.0340, 0.0000])  # Right Lidar to Rear Right Camera
        self.RotMat[(2, 1)] = np.array(
            [[0.9998, -0.0018, 0.0176], [0.0018, 0.9992, 0.0400], [-0.0176, -0.0400, 0.9990]])@self.RotMat[(2, 1)]
        self.RotMat[(3, 1)] = np.array(
            [[0.9951, -0.0024, -0.0986], [0.0000, 0.9997, -0.0244], [0.0986, 0.0243, 0.9948]])@self.RotMat[(3, 1)]
        self.RotMat[(1, 1)] = np.array(
            [[0.9998, 0.0059, 0.0203], [-0.0059, 1.0000, 0.0015], [-0.0202, -0.0016, 0.9998]])@self.RotMat[(1, 1)]
        self.RotMat[(4, 1)] = np.array(
            [[0.9988, -0.0422, 0.0239], [0.0427, 0.9988, -0.0238], [-0.0229, 0.0248, 0.9994]])@self.RotMat[(4, 1)]
        # self.RotMat[(1, 2)] = Rotation.from_quat([-0.153, 0.690, -0.690, -0.153]).as_matrix()
        self.RotMat[(6, 2)] = np.array(
            [[0.9998, -0.0005, 0.0183], [0.0000, 0.9997, 0.0260], [-0.0183, -0.0260, 0.9995]])@self.RotMat[(6, 2)]
        # self.RotMat[(4, 3)] = Rotation.from_quat([0.689, -0.160, 0.146, 0.692]).as_matrix()
        self.RotMat[(5, 3)] = np.array([[1.0000, 0.0000, 0.0000], [0.0000, 1.0000, 0.0000], [0.0000, 0.0000, 1.0000]])@ \
                              self.RotMat[(5, 3)]

        # ---------------------------------------------------------------------------- #
        #                    Calculate Lidar Transformation Matrices                   #
        # ---------------------------------------------------------------------------- #
        # 1,2 and 1,1
        r2inv = np.linalg.inv(self.RotMat[(1, 1)])
        self.LidarTranslation[(2, 1)] = r2inv@self.translation[(1, 2)]-self.translation[(1, 1)]
        self.LidarRotMat[(2, 1)] = r2inv@self.RotMat[(1, 2)]
        # 4,3 and 4,1
        r2inv = np.linalg.inv(self.RotMat[(4, 1)])
        self.LidarTranslation[(3, 1)] = r2inv@self.translation[(4, 3)]-self.translation[(4, 1)]
        self.LidarRotMat[(3, 1)] = r2inv@self.RotMat[(4, 3)]

        # ---------------------------------------------------------------------------- #
        #                                  QOS Profile                                 #
        # ---------------------------------------------------------------------------- #
        self.qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # ---------------------------------------------------------------------------- #
        #                                 Subscriptions                                #
        # ---------------------------------------------------------------------------- #

        # Front LIDAR PointCloud ----------------------------- #
        self.front_pointcloud_sub = self.create_subscription(
            msg_type=PointCloud2,
            topic='/luminar_front_points',
            callback=self.front_cloud_callback,
            qos_profile=self.qos_profile
        )
        self.front_pointcloud_sub  # prevent unused variable warning

        # Left LIDAR PointCloud ----------------------------- #
        self.left_pointcloud_sub = self.create_subscription(
            msg_type=PointCloud2,
            topic='/luminar_left_points',
            callback=self.left_cloud_callback,
            qos_profile=self.qos_profile
        )
        self.left_pointcloud_sub  # prevent unused variable warning

        # Right LIDAR PointCloud ----------------------------- #
        self.right_pointcloud_sub = self.create_subscription(
            msg_type=PointCloud2,
            topic='/luminar_right_points',
            callback=self.right_cloud_callback,
            qos_profile=self.qos_profile
        )
        self.right_pointcloud_sub  # prevent unused variable warning

        # YOLO Detection2DArray Subscription ---------------------------------- #
        self.image_sub = self.create_subscription(
            msg_type=Detection2DArray,
            topic='vimba_front_left_center/out/objects',
            qos_profile=self.qos_profile
        )
        self.image_sub  # prevent unused variable warning

        # ---------------------------------------------------------------------------- #
        #                                  Publishers                                  #
        # ---------------------------------------------------------------------------- #
        self.marker_pub = self.create_publisher(Marker, '/Lidar_car_marker', rclpy.qos.qos_profile_sensor_data)

    # ---------------------------------------------------------------------------- #
    #                     Callback Functions for Subscriptions                     #
    # ---------------------------------------------------------------------------- #
    '''
    No matter what is received first, the execute_projection function will be executed.
    We will have a replacement policy, updating the buffers with the latest messages only.
    Wait till all the buffers are filled before doing the execution.
    '''

    # ------------------------------ YOLO Detection ------------------------------ #
    def YOLO_callback(self, msg):
        # print("B")
        self.bboxes_array_msg = msg
        self.execute_projection()

    # -------------------------------- PointClouds ------------------------------- #
    def front_cloud_callback(self, msg):
        # print("P")
        self.lidar_msg[1] = msg

    def left_cloud_callback(self, msg):
        # print("P")
        self.lidar_msg[2] = msg

    def right_cloud_callback(self, msg):
        # print("P")
        self.lidar_msg[3] = msg

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
        # # 1
        # # Check the buffer is filled (Potential issue: if any of the messages are not received, the projection will not happen)
        # if self.bboxes_array_msg is None:
        #     return
        # 1.1
        # Timestamp check
        do_left, do_right, do_front = False, False, False
        if self.front_cloud_msg is not None and abs(
                self.lidar_msg[1].header.stamp-self.bboxes_array_msg.header.stamp) <= self.time_threshold:
            do_front = True
        if self.left_cloud_msg is not None and abs(
                self.lidar_msg[2].header.stamp-self.bboxes_array_msg.header.stamp) <= self.time_threshold:
            do_left = True
        if self.right_cloud_msg is not None and abs(
                self.lidar_msg[3].header.stamp-self.bboxes_array_msg.header.stamp) <= self.time_threshold:
            do_right = True

        LidarDict = {}
        LidarDict[1] = [[], [], []]
        LidarDict[2] = [[], [], []]
        LidarDict[3] = [[], [], []]
        count = 0
        for bbox_msg in self.bboxes_array_msg.detections:
            camId = bbox_msg.frame_id
            lidarIdList = self.cam2lidar[camId]
            for lidarId in lidarIdList:
                if (lidarId == 1 and do_front) or (lidarId == 2 and do_left) or (lidarId == 3 and do_right):
                    LidarDict[lidarId][0].append(
                        self.BboxInLidarDirection(bbox_msg, self.cam_matrix[camId], self.RotMat[(camId, lidarId)]))
                    LidarDict[lidarId][1].append(self.RotMat[(camId, lidarId)])
                    LidarDict[lidarId][2].append(self.translation[(camId, lidarId)])
                    count += 1

        Lidar1Filtered = self.PointSelection(self.lidar_msg[1], LidarDict[1][0], LidarDict[1][1], LidarDict[1][2])

        Lidar2Filtered = self.PointSelection(self.lidar_msg[2], LidarDict[2][0], LidarDict[2][1], LidarDict[2][2])
        Lidar2FilteredIn1 = (self.LidarRotMat[(2, 1)]@Lidar2Filtered.T+self.LidarTranslation[(2, 1)].reshape(3, 1)).T

        Lidar3Filtered = self.PointSelection(self.lidar_msg[3], LidarDict[3][0], LidarDict[3][1], LidarDict[3][2])
        Lidar3FilteredIn1 = (self.LidarRotMat[(3, 1)]@Lidar3Filtered.T+self.LidarTranslation[(3, 1)].reshape(3, 1)).T

        LidarAllFiltered = np.concatenate((Lidar1Filtered, Lidar2FilteredIn1, Lidar3FilteredIn1), axis=0)
        stamp = self.bboxes_array_msg.header.stamp
        self.marker_pub.publish(self.Cluster2Marker(LidarAllFiltered, stamp, count))

    def BboxInLidarDirection(self, bbox_msg, camera_info, RotMat):
        K_inv = inv(camera_info)
        camera_corners_cam = K_inv@boxes_to_matirx([bbox_msg], 0)
        R_inv = inv(RotMat)
        # Apply the inverse rotation matrix
        camera_corners_lid = R_inv@camera_corners_cam  # This operation will make the bottom row not necessarily zero
        # The First is the Z-axis of the lidar frame
        camera_corners_lid_z = camera_corners_lid[:, 0:1]
        camera_corners_lid_normed = camera_corners_lid[:, 1:]/camera_corners_lid_z
        camera_corners_lid_normed = camera_corners_lid_normed.T
        return camera_corners_lid_normed

    def PointSelection(self, point_cloud_msg, camera_corners_lid_normed_list, RotMat_list, translation_list):
        # 3
        # Normalize all points on their Z-axis

        ptc_numpy_record = pointcloud2_to_array(point_cloud_msg)
        ptc_xyz_lidar = get_xyz_points(ptc_numpy_record)  # (N * 3 matrix)
        mask = np.full((ptc_xyz_lidar.shape[0],), False)

        for j in range(len(camera_corners_lid_normed_list)):
            camera_corners_lid_normed = camera_corners_lid_normed_list[j]
            RotMat = RotMat_list[j]
            translation = translation_list[j]
            R_inv = inv(RotMat)
            trans = R_inv@translation
            ptc_xyz_lidar_trans = ptc_xyz_lidar-trans[np.newaxis, :]
            ptc_z_camera = ptc_xyz_lidar_trans[:, 0].reshape((-1, 1))
            ptc_xyz_lidar_normed = ptc_xyz_lidar_trans[:, 1:]/ptc_z_camera

            # 4
            # Capture all points within the bounding box

            for i in range(camera_corners_lid_normed.shape[2]):  # camera_corners_lid.shape[1]): #(Columns as size)
                mask = (mask | ((ptc_xyz_lidar_normed[:, 0] >= camera_corners_lid_normed[0, 0, i]) &  # x>=left
                                (ptc_xyz_lidar_normed[:, 0] <= camera_corners_lid_normed[1, 0, i]) &  # x<=right
                                (ptc_xyz_lidar_normed[:, 1] >= camera_corners_lid_normed[0, 1, i]) &  # y>=top
                                (ptc_xyz_lidar_normed[:, 1] <= camera_corners_lid_normed[1, 1, i])))  # y<=bottom
                # Space for Optimization here

        ptc_xyz_lidar_filtered = ptc_xyz_lidar[mask]
        num_lidar = np.sum(mask)
        print('num lidar in bbox:', num_lidar)
        return ptc_xyz_lidar_filtered

    # ---------------------------------------------------------------------------- #
    #                                   Custering                                  #
    # ---------------------------------------------------------------------------- #
    def Cluster2Marker(self, ptc_xyz_camera_list, stamp, BboxCount):
        o3d_pcd: o3d.geometry.PointCloud = o3d.geometry.PointCloud(
            o3d.utility.Vector3dVector(ptc_xyz_camera_list)
        )
        labels = np.array(
            o3d_pcd.cluster_dbscan(
                eps=1.8, min_points=max(int(ptc_xyz_camera_list.shape[0]/4/BboxCount), 10), print_progress=False
            )
        )
        max_label = labels.max()
        print('label num:', max_label+1)

        # ---------------------------------------------------------------------------- #
        #                                Calculate 3D BBox                             #
        # ---------------------------------------------------------------------------- #
        median_xyz_camera = []
        bbox_rotate_mode = True
        for i in range(max_label+1):
            print(f'label {i} num:', np.where(labels == i)[0].shape)
            cluster = o3d_pcd.select_by_index(list(np.where(labels == i)[0]))
            if bbox_rotate_mode:
                cluster_bbox = cluster.get_oriented_bounding_box()
            else:
                cluster_bbox = cluster.get_axis_aligned_bounding_box()
            median_xyz_camera.append([cluster_bbox, abs(cluster_bbox.get_center()[0]), i])
        median_xyz_camera.sort(key=lambda x: x[1])
        if max_label > -1:
            print('label choosed:', median_xyz_camera[0][2], 'distance:', median_xyz_camera[0][1])

        # # ---------------------------------------------------------------------------- #
        # #                   Creating a Marker Object to be published                   #
        # # ---------------------------------------------------------------------------- #
        markerList = []
        count = 0
        for cluster_bbox, _, _ in median_xyz_camera:
            count += 1
            if count > BboxCount:
                break
            if bbox_rotate_mode:
                center = cluster_bbox.get_center()
                extent = cluster_bbox.extent
                rotation = cluster_bbox.R.copy()
                rotation[2] *= -1
                rotation = Rotation.from_matrix(rotation).as_quat()
                center = [center[0], -center[1], center[2]]
            else:
                center = cluster_bbox.get_center()
                extent = cluster_bbox.get_extent()
                center = [center[0], -center[1], center[2]]

            marker_msg = Marker()
            marker_msg.header.frame_id = "luminar_right"
            marker_msg.header.stamp = stamp
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
            markerList.append(marker_msg)
        return markerList

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
