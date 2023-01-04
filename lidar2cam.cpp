#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Vector3.h>
#include <tf2/transform_datatypes.h>
#include <tf2_eigen/tf2_eigen.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_ros/point_cloud.h>
#include <pcl_ros/transforms.h>
#include <image_transport/image_transport.h>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>

class Lidar2CamNode : public rclcpp::Node
{
public:
    Lidar2CamNode()
        : Node("lidar_to_cam_node")
    {
        RCLCPP_INFO(this->get_logger(), "Initialized Lidar2Cam Node!");

        // Initialize member variables
        time_threshold_ = std::chrono::duration<double, std::milli>(10);
        bboxes_array_msg_ = nullptr;

        cam_map_ = {
            {"vimba_front_left", 1},
            {"vimba_front_left_center", 2},
            {"vimba_front_right_center", 3},
            {"vimba_front_right", 4},
            {"vimba_rear_left", 5},
            {"vimba_rear_right", 6}
        }

        lidar_msg_ = {
            {1, nullptr},
            {2, nullptr},
            {3, nullptr},
        };

    cam_matrix_ = {
      {1, Eigen::Matrix3d(196.296974, -0.307402, 256.633528, 0.000000, 196.203937, 191.920790, 0.000000, 0.000000, 1.000000)},
      {2, Eigen::Matrix3d(1731.375645, 0.000000, 564.055153, 0.000000, 1728.054397, 290.619860, 0.000000, 0.000000, 1.000000)},
      {3, Eigen::Matrix3d(1760.474682, 0.000000, 619.440720, 0.000000, 1759.762046, 394.231557, 0.000000, 0.000000, 1.000000)},
      {4, Eigen::Matrix3d(162.964445, 2.217544, 255.811038, 0.000000, 163.123969, 192.409497, 0.000000, 0.000000, 1.000000)},
      {5, Eigen::Matrix3d(189.189981, -14.738140,

    translation_[{2, 1}] = Eigen::Vector3d(0.121, -0.026, 0.007);  // Front Lidar to Front Left Center Camera
    translation_[{3, 1}] = Eigen::Vector3d(-0.121, -0.026, 0.007);  // Front Lidar to Front Right Center Camera
    translation_[{1, 1}] = Eigen::Vector3d(0.146, -0.026, -0.107);  // Front Lidar to Front Left Camera
    translation_[{4, 1}] = Eigen::Vector3d(-0.146, -0.026, -0.107);  // Front Lidar to Front Right Camera
    translation_[{1, 2}] = Eigen::Vector3d(-0.575, -0.121, -0.286);  // Front Left Lidar to Front Left Camera
    translation_[{1, 3}] = Eigen::Vector3d(0.575, -0.121, -0.286);   // Front Right Lidar to Front Right Camera
    translation_[{2, 6}] = Eigen::Vector3d(0.316, -0.266, -0.195);   // Left Lidar to Rear Left Camera
    translation_[{3, 5}] = Eigen::Vector3d(-0.316, -0.266, -0.195);  // Right Lidar to Rear Right Camera
    translation_[{1, 4}] = Eigen::Vector3d(-0.057, -0.138, -0.261);  // Front Lidar to Rear Left Camera
    translation_[{1, 5}] = Eigen::Vector3d(0.057, -0.138, -0.261);   // Front Lidar to Rear Right Camera

    RotMat_[{2, 1}] = Eigen::Matrix3d(0.959660, -0.280807, -0.000319, 0.281005, 0.956849, 0.076065, 0.001173, -0.075679, 0.997047);
    RotMat_[{3, 1}] = Eigen::Matrix3d(-0.959660, -0.280807, -0.000319, -0.281005, 0.956849, 0.076065, -0.001173, -0.075679, 0.997047);
    RotMat_[{1, 1}] = Eigen::Matrix3d(-0.993105, 0.116875, 0.024628, -0.116529, -0.993105, 0.038141, -0.025742, -0.037581, -0.998857);
    RotMat_[{4, 1}] = Eigen::Matrix3d(0.993105, 0.116875, 0.024628, 0.116529, -0.993105, 0.038141, 0.025742, -0.037581, -0.998857);
    RotMat_[{1, 2}] = Eigen::Matrix3d(-0.976682, -0.214176, -0.012961, -0.214449, 0.976682, -0.029065, 0.010794, 0.029819, 0.999409);
    RotMat_[{1, 3}] = Eigen::Matrix3d(-0.976682, -0.214176, -0.012961, -0.214449, 0.976682, -0.029065, 0.010794, 0.029819, 0.999409);
    RotMat_[{2, 6}] = Eigen::Matrix3d(-0.985614, -0.168891, -0.018960, -0.169150, 0.985614, 0.012523, 0.017766, -0.014489, 0.999779);
    RotMat_[{3, 5}] = Eigen::Matrix3d(-0.985614, -0.168891, -0.018960, -0.169150, 0.985614, 0.012523, 0.017766, -0.014489, 0.999779);
    RotMat_[{1, 4}] = Eigen::Matrix3d(-0.984840, -0.172939, 0.006484, -0.173578, 0.984840, -0.026565, 0.002549, 0.025735, 0.999656);
    RotMat_[{1, 5}] = Eigen::Matrix3d(-0.984840, -0.172939, 0.006484, -0.173578, 0.984840, -0.026565, 0.002549, 0.025735, 0.999656);

    // Subscribe to the Lidar and Bbox Array Topics
    for (auto const &lidar : lidar_map_)
    {
            auto callback = std::bind(&Lidar2CamNode::lidar_callback, this, std::placeholders::_1, lidar.first);
            lidar_sub_[lidar.first] = this->create_subscription<sensor_msgs::msg::PointCloud2>(
                "lidar_" + std::to_string(lidar.first) + "_points", 10, callback);
    }
    bbox_array_sub_ = this->create_subscription<darknet_ros_msgs::msg::BoundingBoxes>(
        "darknet_ros/bounding_boxes", 10, std::bind(&Lidar2CamNode::bbox_array_callback, this, std::placeholders::_1));

    private:
        void lidar_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg, int id)
        {
            RCLCPP_DEBUG(this->get_logger(), "Lidar %d callback triggered!", id);

            if (lidar_msg_[id] != nullptr && (get_clock()->now() - lidar_msg_[id]->header.stamp).seconds() < time_threshold_.count() / 1000.0)
