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


/* -------------------------------------------------------------------------- */
/*                                 ROS2 NODE                                  */
/* -------------------------------------------------------------------------- */
using namespace std::chrono_literals;

YOLOv7InferenceNode::Lidar2camNode() : rclcpp::Node("yolov7_inference_node")
{
    // TODO: Make _engine_path a launch parameter
    _engine_path = "/home/roar/ART/perception/model_trials/NVIDIA_AI_IOT_tensorrt_yolov7/yolo_deepstream/tensorrt_yolov7/build/yolov7PTQ.engine";

    // Initialize the YOLOv7 Object
    _yolov7 = std::make_unique<Yolov7>(_engine_path);

    // Declare a pointer to bgr msgs in CV2 format
    _bgr_imgs = std::make_shared<std::vector<cv::Mat>>();
    // TODO: Relook at how many images you want to store here when generalizing to multiple cameras.
    _bgr_imgs->reserve(6);


    /* ------------------------------- QOS Profile ------------------------------ */
    rmw_qos_profile_t qos_profile = rmw_qos_profile_sensor_data;
    auto sensor_msgs_qos = rclcpp::QoS(rclcpp::QoSInitialization(qos_profile.history, 1), qos_profile);

    // TODO: Make the camera topic variable and do synchronous subscription over 6 topics.
    /* --------------------------- Image Subscription --------------------------- */
    _front_left_image_sub.subscribe(this, "vimba_front_left/image", qos_profile);
    _front_left_center_image_sub.subscribe(this, "vimba_front_left_center/image", qos_profile);
    _front_right_center_image_sub.subscribe(this, "vimba_front_right_center/image", qos_profile);
    _front_right_image_sub.subscribe(this, "vimba_front_right/image", qos_profile);
    _rear_right_image_sub.subscribe(this, "vimba_rear_right/image", qos_profile);
    _rear_left_image_sub.subscribe(this, "vimba_rear_left/image", qos_profile);

    /* -------------------- Filter size for exact time queue -------------------- */
    // TODO: Consider making this a rosparam
    _msg_filter_queue_size = this->declare_parameter("ms_queue_size", 1);

    /* ------------------------- Initialize Placeholders ------------------------ */
    using std::placeholders::_1;
    using std::placeholders::_2;
    using std::placeholders::_3;
    using std::placeholders::_4;
    using std::placeholders::_5;
    using std::placeholders::_6;
    _sync_ptr = std::make_shared<Sync>( SyncPolicy(this->get_clock()),   
                                        _front_left_image_sub,
                                        _front_left_center_image_sub,
                                        _front_right_center_image_sub,
                                        _front_right_image_sub,
                                        _rear_right_image_sub,
                                        _rear_left_image_sub);

    _sync_ptr->registerCallback(std::bind(
    &YOLOv7InferenceNode::sync_callback, this, _1, _2, _3, _4, _5, _6));

    /* ----------------------------- Image Publisher ---------------------------- */
    // TODO: DEBUG ONLY. REMOVE FROM PRODUCTION CODE.
    _camera_img_with_det_pub = this->create_publisher<sensor_msgs::msg::Image>(
        "/vimba_front_left_center/det_image",
        sensor_msgs_qos
    );

    /* --------------------------- Detection Publisher -------------------------- */
    _detection_pub = this->create_publisher<vision_msgs::msg::Detection2DArray>(
        "/vimba_front_left_center/det3d",
        sensor_msgs_qos
    );
}

void YOLOv7InferenceNode::sync_callback(
    const sensor_msgs::msg::Image::ConstSharedPtr & front_left,
    const sensor_msgs::msg::Image::ConstSharedPtr & front_left_center,
    const sensor_msgs::msg::Image::ConstSharedPtr & front_right_center,
    const sensor_msgs::msg::Image::ConstSharedPtr & front_right,
    const sensor_msgs::msg::Image::ConstSharedPtr & rear_right,
    const sensor_msgs::msg::Image::ConstSharedPtr & rear_left
)
{
    auto current_time = this->now();
    /* ----------------------- Convert all messages to cv2 ---------------------- */
    _cv_ptr_front_left          = this->cv_bridge_convert(front_left); 
    _cv_ptr_front_left_center   = this->cv_bridge_convert(front_left_center); 
    _cv_ptr_front_right_center  = this->cv_bridge_convert(front_right_center); 
    _cv_ptr_front_right         = this->cv_bridge_convert(front_right); 
    _cv_ptr_rear_right          = this->cv_bridge_convert(rear_right); 
    _cv_ptr_rear_left           = this->cv_bridge_convert(rear_left); 

    // Assign the image to memory allocated to it, if unassigned, otherwise overwrite
    if (_bgr_imgs->empty())
    {
        _bgr_imgs->push_back(_cv_ptr_front_left->image);
        _bgr_imgs->push_back(_cv_ptr_front_left_center->image);
        _bgr_imgs->push_back(_cv_ptr_front_right_center->image);
        _bgr_imgs->push_back(_cv_ptr_front_right->image);
        _bgr_imgs->push_back(_cv_ptr_rear_right->image);
        _bgr_imgs->push_back(_cv_ptr_rear_left->image);
    }
    else
    {
        _bgr_imgs->at(0) = _cv_ptr_front_left->image;
        _bgr_imgs->at(1) = _cv_ptr_front_left_center->image;
        _bgr_imgs->at(2) = _cv_ptr_front_right_center->image;
        _bgr_imgs->at(3) = _cv_ptr_front_right->image;
        _bgr_imgs->at(4) = _cv_ptr_rear_right->image;
        _bgr_imgs->at(5) = _cv_ptr_rear_left->image;
    }

    // Preprocess the images
    _yolov7->preProcess(*_bgr_imgs);

    // Run Inference
    _yolov7->infer();

    // Run NMS & PostProcess
    _nmsresults = _yolov7->PostProcess();

    // Initialize a detection 2D array
    vision_msgs::msg::Detection2DArray det2d_array;

    //Assign this message the stamp of the latest message received.
    det2d_array.header.stamp = current_time;

    // Accumulating the headers into an array
    std::vector<std_msgs::msg::Header> _headers = {
        front_left->header,
        front_left_center->header,
        front_right_center->header,
        front_right->header,
        rear_right->header,
        rear_left->header
    };

    det2d_array.header.frame_id   = "";
    det2d_array.header.stamp = current_time;

    // Iterate through for each detection
    for(int detection_index = 0; detection_index < _nmsresults.size(); detection_index++)
    {
        // Draw the bounding boxes on the image (for checking only)
        Yolov7::DrawBoxesonGraph(_bgr_imgs->at(detection_index), _nmsresults[detection_index]);

        // Iterate through each detection box and add it to be published
        for(int i = 0; i < _nmsresults[detection_index].size(); ++i)
        {
            auto& ibox          = _nmsresults[detection_index][i];
            float left          = ibox[0];
            float top           = ibox[1];
            float right         = ibox[2];
            float bottom        = ibox[3];
            int class_label     = ibox[4];
            float confidence    = ibox[5];

            // Create a Detection2D message
            vision_msgs::msg::Detection2D det2d;

            // Set the bounding box
            det2d.bbox.center.x = (left + right) / 2;
            det2d.bbox.center.y = (top + bottom) / 2;
            det2d.bbox.size_x   = right - left;
            det2d.bbox.size_y   = bottom - top;

            // Set header per box
            det2d.header = _headers[detection_index];
            
            // Set the class label and confidence
            vision_msgs::msg::ObjectHypothesisWithPose result;
            result.hypothesis.class_id = std::to_string(class_label);
            result.hypothesis.score = confidence;
            det2d.results.push_back(result);

            // Add the Detection2D message to the Detection2DArray
            det2d_array.detections.push_back(det2d);
            
        }
        _cv_ptr->image = s->at(detection_index);
        _camera_img_with_det_pub->publish(*(_cv_ptr->toImageMsg()).get());
    
    }

    // Publish the Detection2DArray if the size is greater than 0
    if (det2d_array.detections.size() > 0)
        _det2d_array_pub->publish(det2d_array);
}

// ros::Time YOLOv7InferenceNode::find_max_stamp(
//     const sensor_msgs::msg::Image::ConstSharedPtr & msg0,
//     const sensor_msgs::msg::Image::ConstSharedPtr & msg1,
//     const sensor_msgs::msg::Image::ConstSharedPtr & msg2,
//     const sensor_msgs::msg::Image::ConstSharedPtr & msg3,
//     const sensor_msgs::msg::Image::ConstSharedPtr & msg4,
//     const sensor_msgs::msg::Image::ConstSharedPtr & msg5
// )
// {
//     auto max_stamp = msg0->header.stamp.second;
//     max_stamp = std::max(max_stamp, msg1->header.stamp);
//     max_stamp = std::max(max_stamp, msg2->header.stamp);
//     max_stamp = std::max(max_stamp, msg3->header.stamp);
//     max_stamp = std::max(max_stamp, msg4->header.stamp);
//     max_stamp = std::max(max_stamp, msg5->header.stamp);

//     return max_stamp;
// }

cv_bridge::CvImagePtr YOLOv7InferenceNode::cv_bridge_convert(const sensor_msgs::msg::Image::ConstSharedPtr & msg)
{
    /* ------------------ Receive msg and convert to cv2 format ----------------- */
    try
    {
        _cv_ptr = cv_bridge::toCvCopy(msg, msg->encoding);
    }
    catch (cv_bridge::Exception& e)
    {
        RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
        return cv_bridge::CvImagePtr(NULL);
    }
    return _cv_ptr;
}

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<YOLOv7InferenceNode>());
  rclcpp::shutdown();
  return 0;
}
