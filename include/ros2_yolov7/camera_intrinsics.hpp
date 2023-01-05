#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <map>
#include <string>
#include <vector>
#include <memory>

// Struct to hold the intrinsic parameters of a camera
struct CameraIntrinsics {
    cv::Mat cameraMatrix;
    cv::Mat distortionCoefficients;
};

// Define the intrinsic parameters for 6 cameras
static const CameraIntrinsics vimba_front_left_center_intrinsics = {
    (cv::Mat_<double>(3,3) << 1732.571708, 0.000000, 549.797164, 0.000000, 1731.274561, 295.484988, 0.000000, 0.000000, 1.000000),
    (cv::Mat_<double>(1,5) << -0.272455, 0.268395, -0.005054, 0.000391, 0.000000)
};

static const CameraIntrinsics vimba_front_right_center_intrinsics = {
    (cv::Mat_<double>(3,3) << 1687.074326, 0.000000, 553.239937, 0.000000, 1684.357776, 405.545108, 0.000000, 0.000000, 1.000000),
    (cv::Mat_<double>(1,5) << -0.245135, 0.211710, -0.001225, 0.001920, 0.000000)
};

static const CameraIntrinsics vimba_front_left_intrinsics = {
    (cv::Mat_<double>(3,3) << 251.935177, 0.000000, 260.887279, 0.000000, 252.003440, 196.606218, 0.000000, 0.000000, 1.000000),
    (cv::Mat_<double>(1,5) << -0.181920, 0.050231, -0.000085, 0.001209, 0.000000)
};

static const CameraIntrinsics vimba_front_right_intrinsics = {
    (cv::Mat_<double>(3,3) << 250.433937, 0.000000, 257.769878, 0.000000, 250.638855, 186.572321, 0.000000, 0.000000, 1.000000),
    (cv::Mat_<double>(1,5) << -0.185526, 0.050462, -0.000342, 0.000089, 0.000000)
};

static const CameraIntrinsics vimba_rear_left_intrinsics = {
    (cv::Mat_<double>(3,3) << 250.596029, 0.000000, 258.719088, 0.000000, 250.571031, 190.881036, 0.000000, 0.000000, 1.000000),
    (cv::Mat_<double>(1,5) << -0.179032, 0.047829, 0.000395, -0.000710, 0.000000)
};

static const CameraIntrinsics vimba_rear_right_intrinsics = {
    (cv::Mat_<double>(3,3) << 251.353372, 0.000000, 257.997245, 0.000000, 251.239242, 194.330435, 0.000000, 0.000000, 1.000000),
    (cv::Mat_<double>(1,5) << -0.192458, 0.059495, -0.000126, 0.000092, 0.000000)
};


std::map<std::string, int> CamMap = {
        {"vimba_front_left", 1},
        {"vimba_front_left_center", 2},
        {"vimba_front_right_center", 3},
        {"vimba_front_right", 4},
        {"vimba_rear_right", 5},
        {"vimba_rear_left", 6},
};

std::map<std::string, int> LidarMap = {
        {"front", 1},
        {"left", 2},
        {"right", 3},
};

std::map<int, std::vector<int>> cam2lidar = {
        {1, {1, 2}},
        {2, {1}},
        {3, {1}},
        {4, {1, 3}},
        {5, {3}},
        {6, {2}},
};

struct Transformation {
    Eigen::Matrix3d Translation;
    Eigen::Matrix3d Rotation;
};

std::map<int, std::unique_ptr<sensor_msgs::msg::PointCloud2>> lidar_msg = {
        {1, nullptr},
        {2, nullptr},
        {3, nullptr},
};

std::map<int,Eigen::Matrix3d> CamMatrix;
std::map<std::pair<int, int>, transformation> TransformationMap;
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