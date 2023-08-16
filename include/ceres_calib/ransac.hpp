//#pragma once
//
//#include <vector>
//#include <Eigen/Core>
//#include <Eigen/Geometry>
//
//struct PoseEstimationParams {
//    PoseEstimationParams(){
//        ransac_iterations = 8000;
//        ransac_error_thresh = 5.0;
//        robust_kernel_width = 10.0;
//    }
//
//    int ransac_iterations;
//    double ransac_error_thresh;
//    double robust_kernel_width;
//};
//
//class PoseEstimation {
//public:
//    PoseEstimation(const PoseEstimationParams& params = PoseEstimationParams());
//    ~PoseEstimation();
//
//    Eigen::Isometry3d
//    estimate(const camera::GenericCameraBase::ConstPtr& proj, const std::vector<std::pair<Eigen::Vector2d, Eigen::Vector4d>>& correspondences, std::vector<bool>* inliers = nullptr);
//
//private:
//    Eigen::Matrix3d estimate_rotation_ransac(
//            const camera::GenericCameraBase::ConstPtr& proj,
//            const std::vector<std::pair<Eigen::Vector2d, Eigen::Vector4d>>& correspondences,
//            std::vector<bool>* inliers);
//
//    Eigen::Isometry3d estimate_pose_lsq(
//            const camera::GenericCameraBase::ConstPtr& proj,
//            const std::vector<std::pair<Eigen::Vector2d, Eigen::Vector4d>>& correspondences,
//            const Eigen::Isometry3d& T_camera_lidar);
//
//private:
//    const PoseEstimationParams params;
//};
//
//};