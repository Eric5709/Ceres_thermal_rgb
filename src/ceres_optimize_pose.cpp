#include <iostream>
#include <random>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <ceres/ceres.h>
#include <ceres/problem.h>
#include <ceres/rotation.h>
#include <sophus/se3.hpp>
#include <sophus/ceres_manifold.hpp>
#include <nlohmann/json.hpp>
#include <boost/program_options.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>

#include <../thirdparty/dfo/nelder_mead.hpp>
//#include "../include/ceres_calib/reprojection_cost.hpp"


class ReprojectionCost {
public:
    ReprojectionCost(Eigen::Vector3d& point_thermal, Eigen::Vector2d& point_rgb, Eigen::Vector4d& intrinsic) :
            point_thermal(point_thermal),
            point_rgb(point_rgb),
            intrinsic(intrinsic) {}

    ~ReprojectionCost() {}

    template <typename T>
    bool operator()(const T* const T_camera_lidar_params, T* residual) const {
        //both are in the form of ceres::Jet
        const Eigen::Map<Sophus::SE3<T> const> T_camera_lidar(T_camera_lidar_params);
        const Eigen::Matrix<T, 3, 1> pt_camera = T_camera_lidar * point_thermal;

        const T fx = T(intrinsic[0]);
        const T fy = T(intrinsic[1]);
        const T cx = T(intrinsic[2]);
        const T cy = T(intrinsic[3]);

        const T projected_x = fx * pt_camera[0] / pt_camera[2] + cx;
        const T projected_y = fy * pt_camera[1] / pt_camera[2] + cy;
//        std::cout<<"******"<<std::endl;
//        std::cout<< projected_x << ",  "<<projected_y<<std::endl;
        residual[0] = projected_x - T(point_rgb[0]);
        residual[1] = projected_y - T(point_rgb[1]);
        return true;
    }

private:
    const Eigen::Vector3d point_thermal;
    const Eigen::Vector2d point_rgb;
    const Eigen::Vector4d intrinsic;
};


std::vector<std::pair<Eigen::Vector2d, Eigen::Vector2d>> read_correspondences(const std::string& json_filename){
    std::ifstream matches_ifs(json_filename);
    if(!matches_ifs){
        std::cerr<<"ERROR : Failed to open "<<std::endl;
        abort();
    }

    nlohmann::json matching_result;
    matches_ifs >>matching_result;

    std::vector<int> kpts0         = matching_result["kpts0"];
    std::vector<int> kpts1         = matching_result["kpts1"];
    std::vector<int> matches       = matching_result["matches"];
    std::vector<double> confidence = matching_result["confidence"];

//    std::cout<<json_filename.substr(json_filename.size() - 19, 6)<<std::endl;
//    std::cout<<"Key point1 : " <<kpts0.size()<<std::endl;
//    std::cout<<"Key point2 : " <<kpts1.size()<<std::endl;
//    std::cout<<"Matches    : " <<matches.size()<<std::endl;
//    std::cout<<confidence.size()<<std::endl;
    int count_not_matched = 0;
    int count_matched = 0;

    std::vector<std::pair<Eigen::Vector2d, Eigen::Vector2d>> correspondences;
    for (int i = 0; i < matches.size(); i++){
        if (matches[i] < 0){
//            std::cout<< i <<"-th iteration :  Not matched" << std::endl;
            count_not_matched++;
            continue;
        }
        count_matched ++;
        Eigen::Vector2i kp0(kpts0[2 * i], kpts0[2 * i + 1]);
        Eigen::Vector2i kp1(kpts1[2 * matches[i]], kpts1[2 * matches[i] + 1]);

//        std::cout<< i <<"-th iteration :  "<< kp0.transpose() << ",   "<< kp1.transpose()<<"      , matches : "<<matches[i]<<std::endl;
        correspondences.emplace_back(kp0.cast<double>(), kp1.cast<double>());
    }
//    std::cout<<count_matched<<std::endl;
//    std::cout<<count_not_matched<<std::endl;

    return correspondences;
//    std::cout<<correspondences.size()<<std::endl;
}



Eigen::Vector2d world2image(Eigen::Vector3d& pt_3d, Eigen::Vector4d intrinsic){
    Eigen::Vector2d pt_2d;
    pt_2d[0] = pt_3d[0] / pt_3d[2];
    pt_2d[1] = pt_3d[1] / pt_3d[2];

    const auto& fx = intrinsic[0];
    const auto& fy = intrinsic[1];
    const auto& cx = intrinsic[2];
    const auto& cy = intrinsic[3];

    return {fx * pt_2d[0] + cx, fy * pt_2d[1] + cy};
}


Eigen::Vector3d estimate_direction(Eigen::Vector2d& pt_2d, Eigen::Vector4d intrinsic){
    const auto to_dir = [](const Eigen::Vector2d& x) {
        return Eigen::AngleAxisd(x[0], Eigen::Vector3d::UnitX()) * Eigen::AngleAxisd(x[1], Eigen::Vector3d::UnitY()) * Eigen::Vector3d::UnitZ();
    };
    const auto f = [&](const Eigen::Vector2d& x){
        Eigen::Vector3d dir = to_dir(x);
        Eigen::Vector2d x_2d = world2image(dir, intrinsic);
        const double err =(pt_2d - x_2d).squaredNorm();
        return std::isfinite(err) ? err : std::numeric_limits<double>::max();
    };

    dfo::NelderMead<2>::Params params;
    dfo::NelderMead<2> optimizer(params);
    auto result = optimizer.optimize(f, Eigen::Vector2d::Zero());

    return to_dir(result.x);
}


Eigen::Matrix3d estimate_rotation_ransac(std::vector<std::pair<Eigen::Vector2d, Eigen::Vector2d>> correspondences, std::vector<bool>* inliers, Eigen::Vector4d RGB_intrin, Eigen::Vector4d thermal_intrin){
    std::vector<Eigen::Vector4d> direction_camera(correspondences.size());
    std::vector<Eigen::Vector4d> direction_thermal(correspondences.size());


    for (int i = 0; i < correspondences.size(); i++){
        direction_camera[i] << estimate_direction(correspondences[i].first, RGB_intrin), 0.0;
        direction_thermal[i] << estimate_direction(correspondences[i].second, thermal_intrin), 0.0;
//        auto reproject_err = new ReprojectionCost(direction_thermal[i].head<3>(), correspondences[i].first, RGB_intrin);
//        std::cout<<reproject_err<<std::endl;
    }

    const auto find_rotation = [&](const int index){
        Eigen::Vector3d A = direction_camera[index].head<3>();
        Eigen::Vector3d B = direction_thermal[index].head<3>();

        const Eigen::Matrix3d AB = A * B.transpose();

        Eigen::JacobiSVD<Eigen::Matrix3d> svd(AB, Eigen::ComputeFullU | Eigen::ComputeFullV);
        const Eigen::Matrix3d U = svd.matrixU();
        const Eigen::Matrix3d V = svd.matrixV();
        const Eigen::Matrix3d D = svd.singularValues().asDiagonal();
        Eigen::Matrix3d S = Eigen::Matrix3d::Identity();

        double det = U.determinant() * V.determinant();
        if (det < 0.0){
            S(2,2) = -1.0;
        }

        const Eigen::Matrix3d R_camera_thermal = U * S * V.transpose();

//        std::cout<< R_camera_thermal<<std::endl;
        return R_camera_thermal;
    };

//    int iteration = 2000;

// ******************** RANSAC ********************
//    std::cout << "estimating rotation using RANSAC" << std::endl;

    const double thresh_ransac = 100.0;
    Eigen::Matrix4d best_R;
    int best_num_inliers = 0;
    for (int i = 0; i < correspondences.size(); i++){
        Eigen::Matrix4d R_ct = Eigen::Matrix4d::Zero();
        R_ct.topLeftCorner<3,3>() = find_rotation(i);
        int num_inliers = 0;
        for(int j = 0; j < correspondences.size(); j++){
            Eigen::Vector4d direction_cam = R_ct * direction_thermal[j];
            Eigen::Vector3d pt_3d_copy = direction_cam.head<3>();
            Eigen::Vector2d pt_2d = world2image(pt_3d_copy, RGB_intrin);

            if((correspondences[j].first - pt_2d).squaredNorm() < thresh_ransac){
                num_inliers++;
            }
        }

        if (num_inliers > best_num_inliers){
            best_num_inliers = num_inliers;
            best_R = R_ct;
        }
    }
    std::cout<<"--- T (RANSAC) ---"<<std::endl;
    std::cout << best_R << std::endl;
    std::cout << "num_inliers: " << best_num_inliers << " / " << correspondences.size() << "\n"<< std::endl;

//    if (inliers){
//        inliers->resize(correspondences.size());
//        for (int i = 0; i < correspondences.size(); i++){
//            const Eigen::Vector4d direction_cam = best_R * direction_thermal[i];
//            Eigen::Vector3d pt_3d_copy = direction_cam.head<3>();
//            Eigen::Vector2d pt_2d = world2image(pt_3d_copy, RGB_intrin);
//            (*inliers)[i] = (correspondences[i].first - pt_2d).squaredNorm() < thresh_ransac;
//        }
//    }
//    std::cout<<"Process1"<<std::endl;
    return best_R.block<3, 3>(0, 0);
}


Eigen::Isometry3d estimate_pose_lsq(std::vector<std::pair<Eigen::Vector2d, Eigen::Vector2d>>& correspondences, const Eigen::Isometry3d& init_T_camera_thermal, Eigen::Vector4d& RGB_intrinsic, Eigen::Vector4d& thermal_intrinsic) {
    Sophus::SE3d T_ct = Sophus::SE3d(init_T_camera_thermal.matrix());

    ceres::Problem problem;
    problem.AddParameterBlock(T_ct.data(), Sophus::SE3d::num_parameters, new Sophus::Manifold<Sophus::SE3>());

    for (auto &[pt_c, pt_t]: correspondences) {
        Eigen::Vector3d thermal_direction = estimate_direction(pt_t, thermal_intrinsic);
        auto reproject_error = new ReprojectionCost(thermal_direction, pt_c, RGB_intrinsic);
        auto ad_cost = new ceres::AutoDiffCostFunction<ReprojectionCost, 2, Sophus::SE3d::num_parameters>(reproject_error);
        auto loss = new ceres::CauchyLoss(10);
        problem.AddResidualBlock(ad_cost, loss, T_ct.data());
    }


    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    std::cout << summary.BriefReport() << std::endl;

    return Eigen::Isometry3d(T_ct.matrix());
}


int main()
{
    Eigen::Vector4d RGB_intrinsic;
    Eigen::Vector4d thermal_intrinsic;
    RGB_intrinsic << 920.441, 919.068, 635.317, 340.274;
    thermal_intrinsic << 417.873, 455.522, 325.936, 230.134;
    std::vector<bool>* inliers;
    std::string json_path = "/media/jsh/2060b9c6-3d48-4115-ae61-3a2b13150f93/Diter_full/calibration/Output/json/";

    std::vector<cv::String> json_filenames;
    cv::glob(json_path, json_filenames);
    std::vector<std::pair<Eigen::Vector2d, Eigen::Vector2d>> correspondences;

    Eigen::Isometry3d cal_R = Eigen::Isometry3d::Identity();

    for (int i = 0; i < json_filenames.size(); i++) {
        std::string json_filename = json_filenames[i];
        auto corrs = read_correspondences(json_filename);
        correspondences.insert(correspondences.end(), corrs.begin(), corrs.end());
    }

    cal_R.linear() = estimate_rotation_ransac(correspondences, inliers, RGB_intrinsic, thermal_intrinsic);
    std::cout<<"--- T (LSQ) ---"<<std::endl;
    cal_R = estimate_pose_lsq(correspondences, cal_R, RGB_intrinsic, thermal_intrinsic);
    std::cout << cal_R.matrix() <<std::endl;


    return 0;
}