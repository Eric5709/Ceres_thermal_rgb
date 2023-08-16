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


bool save_csv = true;
int target = 1;
https://code-with-me.global.jetbrains.com/LrLePfE0okgzk5TLVd5YjA#p=CL&fp=C1C6B17D9AD730570874DDD267B1D8D17203A3C15AAA5D1C0CA6F2B417284932&newUi=false

class ReprojectionCost {
public:
    ReprojectionCost(Eigen::Vector3d& thermal_direction, Eigen::Vector2d& point_rgb, Eigen::Vector4d& intrinsic) :
            thermal_direction(thermal_direction),
            point_rgb(point_rgb),
            intrinsic(intrinsic) {}

    ~ReprojectionCost() {}

    template <typename T>
    // T = ceres::Jet<double, 7>
    bool operator()(const T* const T_camera_thermal_params, T* residual) const {
        //both are in the form of ceres::Jet
        const Eigen::Map<Sophus::SE3<T> const> T_camera_thermal(T_camera_thermal_params);
//        std::cout<<typeid(T_camera_thermal).name()<<std::endl;

        const Eigen::Matrix<T, 3, 1> pt_camera = T_camera_thermal * thermal_direction;
//        std::cout<<typeid(pt_camera).name()<<std::endl;

        const auto fx = (intrinsic[0]);
        const auto fy = (intrinsic[1]);
        const auto cx = (intrinsic[2]);
        const auto cy = (intrinsic[3]);

//        std::cout<<"******"<<std::endl;
//        std::cout<<pt_camera[0] <<", "<< pt_camera[1]<<", "<<pt_camera[2]<<std::endl;
//        std::cout<<pt_camera.transpose()<<std::endl;
//        std::cout<<std::endl;
        const T projected_x = fx * pt_camera[0] / pt_camera[2] + cx;
        const T projected_y = fy * pt_camera[1] / pt_camera[2] + cy;

//        std::cout<< projected_x << "\n"<<projected_y<<std::endl;

        residual[0] = projected_x - T(point_rgb[0]);
        residual[1] = projected_y - T(point_rgb[1]);
        return true;
    }

private:
    const Eigen::Vector3d thermal_direction;
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


    std::vector<std::pair<Eigen::Vector2d, Eigen::Vector2d>> correspondences;
    for (int i = 0; i < matches.size(); i++){
        if (matches[i] < 0){
            continue;
        }
        Eigen::Vector2i kp0(kpts0[2 * i], kpts0[2 * i + 1]);
        Eigen::Vector2i kp1(kpts1[2 * matches[i]], kpts1[2 * matches[i] + 1]);

        correspondences.emplace_back(kp0.cast<double>(), kp1.cast<double>());
    }
    return correspondences;
}
EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE
EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE
// 3d direction vector projection to 2d camera's coordinate adding distortion.
Eigen::Vector2d world2image(Eigen::Vector3d& pt_3d, Eigen::Vector4d intrinsic, Eigen::VectorXd distortion){
    Eigen::Vector2d pt_2d = Eigen::Vector2d::Zero();
    pt_2d = pt_2d.normalized();
    pt_2d[0] = pt_3d[0] / pt_3d[2];
    pt_2d[1] = pt_3d[1] / pt_3d[2];

    double k1 = distortion[0];
    double k2 = distortion[1];
    double k3 = distortion[4];

    double p1 = distortion[2];
    double p2 = distortion[3];

    double x2 = std::pow(pt_2d[0], 2);
    double y2 = std::pow(pt_2d[1], 2);

    double r2 = x2 + y2;
    double r4 = std::pow(r2, 2);
    double r6 = std::pow(r2, 3);

    double r_coeff = 1.0 + k1 * r2 + k2 * r4 + k3 * r6;
    double t_coeff1 = 2.0 * pt_2d[0] * pt_2d[1];
    double t_coeff2 = r2 + 2.0 * x2;
    double t_coeff3 = r2 + 2.0 * y2;

    pt_2d[0] = r_coeff * pt_2d[0] + p1 * t_coeff1 + p2 * t_coeff2;
    pt_2d[1] = r_coeff * pt_2d[1] + p1 * t_coeff3 + p2 * t_coeff1;

    const auto& fx = intrinsic[0];
    const auto& fy = intrinsic[1];
    const auto& cx = intrinsic[2];
    const auto& cy = intrinsic[3];

    return {fx * pt_2d[0] + cx, fy * pt_2d[1] + cy};
}



Eigen::Vector3d estimate_direction(Eigen::Vector2d& pt_2d, Eigen::Vector4d intrinsic, Eigen::VectorXd distortion){
    const auto to_dir = [](const Eigen::Vector2d& x) {
        return Eigen::AngleAxisd(x[0], Eigen::Vector3d::UnitX()) * Eigen::AngleAxisd(x[1], Eigen::Vector3d::UnitY()) * Eigen::Vector3d::UnitZ();
    };
    const auto f = [&](const Eigen::Vector2d& x){
        Eigen::Vector3d dir = to_dir(x);
        Eigen::Vector2d x_2d = world2image(dir, intrinsic, distortion);
        const double err =(pt_2d - x_2d).squaredNorm();
        return std::isfinite(err) ? err : std::numeric_limits<double>::max();
    };

    dfo::NelderMead<2>::Params params;
    dfo::NelderMead<2> optimizer(params);
    auto result = optimizer.optimize(f, Eigen::Vector2d::Zero());

//    std::cout<<"*******Result*******"<<std::endl;
//    std::cout<<"result.x         : "<<result.x.transpose()<<std::endl;
//    std::cout<<"to_dir(result.x) : "<<to_dir(result.x)<<std::endl;
    return to_dir(result.x);
}


Eigen::Matrix3d estimate_rotation_ransac(std::vector<std::pair<Eigen::Vector2d, Eigen::Vector2d>> correspondences, std::vector<std::pair<Eigen::Vector2d, Eigen::Vector2d>>& inliers, Eigen::Vector4d& RGB_intrin, Eigen::Vector4d& thermal_intrin,
                                         Eigen::VectorXd& distortion){
    std::vector<std::tuple<Eigen::Vector2d, Eigen::Vector2d, Eigen::Vector2d>> corres_csv;

    // the direction vector's magnitude is 1.
    std::vector<Eigen::Vector4d> direction_camera(correspondences.size());
    std::vector<Eigen::Vector4d> direction_thermal(correspondences.size());

    for (int i = 0; i < correspondences.size(); i++){
        direction_camera[i] << estimate_direction(correspondences[i].first, RGB_intrin, distortion), 0.0;
        direction_thermal[i] << estimate_direction(correspondences[i].second, thermal_intrin, distortion), 0.0;
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
        // The result of R_camera_thermal is almost in the form of symmetric matrix. I think it is because the two camera's coordinate is the same.
        // So A * R = B,
        //    B * R = A.
        const Eigen::Matrix3d R_camera_thermal = U * S * V.transpose();
        return R_camera_thermal;
    };


// ******************** RANSAC ********************
//    std::cout << "estimating rotation using RANSAC" << std::endl;

    const double thresh_ransac = 80.0;
    Eigen::Matrix4d best_R = Eigen::Matrix4d::Zero();
    int best_num_inliers = 0;


    for (int i = 0; i < correspondences.size(); i++){
        Eigen::Matrix4d R_ct = Eigen::Matrix4d::Zero();
        R_ct.topLeftCorner<3,3>() = find_rotation(i);
        int num_inliers = 0;
        for(int j = 0; j < correspondences.size(); j++){
            Eigen::Vector4d direction_cam = R_ct * direction_thermal[j];
            Eigen::Vector3d pt_3d_copy = direction_cam.head<3>();
            Eigen::Vector2d pt_2d = world2image(pt_3d_copy, RGB_intrin, distortion);

            if((correspondences[j].first - pt_2d).squaredNorm() < thresh_ransac){
                num_inliers++;
            }
        }

        if (num_inliers > best_num_inliers){
            best_num_inliers = num_inliers;
            best_R = R_ct;
        }
    }

    // make inliers
    for(int j = 0; j < correspondences.size(); j++){
        // best_R = rotation from thermal to camera
        Eigen::Vector4d direction_cam = best_R * direction_thermal[j];
        Eigen::Vector3d pt_3d_copy = direction_cam.head<3>();
        Eigen::Vector2d pt_2d = world2image(pt_3d_copy, RGB_intrin, distortion);

        if((correspondences[j].first - pt_2d).squaredNorm() < thresh_ransac){
            std::cout<<direction_cam.transpose() - direction_camera[j].transpose()<<std::endl;
            std::cout<<pt_2d.transpose() - correspondences[j].first.transpose()<<std::endl;
            inliers.emplace_back(correspondences[j].first, correspondences[j].second);
            corres_csv.emplace_back(correspondences[j].first, correspondences[j].second, pt_2d);
        }
    }

    //save to csv
    if(save_csv) {
        std::ofstream csv_file("/home/jsh/Diter.csv");
        if (csv_file.is_open()) {
            csv_file << "rgb_x,rgb_y,thermal_x,thermal_y,projected_x,projected_y\n";
            for (const auto &entry: corres_csv) {
                const Eigen::Vector2d &rgb_coords = std::get<0>(entry);
                const Eigen::Vector2d &thermal_coords = std::get<1>(entry);
                const Eigen::Vector2d &projected_thermal = std::get<2>(entry);

                csv_file << rgb_coords.x() << "," << rgb_coords.y() << ","
                         << thermal_coords.x() << "," << thermal_coords.y() << ","
                         << projected_thermal.x() << "," << projected_thermal.y() << "\n";
            }
            csv_file.close();
        }
    }




    std::cout<<"--- T (RANSAC) ---"<<std::endl;
    std::cout << best_R << std::endl;
    std::cout << "num_inliers: " << best_num_inliers << " / " << correspondences.size() << "\n"<< std::endl;

    return best_R.block<3, 3>(0, 0);
}

Eigen::Isometry3d estimate_pose_lsq(std::vector<std::pair<Eigen::Vector2d, Eigen::Vector2d>>& inliers, const Eigen::Isometry3d& init_T_camera_thermal, Eigen::Vector4d& RGB_intrinsic, Eigen::Vector4d& thermal_intrinsic,
                                    Eigen::VectorXd& distortion) {
    Sophus::SE3d T_camera_thermal = Sophus::SE3d(init_T_camera_thermal.matrix());
    ceres::Problem problem;
    problem.AddParameterBlock(T_camera_thermal.data(), Sophus::SE3d::num_parameters, new Sophus::Manifold<Sophus::SE3>());

    for (auto &[pt_c, pt_t]: inliers) {
        Eigen::Vector3d thermal_direction = estimate_direction(pt_t, thermal_intrinsic, distortion);
        auto reproject_error = new ReprojectionCost(thermal_direction, pt_c, RGB_intrinsic);
        auto ad_cost = new ceres::AutoDiffCostFunction<ReprojectionCost, 2, Sophus::SE3d::num_parameters>(reproject_error);
        auto loss = new ceres::CauchyLoss(1);
        problem.AddResidualBlock(ad_cost, loss, T_camera_thermal.data());
    }


    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    std::cout << summary.BriefReport() << std::endl;

    return Eigen::Isometry3d(T_camera_thermal.matrix());
}


Eigen::Isometry3d inverse_trans(Eigen::Isometry3d input_matrix) {
    Eigen::Isometry3d output;
    Eigen::Matrix3d rotation = input_matrix.linear();
    Eigen::Vector3d translation = input_matrix.translation();
    Eigen::Matrix3d inverse_rotation = rotation.inverse();
    Eigen::Vector3d inverse_translation = -inverse_rotation * translation;
    output.linear() = inverse_rotation;
    output.translation() = inverse_translation;
}



int main()
{
    Eigen::Vector4d RGB_intrinsic;
    Eigen::Vector4d thermal_intrinsic;
    Eigen::Vector4d Diter_RGB_intrinsic;
    Eigen::Vector4d Diter_thermal_intrinsic;
    Eigen::Vector4d sthereo_RGB_intrinsic_left;
    Eigen::Vector4d sthereo_RGB_intrinsic_right;
    Eigen::Vector4d sthereo_thermal_intrinsic;

    Diter_RGB_intrinsic << 920.441, 919.068, 635.317, 340.274;                   // Diter rgb
    Diter_thermal_intrinsic << 417.873, 455.522, 325.936, 230.134;               // Diter thermal

    sthereo_RGB_intrinsic_left << 788.41328, 790.92597, 633.40907, 237.46688;    // sthereo rgb left
    sthereo_RGB_intrinsic_right << 793.242, 793.882, 610.081, 265.805;           // sthereo rgb right
    sthereo_thermal_intrinsic << 429.43288, 429.53142, 311.11923, 266.128175;    // sthereo thermal left
    std::vector<std::pair<Eigen::Vector2d, Eigen::Vector2d>> inliers;

    Eigen::VectorXd distortion(5);
    Eigen::VectorXd Diter_distortion(5);
    Eigen::VectorXd sthereo_distortion(5);

    Diter_distortion << 0.090986, -0.183563, -0.004613, -0.000663, 0.000000;   // Diter distortion
    sthereo_distortion << -0.054981, 0.042759, -0.011477, 0.009880, 0.000000;  // sthereo distortion

    // Output1 : Diter rgb-thermal
    // Output2 : sthereo rgb-thermal
    // Output3 : sthereo rgb-rgb
    if(target == 0) {
        RGB_intrinsic = Diter_RGB_intrinsic;
        thermal_intrinsic = Diter_thermal_intrinsic;
        distortion = Diter_distortion;

        std::string json_path;
        if(save_csv)
            json_path = "/media/jsh/2060b9c6-3d48-4115-ae61-3a2b13150f93/Diter_full/calibration/Output/json_csv/";
        else
            json_path = "/media/jsh/2060b9c6-3d48-4115-ae61-3a2b13150f93/Diter_full/calibration/Output/json/";

        std::vector<cv::String> json_filenames;
        cv::glob(json_path, json_filenames);
        std::vector<std::pair<Eigen::Vector2d, Eigen::Vector2d>> correspondences;

        Eigen::Isometry3d cal_R = Eigen::Isometry3d::Identity();

        for (int i = 0; i < json_filenames.size(); i++) {
            std::string json_filename = json_filenames[i];
            auto corrs = read_correspondences(json_filename);
            correspondences.insert(correspondences.end(), corrs.begin(), corrs.end());
        }

        cal_R.linear() = estimate_rotation_ransac(correspondences, inliers, RGB_intrinsic, thermal_intrinsic,
                                                  distortion);
        std::cout << "--- T (LSQ) ---" << std::endl;
//    for (const auto& point : inliers){
//        std::cout<<point.first.transpose()<<",  "<<point.second.transpose()<<std::endl;
//    }

        cal_R = estimate_pose_lsq(inliers, cal_R, RGB_intrinsic, thermal_intrinsic, distortion);
        std::cout << cal_R.matrix() << std::endl;

        if (save_csv)
            std::cout << "Target :  Diter, csv is saved" << std::endl;
        else
            std::cout << "Target :  Diter, csv is not saved" << std::endl;
    }


    else if(target == 1) {
        RGB_intrinsic = sthereo_RGB_intrinsic_left;
        thermal_intrinsic = sthereo_thermal_intrinsic;
        distortion = sthereo_distortion;

        std::string json_path;
        if(save_csv)
            json_path = "/media/jsh/2060b9c6-3d48-4115-ae61-3a2b13150f93/Diter_full/calibration/Output2/json_csv/";
        else
            json_path = "/media/jsh/2060b9c6-3d48-4115-ae61-3a2b13150f93/Diter_full/calibration/Output2/json/";

        std::vector<cv::String> json_filenames;
        cv::glob(json_path, json_filenames);
        std::vector<std::pair<Eigen::Vector2d, Eigen::Vector2d>> correspondences;

        Eigen::Isometry3d cal_R = Eigen::Isometry3d::Identity();

        for (int i = 0; i < json_filenames.size(); i++) {
            std::string json_filename = json_filenames[i];
            auto corrs = read_correspondences(json_filename);
            correspondences.insert(correspondences.end(), corrs.begin(), corrs.end());
        }

        cal_R.linear() = estimate_rotation_ransac(correspondences, inliers, RGB_intrinsic, thermal_intrinsic,
                                                  distortion);
        std::cout << "--- T (LSQ) ---" << std::endl;
        std::cout<<inliers.size()<<std::endl;

        cal_R = estimate_pose_lsq(inliers, cal_R, RGB_intrinsic, thermal_intrinsic, distortion);
        std::cout << cal_R.matrix() << std::endl;

        if (save_csv)
            std::cout << "Target :  sthereo_c2t, csv is saved" << std::endl;
        else
            std::cout << "Target :  sthereo_c2t, csv is not saved" << std::endl;
    }

    else {
        RGB_intrinsic = sthereo_RGB_intrinsic_left;
        thermal_intrinsic = sthereo_RGB_intrinsic_right;
        distortion = sthereo_distortion;

        std::string json_path;
        if(save_csv)
            json_path = "/media/jsh/2060b9c6-3d48-4115-ae61-3a2b13150f93/Diter_full/calibration/Output3/json_csv/";
        else
            json_path = "/media/jsh/2060b9c6-3d48-4115-ae61-3a2b13150f93/Diter_full/calibration/Output3/json/";

        std::vector<cv::String> json_filenames;
        cv::glob(json_path, json_filenames);
        std::vector<std::pair<Eigen::Vector2d, Eigen::Vector2d>> correspondences;

        Eigen::Isometry3d cal_R = Eigen::Isometry3d::Identity();

        for (int i = 0; i < json_filenames.size(); i++) {
            std::string json_filename = json_filenames[i];
            auto corrs = read_correspondences(json_filename);
            correspondences.insert(correspondences.end(), corrs.begin(), corrs.end());
        }

        cal_R.linear() = estimate_rotation_ransac(correspondences, inliers, RGB_intrinsic, thermal_intrinsic,
                                                  distortion);
        std::cout << "--- T (LSQ) ---" << std::endl;
//    for (const auto& point : inliers){
//        std::cout<<point.first.transpose()<<",  "<<point.second.transpose()<<std::endl;
//    }

        cal_R = estimate_pose_lsq(inliers, cal_R, RGB_intrinsic, thermal_intrinsic, distortion);
        std::cout << cal_R.matrix() << std::endl;

        if (save_csv)
            std::cout << "Target :  sthereo_c2c, csv is saved" << std::endl;
        else
            std::cout << "Target :  sthereo_c2c, csv is not saved" << std::endl;
    }


    return 0;
}