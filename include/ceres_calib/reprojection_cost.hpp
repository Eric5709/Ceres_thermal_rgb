#pragma once

#include <Eigen/Core>
#include <sophus/se3.hpp>
#include <utility>
#include "project.hpp"

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

    residual[0] = projected_x - T(point_rgb[0]);
    residual[1] = projected_y - T(point_rgb[1]);
    return true;
  }

private:
  const Eigen::Vector3d point_thermal;
  const Eigen::Vector2d point_rgb;
  const Eigen::Vector4d intrinsic;
};