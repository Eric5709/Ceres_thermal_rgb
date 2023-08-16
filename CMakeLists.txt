cmake_minimum_required(VERSION 3.8)
project(ceres_calib)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(PCL REQUIRED)
find_package(Ceres REQUIRED)
find_package(GTSAM REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Boost REQUIRED COMPONENTS filesystem program_options)
find_package(rclcpp REQUIRED)
find_package(cv_bridge REQUIRED)

find_package(OpenMP)
if (OPENMP_FOUND)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # uncomment the line when a copyright and license is not present in all source files
  #set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # uncomment the line when this package is not in a git repo
  #set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

include_directories(include/ceres_calib)
include_directories(${PCL_INCLUDE_DIRS} ${CERES_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})


add_executable(${PROJECT_NAME}_optimize src/ceres_optimize_pose.cpp)
ament_target_dependencies(${PROJECT_NAME}_optimize rclcpp PCL)

add_executable(${PROJECT_NAME}_ransac src/ransac.cpp)
ament_target_dependencies(${PROJECT_NAME}_ransac rclcpp PCL)

target_link_libraries(${PROJECT_NAME}_optimize fmt ${PCL_LIBRARIES} ${OpenCV_LIBS} ${CERES_LIBRARIES})
target_include_directories(${PROJECT_NAME}_optimize PUBLIC
        include
        thirdparty/json/include
        thirdparty/Bonxai/include
        thirdparty/nanoflann/include
        thirdparty/Sophus
        thirdparty/dfo
        ${Boost_INCLUDE_DIRS}
        ${CERES_INCLUDE_DIRS}
        ${GTSAM_INCLUDE_DIRS}
        ${PCL_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(${PROJECT_NAME}_ransac fmt ${PCL_LIBRARIES} ${OpenCV_LIBS} ${CERES_LIBRARIES})
target_include_directories(${PROJECT_NAME}_ransac PUBLIC
        include
        thirdparty/json/include
        thirdparty/Bonxai/include
        thirdparty/nanoflann/include
        thirdparty/Sophus
        thirdparty/dfo
        ${Boost_INCLUDE_DIRS}
        ${CERES_INCLUDE_DIRS}
        ${GTSAM_INCLUDE_DIRS}
        ${PCL_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
)
ament_package()
