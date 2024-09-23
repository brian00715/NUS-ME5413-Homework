
include "map_builder.lua"
include "trajectory_builder.lua"

-- options = {
--   map_builder = MAP_BUILDER,
--   trajectory_builder = TRAJECTORY_BUILDER,
--   map_frame = "map",
--   tracking_frame = "laser_link",
--   published_frame = "base_footprint",
--   odom_frame = "odom",
--   provide_odom_frame = false,
--   publish_frame_projected_to_2d = false,
--   -- Add a hidden parameter 'publish_tracked_pose' for 'evo' plotting
--   publish_tracked_pose = true,
--   use_pose_extrapolator = true,
--   use_odometry = true,
--   use_nav_sat = false,
--   use_landmarks = false,
--   num_laser_scans = 1,
--   num_multi_echo_laser_scans = 0,
--   num_subdivisions_per_laser_scan = 1,
--   num_point_clouds = 0,
--   lookup_transform_timeout_sec = 0.2,
--   submap_publish_period_sec = 0.3,
--   pose_publish_period_sec = 5e-3,
--   trajectory_publish_period_sec = 30e-3,
--   rangefinder_sampling_ratio = 1.,
--   odometry_sampling_ratio = 1.,
--   fixed_frame_pose_sampling_ratio = 1.,
--   imu_sampling_ratio = 1.,
--   landmarks_sampling_ratio = 1.,
-- }

options = {
    map_builder = MAP_BUILDER,
    trajectory_builder = TRAJECTORY_BUILDER,
    map_frame = "map",
    tracking_frame = "base_footprint",
    published_frame = "odom",
    odom_frame = "odom",
    provide_odom_frame = false,
    publish_frame_projected_to_2d = false,
    -- Add a hidden parameter 'publish_tracked_pose' for 'evo' plotting
    publish_tracked_pose = true,
    use_pose_extrapolator = true,
    use_odometry = true,
    use_nav_sat = false,
    use_landmarks = false,
    num_laser_scans = 1,
    num_multi_echo_laser_scans = 0,
    num_subdivisions_per_laser_scan = 1,
    num_point_clouds = 0,
    lookup_transform_timeout_sec = 0.2,
    submap_publish_period_sec = 0.3,
    pose_publish_period_sec = 5e-3,
    trajectory_publish_period_sec = 30e-3,
    rangefinder_sampling_ratio = 1.,
    odometry_sampling_ratio = 1.,
    fixed_frame_pose_sampling_ratio = 1.,
    imu_sampling_ratio = 1.,
    landmarks_sampling_ratio = 1.,
  }

-- -------------------------------------------------------- --
-- Tips: 'TRAJECTORY_BUILDER_2D.use_imu_data' should always --
-- be included in the .lua, or the Cartographer node can not -
-- generate 'map' frame.                                    --
-- -------------------------------------------------------- --

-- -------------------------
-- Initial Configuration ---
-- -------------------------
-- This configuration shows best mapping performance when /odom is used
-- MAP_BUILDER.use_trajectory_builder_2d = true
-- TRAJECTORY_BUILDER_2D.use_imu_data = false
-- TRAJECTORY_BUILDER_2D.num_accumulated_range_data = 1


-- -------------------------
-- Improved Configuration --
-- -------------------------
-- This configuration can improve the pure LiDAR mapping performance when /odom is not used
-- MAP_BUILDER.use_trajectory_builder_2d = true
 
-- TRAJECTORY_BUILDER_2D.submaps.num_range_data = 35
-- TRAJECTORY_BUILDER_2D.min_range = 0.3
-- TRAJECTORY_BUILDER_2D.max_range = 8.
-- TRAJECTORY_BUILDER_2D.missing_data_ray_length = 1.
-- TRAJECTORY_BUILDER_2D.use_imu_data = false

-- -- Local SLAM
-- -- RealTimeCorrelativeScanMatcher 
-- TRAJECTORY_BUILDER_2D.use_online_correlative_scan_matching = true
-- TRAJECTORY_BUILDER_2D.real_time_correlative_scan_matcher.linear_search_window = 0.1
-- TRAJECTORY_BUILDER_2D.real_time_correlative_scan_matcher.translation_delta_cost_weight = 10.
-- TRAJECTORY_BUILDER_2D.real_time_correlative_scan_matcher.rotation_delta_cost_weight = 1e-1

-- -- Global SLAM
-- POSE_GRAPH.optimization_problem.huber_scale = 1e2
-- POSE_GRAPH.optimize_every_n_nodes = 35
-- POSE_GRAPH.constraint_builder.min_score = 0.65

-- ---------------------------------
-- Another Improved Configuration --
-- ---------------------------------
-- MAP_BUILDER.use_trajectory_builder_2d = true
-- TRAJECTORY_BUILDER_2D.use_imu_data = false
-- TRAJECTORY_BUILDER_2D.num_accumulated_range_data = 1

-- -- Local SLAM
-- TRAJECTORY_BUILDER_2D.submaps.num_range_data = 5

-- -- Global SLAM
-- POSE_GRAPH.optimize_every_n_nodes = 10


-- ---------------------------
-- Modifiable Configuration --
-- ---------------------------
MAP_BUILDER.use_trajectory_builder_2d = true

-- Bandpass filter to keep values in a certain range
TRAJECTORY_BUILDER_2D.min_range = 0.3
TRAJECTORY_BUILDER_2D.max_range = 8.
TRAJECTORY_BUILDER_2D.missing_data_ray_length = 1.

-- ROS msgs batch
TRAJECTORY_BUILDER_2D.num_accumulated_range_data = 1

-- Fixed-size voxel filter
-- TRAJECTORY_BUILDER_nD.voxel_filter_size

-- Adaptive voxel filter
-- TRAJECTORY_BUILDER_nD.*adaptive_voxel_filter.max_length
-- TRAJECTORY_BUILDER_nD.*adaptive_voxel_filter.min_num_points

-- IMU data
TRAJECTORY_BUILDER_2D.use_imu_data = false
-- TRAJECTORY_BUILDER_nD.imu_gravity_time_constant

-- ----------------
-- -- Local SLAM --
-- ----------------
-- CeresScanMatcher 
-- TRAJECTORY_BUILDER_2D.ceres_scan_matcher.occupied_space_weight
-- TRAJECTORY_BUILDER_3D.ceres_scan_matcher.occupied_space_weight_0
-- TRAJECTORY_BUILDER_3D.ceres_scan_matcher.occupied_space_weight_1
-- TRAJECTORY_BUILDER_nD.ceres_scan_matcher.translation_weight
-- TRAJECTORY_BUILDER_nD.ceres_scan_matcher.rotation_weight

-- TRAJECTORY_BUILDER_nD.ceres_scan_matcher.ceres_solver_options.use_nonmonotonic_steps
-- TRAJECTORY_BUILDER_nD.ceres_scan_matcher.ceres_solver_options.max_num_iterations
-- TRAJECTORY_BUILDER_nD.ceres_scan_matcher.ceres_solver_options.num_threads

-- RealTimeCorrelativeScanMatcher 
TRAJECTORY_BUILDER_2D.use_online_correlative_scan_matching = true
TRAJECTORY_BUILDER_2D.real_time_correlative_scan_matcher.linear_search_window = 0.1
TRAJECTORY_BUILDER_2D.real_time_correlative_scan_matcher.translation_delta_cost_weight = 10.
TRAJECTORY_BUILDER_2D.real_time_correlative_scan_matcher.rotation_delta_cost_weight = 1e-1

-- Motion filter
-- TRAJECTORY_BUILDER_nD.motion_filter.max_time_seconds
-- TRAJECTORY_BUILDER_nD.motion_filter.max_distance_meters
-- TRAJECTORY_BUILDER_nD.motion_filter.max_angle_radians

-- Submap
TRAJECTORY_BUILDER_2D.submaps.num_range_data = 5
-- TRAJECTORY_BUILDER_2D.submaps.grid_options_2d.grid_type -- Data structure

-- TRAJECTORY_BUILDER_2D.submaps.range_data_inserter.probability_grid_range_data_inserter.hit_probability
-- TRAJECTORY_BUILDER_2D.submaps.range_data_inserter.probability_grid_range_data_inserter.miss_probability
-- TRAJECTORY_BUILDER_3D.submaps.range_data_inserter.hit_probability
-- TRAJECTORY_BUILDER_3D.submaps.range_data_inserter.miss_probability

-- TRAJECTORY_BUILDER_2D.submaps.grid_options_2d.resolution
-- TRAJECTORY_BUILDER_3D.submaps.high_resolution
-- TRAJECTORY_BUILDER_3D.submaps.low_resolution
-- TRAJECTORY_BUILDER_3D.high_resolution_adaptive_voxel_filter.max_range
-- TRAJECTORY_BUILDER_3D.low_resolution_adaptive_voxel_filter.max_range

-- -----------------
-- -- Global SLAM --
-- -----------------
-- This item can significantly effect the mapping performance (FIX: NO DIFFERENCE)
-- Set to 0 to disable global SLAM
POSE_GRAPH.optimize_every_n_nodes = 10 -- 10, the mapping performance worse (FIX: NO DIFFERENCE)

-- Constraints of nodes & submaps
-- Global constraints (find loop closures)
-- POSE_GRAPH.constraint_builder.max_constraint_distance
-- POSE_GRAPH.fast_correlative_scan_matcher.linear_search_window
-- POSE_GRAPH.fast_correlative_scan_matcher_3d.linear_xy_search_window
-- POSE_GRAPH.fast_correlative_scan_matcher_3d.linear_z_search_window
-- POSE_GRAPH.fast_correlative_scan_matcher*.angular_search_window

-- Limit the amount of constraints
-- POSE_GRAPH.constraint_builder.sampling_ratio 

-- FastCorrelativeScanMatcher (real-time loop closures scan)
-- Exploration tree depth control
-- POSE_GRAPH.constraint_builder.fast_correlative_scan_matcher.branch_and_bound_depth
-- POSE_GRAPH.constraint_builder.fast_correlative_scan_matcher_3d.branch_and_bound_depth
-- POSE_GRAPH.constraint_builder.fast_correlative_scan_matcher_3d.full_resolution_depth

-- Refine pose via CeresScanMatcher 
POSE_GRAPH.constraint_builder.min_score = 0.65
-- POSE_GRAPH.constraint_builder.ceres_scan_matcher_3d
-- POSE_GRAPH.constraint_builder.ceres_scan_matcher

-- Weight & Ceres options configuration (Similar to Local SLAM)
-- POSE_GRAPH.constraint_builder.loop_closure_translation_weight
-- POSE_GRAPH.constraint_builder.loop_closure_rotation_weight
-- POSE_GRAPH.matcher_translation_weight
-- POSE_GRAPH.matcher_rotation_weight
-- POSE_GRAPH.optimization_problem.*_weight
-- POSE_GRAPH.optimization_problem.ceres_solver_options
-- POSE_GRAPH.log_residual_histograms = false

-- IMU residual
-- POSE_GRAPH.optimization_problem.log_solver_summary
-- POSE_GRAPH.optimization_problem.use_online_imu_extrinsics_in_3d

-- Huber loss function, defining the influence of outliers
POSE_GRAPH.optimization_problem.huber_scale = 1e2

-- Final result polishment (A large number of iterations is recommended)
-- POSE_GRAPH.max_num_final_iterations = 1e2

return options
