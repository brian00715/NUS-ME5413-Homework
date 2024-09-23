#!/usr/bin/env python3

import rospy
from nav_msgs.msg import Odometry
from nav_msgs.msg import Path
from geometry_msgs.msg import Twist, Pose
from nmpc_controller import NMPCC
import numpy as np
import  tf2_geometry_msgs
import tf2_ros
import math
import time

class PathTrackerNode1:
    def __init__(self):
        rospy.init_node('path_tracker_node')

        self.flag = True

        prob_params = {
            "control_dim": 2,
            "state_dim": 3,
            "max_thrust": 10,
            "init_pose": np.array([0, 0, 0]),
        }

        self.N = 20

        self.opti = NMPCC(N=self.N, prob_params=prob_params)

        # 订阅机器人里程计信息
        self.sub_robot_odom = rospy.Subscriber("/gazebo/ground_truth/state", Odometry, self.robot_odom_callback)

        # 发布速度控制指令
        self.pub_cmd_vel = rospy.Publisher("/jackal_velocity_controller/cmd_vel", Twist, queue_size=1)
        
        # 订阅局部路径规划信息
        self.sub_local_path = rospy.Subscriber("/me5413_world/planning/local_path", Path, self.local_path_callback)






    def pos_to_state(self, pos):
        # 将位置信息转换为状态信息
        yaw = self.quat_to_yaw(pos.orientation)
        state = np.array([pos.position.x, pos.position.y, yaw])
        return state

    def quat_to_yaw(self, quat):
        # 将四元数转换为yaw角
        yaw = np.arctan2(2*(quat.w*quat.z + quat.x*quat.y), 1-2*(quat.y**2 + quat.z**2))
        return yaw



    def robot_odom_callback(self, odom):
        # 处理机器人里程计信息
        if self.flag:
            self.flag = False
            pose = odom.pose.pose
            self.init_pose_ = self.pos_to_state(pose)
            return
        self.world_frame_ = odom.header.frame_id
        self.robot_frame_ = odom.child_frame_id
        self.odom_world_robot_ = odom
        
    def local_path_callback(self, path):
        # 处理局部路径规划信息
        # self.pose_world_goal_ = path.pose[11].pose
        # self.pose_world_goal_ 应该包含后续一段时间内的路径规划信息
        # goal_poses = [pose.pose.position for pose in path.poses]
        goal_poses = [[path.poses[i].pose.position.x, path.poses[i].pose.position.y, \
                      self.quat_to_yaw(path.poses[i].pose.orientation)] for i in range(11, min(len(path.poses), 11+self.N+ 1))]
        if len(goal_poses) < self.N+1:
            goal_poses += [goal_poses[-1]]*(self.N+1-len(goal_poses))
        for i in range(len(goal_poses)):
            if goal_poses[i][2] < -math.pi:
                goal_poses[i][2] += math.pi
            if goal_poses[i][2] > math.pi:
                goal_poses[i][2] -= math.pi
        cur_pose = [self.odom_world_robot_.pose.pose.position.x, self.odom_world_robot_.pose.pose.position.y, \
                    self.quat_to_yaw(self.odom_world_robot_.pose.pose.orientation)]
        self.pub_cmd_vel.publish(self.computeControlOutputs(cur_pose, goal_poses))
        

    def run(self):
        # 主循环
        rate = rospy.Rate(10)  # 10Hz
        while not rospy.is_shutdown():
            rate.sleep()


    def computeControlOutputs(self, odom_robot, pose_goal):
        u = self.opti.solve(odom_robot, pose_goal, [[0.5, 0]]*self.N)
        # # 计算控制输出
        # q_robot = tf2_ros.Quaternion.from_msg(odom_robot.pose.pose.orientation)
        # q_goal = tf2_ros.Quaternion.from_msg(pose_goal.orientation)
        # roll, pitch, yaw_robot = tf2_ros.transformations.euler_from_quaternion(q_robot)
        # roll, pitch, yaw_goal = tf2_ros.transformations.euler_from_quaternion(q_goal)

        # heading_error = self.unify_angle_range(yaw_robot - yaw_goal)

        # # Lateral Error
        # point_robot = tf2_ros.Vector3.from_msg(odom_robot.pose.pose.position)
        # point_goal = tf2_ros.Vector3.from_msg(pose_goal.position)
        # V_goal_robot = point_robot - point_goal
        # angle_goal_robot = math.atan2(V_goal_robot.y, V_goal_robot.x)
        # angle_diff = angle_goal_robot - yaw_goal
        # lat_error = V_goal_robot.length() * math.sin(angle_diff)

        # # Velocity
        # robot_vel = tf2_ros.Vector3.from_msg(odom_robot.twist.twist.linear)
        # velocity = robot_vel.length()

        cmd_vel = Twist()

        # res = self.opti.solve(x_ref, u_ref)

        # cmd_vel.linear.x = self.pid.calculate(SPEED_TARGET, velocity)
        # cmd_vel.angular.z = self.compute_stanely_control(heading_error, lat_error, velocity)

        cmd_vel.linear.x = u[0][0]
        cmd_vel.angular.z = u[0][1]

        return cmd_vel
        

if __name__ == '__main__':
    try:
        path_tracker_node = PathTrackerNode1()
        path_tracker_node.run()
    except rospy.ROSInterruptException:
        pass
