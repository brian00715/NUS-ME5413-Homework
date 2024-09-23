"""
 # @ Author: Kenneth Simon
 # @ Email: smkk00715@gmail.com
 # @ Create Time: 2024-03-14 02:11:11
 # @ Modified time: 2024-03-14 02:12:04
 # @ Description:
 """

import numpy as np

trajectory_file_path = "/home/simon/LocalDiskExt/Datasets/HW2_SLAM/Task2/A-LOAM/aloam.txt"
transformed_trajectory_file_path = "/home/simon/LocalDiskExt/Datasets/HW2_SLAM/Task2/A-LOAM/aloam_cam.txt"


transformation_matrix = np.array(
    [
        [0, 0, 1, 0.27],
        [-1, 0, 0, 0],
        [0, -1, 0, -0.08],
        [0, 0, 0, 1],
    ]
)


def transform_pose(pose: np.ndarray, transformation_matrix):
    pose_homogeneous = pose.reshape((3, 4))
    pose_homogeneous = np.vstack((pose_homogeneous, [0, 0, 0, 1]))

    transformed_pose = pose_homogeneous @ transformation_matrix
    return transformed_pose


def read_kitti_trajectory(file_path):
    with open(file_path, "r") as file:
        trajectory = []
        for line in file:
            pose = np.array(line.strip().split(" ")).astype(float)
            trajectory.append(pose)
        return trajectory


def transform_trajectory(trajectory, transformation_matrix):
    transformed_trajectory = [transform_pose(pose, transformation_matrix) for pose in trajectory]
    return transformed_trajectory


def save_transformed_trajectory(trajectory, file_path):
    with open(file_path, "w+") as file:
        for pose in trajectory:
            str_write = ""
            for i, data in enumerate(pose):
                str_write += str(data)
                if i != 11:
                    str_write += " "
            file.write(str_write + "\n")


if __name__ == "__main__":
    trajectory = read_kitti_trajectory(trajectory_file_path)
    transformed_trajectory = transform_trajectory(trajectory, transformation_matrix)

    pose_write_all = []
    for pose in transformed_trajectory:
        pose_write = pose.reshape((16,)).tolist()
        pose_write_all.append(pose_write[:12])
        for i in range(12):
            print(pose_write[i], end=" ")
        print()

    save_transformed_trajectory(pose_write_all, transformed_trajectory_file_path)
