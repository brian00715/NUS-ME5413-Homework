#!/bin/bash

gnome-terminal  -- bash -c "roscore"
sleep 1
# gnome-terminal  -- bash -c "rqt_graph"
gnome-terminal  -- bash -c "python3 bonus_task.py --seq $1"
gnome-terminal  -- bash -c "rviz -d bonus_task.rviz"
cd ../data/Bonus\ Task
sleep 2
gnome-terminal  -- bash -c "rosbag play seq_$1.bag"
