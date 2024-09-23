#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
 # @ Author: Kenneth Simon
 # @ Email: kuankuan_sima@u.nus.edu
 # @ Description:
 """

import argparse
import os
import sys
import copy

import cv2
import rosbag
import matplotlib.pyplot as plt
import numpy as np
import rospy
import yaml
from sensor_msgs.msg import Image
from std_msgs.msg import String
from vision_msgs.msg import Detection2D
from visualization_msgs.msg import Marker

sys.path.append(os.path.join(sys.path[0], ".."))
from utils import (
    calcu_center_dist,
    calcu_iou,
    draw_demo_img,
    draw_polygon,
    print_dict,
    restrain_bbox,
    calcu_temp_delta_hsv,
    get_obj_in_img_hsv,
)


class KalmanFilter:
    def __init__(self, x, P, A, Q, H, R):
        self.x = x  # state vector
        self.P = P  # state covariance matrix
        self.A = A  # state transition matrix
        self.Q = Q  # process noise covariance matrix
        self.H = H  # measurement matrix
        self.R = R  # measurement noise covariance matrix

    def predict(self):
        self.x = self.A @ self.x
        self.P = self.A @ self.P @ self.A.T + self.Q
        return self.x

    def update(self, z):
        y = z - self.H @ self.x  # measurement residual
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)  # kalman gain
        self.x = self.x + K @ y
        self.P = self.P - K @ self.H @ self.P


def load_data(seq_to_run=[1]):
    """
    track coordinate: [y,x,w,h]
    """
    data_path = os.path.join(sys.path[0], "../data/Task 1/")
    seqs = ["seq_" + str(i) for i in seq_to_run]
    data = {}
    img4all_seq = []
    ground_truth4all_seq = []
    fir_track4all_seq = []

    for seq in seqs:
        seq_path = os.path.join(data_path, seq)
        file_ptr = open(os.path.join(seq_path, "firsttrack.txt"), "r")
        fir_track = file_ptr.readline()[1:-2].split(",")
        fir_track = [int(x) for x in fir_track]
        file_ptr = open(os.path.join(seq_path, "groundtruth.txt"), "r")
        ground_truth = file_ptr.readlines()
        for i, gt in enumerate(ground_truth):
            gt = gt[1:-2].split(",")
            gt = [int(x) for x in gt]
            ground_truth[i] = gt
        img_path = os.path.join(seq_path, "img")
        img_files = os.listdir(img_path)
        img_files.sort()  # sort the files in ascending order, ensure the true sequential read
        imgs = [cv2.cvtColor(cv2.imread(os.path.join(img_path, img_file)), cv2.COLOR_BGR2RGB) for img_file in img_files]
        img4all_seq.append(imgs)
        ground_truth4all_seq.append(ground_truth)
        fir_track4all_seq.append(fir_track)
        data[seq] = {"img": imgs, "ground_truth": ground_truth, "fir_track": fir_track}

    # test and visualize
    if 0:
        fig, axs = plt.subplots(2, 1)
        axs[0].imshow(imgs[0])
        fir_track = {"x": fir_track[1], "y": fir_track[0], "w": fir_track[2], "h": fir_track[3]}
        draw_polygon(fir_track, axs[0])
        fir_track_img = imgs[0][
            fir_track["x"] : fir_track["x"] + fir_track["h"], fir_track["y"] : fir_track["y"] + fir_track["w"]
        ]
        axs[1].imshow(fir_track_img)
        plt.show()

    return data


class TemplateMatchROS:
    def __init__(self, param, seq=1):
        self.param = param
        data_all = load_data([seq])
        seq = "seq_" + str(seq)
        print(f">>> {seq} data loaded.")
        self.rosbag = rosbag.Bag(os.path.join(sys.path[0], f"{seq}.bag"), "w")

        self.gts = data_all[seq]["ground_truth"]
        self.fir_track = data_all[seq]["fir_track"]
        self.iou = []
        self.center_dist = []

        self.img_curr = data_all[seq]["img"][0]
        self.img_max_idx = len(data_all[seq]["img"]) - 1
        self.img_idx = 0

        # seq related params
        self.enable_hsv_finetune = param["task1"][seq]["enable_hsv_finetune"]
        self.enable_kalman = param["task1"][seq]["enable_kalman"]
        self.enable_multi_scale = param["task1"][seq]["enable_multi_scale"]
        if self.enable_kalman:
            if self.enable_hsv_finetune:
                self.temp_delta = {"dx": 0, "dy": 0, "dw": 0, "dh": 0}
                self.search_region_delta = {"dx": -50, "dy": -50, "dw": 50, "dh": 50}
            else:
                self.temp_delta = param["task1"][seq]["kalman"]["temp_delta"]
                self.search_region_delta = param["task1"][seq]["kalman"]["search_region_delta"]
        else:
            self.temp_delta = param["task1"][seq]["temp_delta"]
            self.search_region_delta = param["task1"][seq]["search_region_delta"]
        if self.enable_multi_scale:
            self.tp_adap_scale = (0.9, 0.95, 1, 1.05, 1.1)
        else:
            self.tp_adap_scale = [1]

        self.template = {
            "x": self.fir_track[1] + self.temp_delta["dx"],
            "y": self.fir_track[0] + self.temp_delta["dy"],
            "w": self.fir_track[2] + self.temp_delta["dw"],
            "h": self.fir_track[3] + self.temp_delta["dh"],
        }
        self.init_temp = copy.deepcopy(self.template)
        self.img_temp = self.img_curr[
            self.template["x"] : self.template["x"] + self.template["h"],
            self.template["y"] : self.template["y"] + self.template["w"],
        ]

        self.obj_hsv = param["task1"][seq]["hsv"]
        x, y, w, h = get_obj_in_img_hsv(self.img_temp, self.obj_hsv)
        self.obj_in_temp_ratio = {
            "w": w / self.template["w"],
            "h": h / self.template["h"],
        }  # the initial ratio of the object size to the template size
        print("initial template")
        self.temp_adj_max = param["task1"]["temp_size_adjust_ratio_limit"]
        self.temp_adj_coef = param["task1"]["temp_adjust_coef"]

        dt = 1 / 15  # according to rosbag hz
        Q = self.param["task1"][seq]["kalman"]["Q"]
        R = self.param["task1"][seq]["kalman"]["R"]
        vx = self.param["task1"][seq]["kalman"]["init"]["vx"]
        vy = self.param["task1"][seq]["kalman"]["init"]["vy"]
        self.kalman = KalmanFilter(
            x=np.array(np.array([self.template["x"], self.template["y"], vx, vy]), dtype=float),
            P=np.eye(4),
            A=np.array(
                [[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float
            ),  # x(k+1) = x(k) + v(k)*dt
            H=np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float),
            Q=np.array(Q),
            R=np.array(R),
        )
        self.kalman4size = KalmanFilter(
            x=np.array([self.template["w"], self.template["h"], 0, 0], dtype=float),
            P=np.eye(4),
            A=np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float),
            H=np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float),
            Q=np.diag([1, 1, 0.01, 0.01]),
            R=np.diag([100, 100]),
        )
        self.last_time_get_img = rospy.Time.now()

        print(f">>> {seq} param:")
        print_dict(self.param["task1"][seq]["kalman"])

        # ros related
        self.img_sub = rospy.Subscriber("/me5413/image_raw", Image, self.img_cb)
        self.obj_track_pub = rospy.Publisher("/task1/obj_tracked", Detection2D, queue_size=1)
        self.obj_gt_pub = rospy.Publisher("/task1/obj_gt", Detection2D, queue_size=1)
        self.track_img_pub = rospy.Publisher("/task1/track_img", Image, queue_size=1)
        self.text_marker_pub = rospy.Publisher("/task1/text_marker", Marker, queue_size=1)
        self.matric_pub = rospy.Publisher("/matric", String, queue_size=1)
        self.matric_pub.publish("A0284990M")
        rospy.loginfo("Task1 node initialized. Ready to subscribe images.")

    def img_cb(self, msg: Image):
        # convert Image to opencv format
        img = msg.data
        img = np.frombuffer(img, dtype=np.uint8).reshape(msg.height, msg.width, -1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.img_curr = img
        self.img_idx += 1
        self.img_idx = min(self.img_idx, self.img_max_idx)
        self.last_time_get_img = rospy.Time.now()
        self.tm()

    def tm(self):

        # gt_list = self.gts[self.img_idx]
        # gt = {"x": gt_list[1], "y": gt_list[0], "w": gt_list[2], "h": gt_list[3]}

        # search_region = {
        #     "x": self.template["x"] + self.search_region_delta["dx"],
        #     "y": self.template["y"] + self.search_region_delta["dy"],
        #     "h": self.template["h"] + self.search_region_delta["dh"]*2,
        #     "w": self.template["w"] + self.search_region_delta["dw"]*2,
        # }
        # restrain_bbox(search_region, self.img_curr.shape)  # restrain search region according to the image size

        # img_search = self.img_curr[
        #     search_region["x"] : search_region["x"] + search_region["h"],
        #     search_region["y"] : search_region["y"] + search_region["w"],
        # ]
        # match_result = cv2.matchTemplate(img_search, self.img_temp, match_method)
        # min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(match_result)  # find the best match
        # if match_method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        #     top_left = min_loc
        # else:
        #     top_left = max_loc
        # new_x = search_region["x"] + top_left[1]
        # new_y = search_region["y"] + top_left[0]

        # # update the template
        # x_obs = new_x
        # y_obs = new_y
        # z = np.array([x_obs, y_obs], dtype=float)
        # self.kalman.update(z)
        # x_pred = self.kalman.predict()
        # print(f"==>> x_pred: {[int(x) for x in x_pred]}")

        # self.template["x"] = new_x
        # self.template["y"] = new_y
        # self.template["x"] = int(x_pred[0])
        # self.template["y"] = int(x_pred[1])
        # self.img_temp = self.img_curr[
        #     self.template["x"] : self.template["x"] + self.template["h"],
        #     self.template["y"] : self.template["y"] + self.template["w"],
        # ]

        # self.iou.append(calcu_iou(self.template, gt))
        # self.center_dist.append(calcu_center_dist(self.template, gt))

        gt = {
            "x": self.gts[self.img_idx][1],
            "y": self.gts[self.img_idx][0],
            "w": self.gts[self.img_idx][2],
            "h": self.gts[self.img_idx][3],
        }

        search_region = {
            "x": self.template["x"] + self.search_region_delta["dx"],
            "y": self.template["y"] + self.search_region_delta["dy"],
            "w": self.template["w"] + self.search_region_delta["dw"] * 2,
            "h": self.template["h"] + self.search_region_delta["dh"] * 2,
        }
        restrain_bbox(search_region, self.img_curr.shape)  # restrain search region according to the image size

        img_search = self.img_curr[
            search_region["x"] : search_region["x"] + search_region["h"],
            search_region["y"] : search_region["y"] + search_region["w"],
        ]

        min_val_best = 1
        max_val_best = 0
        loc_best = None
        for i in range(len(self.tp_adap_scale)):  # try different scales
            width = int(self.img_temp.shape[1] * self.tp_adap_scale[i])
            height = int(self.img_temp.shape[0] * self.tp_adap_scale[i])
            if width > img_search.shape[1] or height > img_search.shape[0]:
                continue
            img_temp_scaled = cv2.resize(self.img_temp, (width, height))
            match_result = cv2.matchTemplate(img_search, img_temp_scaled, match_method)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(match_result)  # find the best match
            if match_method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
                if min_val < min_val_best:
                    min_val_best = min_val
                    loc_best = min_loc
            else:
                if max_val > max_val_best:
                    max_val_best = max_val
                    loc_best = max_loc

        new_x = search_region["x"] + loc_best[1]
        new_y = search_region["y"] + loc_best[0]
        new_w = self.template["w"]
        new_h = self.template["h"]

        temp_before = copy.deepcopy(self.template)

        # adjust template to center the object
        if self.enable_hsv_finetune:
            img_temp_tmp = self.img_curr[new_x : new_x + new_h, new_y : new_y + new_w]
            delta_center, delta_size = calcu_temp_delta_hsv(
                img_temp_tmp, self.obj_hsv, size_ratio=self.obj_in_temp_ratio
            )
            delta_center["x"] *= self.temp_adj_coef["pos"]
            delta_center["y"] *= self.temp_adj_coef["pos"]
            delta_size["w"] *= self.temp_adj_coef["size"]
            delta_size["h"] *= self.temp_adj_coef["size"]
            new_x += delta_center["x"]
            new_y += delta_center["y"]
            new_w += delta_size["w"]
            new_h += delta_size["h"]
            max_w = self.temp_adj_max["max"] * self.init_temp["w"]
            min_w = self.temp_adj_max["min"] * self.init_temp["w"]
            max_h = self.temp_adj_max["max"] * self.init_temp["h"]
            min_h = self.temp_adj_max["min"] * self.init_temp["h"]
            new_w = np.clip(new_w, min_w, max_w)
            new_h = np.clip(new_h, min_h, max_h)

        # update the template using kalman
        if self.enable_kalman:
            z = np.array([new_x, new_y], dtype=float)
            self.kalman.update(z)
            pos_pred = self.kalman.predict()
            new_x = int(pos_pred[0])
            new_y = int(pos_pred[1])
            if self.enable_hsv_finetune:
                z = np.array([new_w, new_h], dtype=float)
                self.kalman4size.update(z)
                size_pred = self.kalman4size.predict()
                new_w = int(size_pred[0])
                new_h = int(size_pred[1])

        self.template["x"] = int(new_x)
        self.template["y"] = int(new_y)
        self.template["w"] = int(new_w)
        self.template["h"] = int(new_h)
        restrain_bbox(self.template, self.img_curr.shape)

        self.img_temp = self.img_curr[
            self.template["x"] : self.template["x"] + self.template["h"],
            self.template["y"] : self.template["y"] + self.template["w"],
        ]

        self.iou.append(calcu_iou(self.template, gt))
        self.center_dist.append(calcu_center_dist(self.template, gt))

        # publish the result
        str_msg = String()
        str_msg.data = "A0284990M"
        self.rosbag.write("/task1/matric", str_msg)
        marker_msg = Marker()
        marker_msg.header.stamp = rospy.Time.now()
        marker_msg.header.frame_id = "map"
        marker_msg.type = Marker.TEXT_VIEW_FACING
        marker_msg.scale.z = 0.1
        marker_msg.pose.position.z = 1.0
        marker_msg.color.a = 1.0
        marker_msg.color.r = 1.0
        marker_msg.color.g = 1.0
        marker_msg.color.b = 1.0
        marker_msg.text = f"matric: A0284990M\n iou: {self.iou[-1]:.2f}\ncenter_dist: {self.center_dist[-1]:.2f}"
        self.text_marker_pub.publish(marker_msg)
        self.rosbag.write("/task1/marker", marker_msg)

        track_bb = Detection2D()
        track_bb.header.stamp = rospy.Time.now()
        track_bb.bbox.center.x = self.template["x"] + self.template["h"] / 2
        track_bb.bbox.center.y = self.template["y"] + self.template["w"] / 2
        track_bb.bbox.size_x = self.template["h"]
        track_bb.bbox.size_y = self.template["w"]
        gt_bb = Detection2D()
        gt_bb.header.stamp = rospy.Time.now()
        gt_bb.bbox.center.x = gt["x"] + gt["h"] / 2
        gt_bb.bbox.center.y = gt["y"] + gt["w"] / 2
        gt_bb.bbox.size_x = gt["h"]
        gt_bb.bbox.size_y = gt["w"]
        self.obj_track_pub.publish(track_bb)
        self.obj_gt_pub.publish(gt_bb)
        self.rosbag.write("/task1/obj_gt", gt_bb)
        self.rosbag.write("/task1/obj_tracked", track_bb)

        demo_img = self.img_curr.copy()
        demo_img = draw_demo_img(demo_img, self.template, gt, search_region=search_region)
        img_msg = Image()
        img_msg.header.stamp = rospy.Time.now()
        img_msg.data = demo_img.tobytes()
        img_msg.height = demo_img.shape[0]
        img_msg.width = demo_img.shape[1]
        img_msg.encoding = "bgr8"
        self.track_img_pub.publish(img_msg)
        self.rosbag.write("/task1/track_img", img_msg)

    def run(self):
        while not rospy.is_shutdown():
            if (rospy.Time.now() - self.last_time_get_img).to_sec() > 3:
                rospy.logwarn("No image received for 3 second. Node will exit.")
                self.rosbag.close()
                # break
            rospy.sleep(0.5)
        self.rosbag.close()


match_method = cv2.TM_CCOEFF_NORMED


if __name__ == "__main__":
    demo_output_type = "video"  # "plt" or "video"

    param = yaml.load(open(os.path.join(sys.path[0], "../config/param.yaml"), "r"), Loader=yaml.FullLoader)

    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--seq", type=int, required=True, help="sequence to run")
    args = parser.parse_args()

    rospy.init_node("bonus_task", anonymous=False)
    tm_ros = TemplateMatchROS(param, seq=args.seq)
    tm_ros.run()
