"""
 # @ Author: Kenneth Simon
 # @ Email: kuankuan_sima@u.nus.edu
 # @ Description:
 """

import argparse
import copy
import json
import os
import sys
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import yaml

sys.path.append(os.path.join(sys.path[0], "../"))
from utils import (
    calcu_center_dist,
    calcu_iou,
    calcu_temp_delta_hsv,
    draw_demo_img,
    draw_in_plt,
    draw_polygon,
    get_obj_in_img_hsv,
    load_data,
    restrain_bbox,
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


def tm(data_all, param, demo_output_type="video", write_log=False, realtime=True):
    for seq in data_all.keys():  # traverse all the sequences
        data = data_all[seq]
        imgs, gts, fir_track = data["img"], data["ground_truth"], data["fir_track"]

        if demo_output_type == "video":
            file_name = seq + ".avi"
            file_path = os.path.join(sys.path[0], "../temp/", file_name)
            video_out = cv2.VideoWriter(
                file_path, cv2.VideoWriter_fourcc(*"MJPG"), 30, (imgs[0].shape[1], imgs[0].shape[0])
            )

        temp_delta = param["task1"][seq]["temp_delta"]
        search_region_delta = param["task1"][seq]["search_region_delta"]

        # template = {"x": fir_track[1], "y": fir_track[0], "w": fir_track[2], "h": fir_track[3]}
        template = {
            "x": fir_track[1] + temp_delta["dx"],
            "y": fir_track[0] + temp_delta["dy"],
            "w": fir_track[2] + temp_delta["dw"],
            "h": fir_track[3] + temp_delta["dh"],
        }

        error = []
        img_temp = imgs[0][template["x"] : template["x"] + template["h"], template["y"] : template["y"] + template["w"]]

        for j, img in enumerate(imgs):  # traverse all the images in the sequence
            gt_list = gts[j]
            gt = {"x": gt_list[1], "y": gt_list[0], "w": gt_list[2], "h": gt_list[3]}

            search_region = {
                "x": template["x"] + search_region_delta["dx"],
                "y": template["y"] + search_region_delta["dy"],
                "w": template["w"] + search_region_delta["dw"],
                "h": template["h"] + search_region_delta["dh"],
            }
            restrain_bbox(search_region, img.shape)  # restrain search region according to the image size

            img_search = img[
                search_region["x"] : search_region["x"] + search_region["h"],
                search_region["y"] : search_region["y"] + search_region["w"],
            ]
            match_result = cv2.matchTemplate(img_search, img_temp, match_method)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(match_result)  # find the best match
            if match_method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
                top_left = min_loc
            else:
                top_left = max_loc
            new_x = search_region["x"] + top_left[1]
            new_y = search_region["y"] + top_left[0]

            # visualize the result
            if demo_output_type == "plt":
                draw_in_plt(img, template, gt, search_region=search_region, match_result=match_result)
            if demo_output_type == "video":
                demo_img = img.copy()
                demo_img = draw_demo_img(demo_img, template, gt, search_region=search_region)
                cv2.imshow("demo", demo_img)
                if realtime:
                    cv2.waitKey(1)
                else:
                    cv2.waitKey(0)
                if write_log:
                    video_out.write(demo_img)

            # update the template
            template["x"] = new_x
            template["y"] = new_y
            img_temp = img[template["x"] : template["x"] + template["h"], template["y"] : template["y"] + template["w"]]

        if demo_output_type == "video":
            video_out.release()


def tm_kalman(data_all, param, demo_output_type="video", write_log=False, realtime=True):
    dt = 1 / 30
    # init the kalman filter, x = [x,y,vx,vy]

    kalman = KalmanFilter(
        x=np.array([0, 0, 0, 0], dtype=float),
        P=np.eye(4),
        A=np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float),  # x(k+1) = x(k) + v(k)*dt
        H=np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float),
        # Q is greater than R means the model is more trustful than the measurement
        Q=np.eye(4),
        R=np.eye(2) * 1,
    )

    for seq in data_all.keys():  # traverse all the sequences
        data = data_all[seq]
        imgs, gts, fir_track = data["img"], data["ground_truth"], data["fir_track"]

        if demo_output_type == "video":
            file_name = seq + ".avi"
            file_path = os.path.join(sys.path[0], "./results/kalman", file_name)
            video_out = cv2.VideoWriter(
                file_path, cv2.VideoWriter_fourcc(*"MJPG"), 30, (imgs[0].shape[1], imgs[0].shape[0])
            )

        temp_delta = param["task1"][seq]["kalman"]["temp_delta"]
        search_region_delta = param["task1"][seq]["kalman"]["search_region_delta"]

        # template = {"x": fir_track[1], "y": fir_track[0], "w": fir_track[2], "h": fir_track[3]}
        template = {
            "x": fir_track[1] + temp_delta["dx"],
            "y": fir_track[0] + temp_delta["dy"],
            "w": fir_track[2] + temp_delta["dw"],
            "h": fir_track[3] + temp_delta["dh"],
        }
        img_temp = imgs[0][template["x"] : template["x"] + template["h"], template["y"] : template["y"] + template["w"]]

        iou = []
        center_dist = []
        Q = param["task1"][seq]["kalman"]["Q"]
        R = param["task1"][seq]["kalman"]["R"]
        vx = param["task1"][seq]["kalman"]["init"]["vx"]
        vy = param["task1"][seq]["kalman"]["init"]["vy"]
        kalman.Q = np.array(Q)
        kalman.R = np.array(R)
        kalman.x = np.array([template["x"], template["y"], vx, vy], dtype=float)

        for j, img in enumerate(imgs):  # traverse all the images in the sequence
            gt = {"x": gts[j][1], "y": gts[j][0], "w": gts[j][2], "h": gts[j][3]}

            search_region = {
                "x": template["x"] + search_region_delta["dx"],
                "y": template["y"] + search_region_delta["dy"],
                "w": template["w"] + search_region_delta["dw"],
                "h": template["h"] + search_region_delta["dh"],
            }
            restrain_bbox(search_region, img.shape)  # restrain search region according to the image size

            img_search = img[
                search_region["x"] : search_region["x"] + search_region["h"],
                search_region["y"] : search_region["y"] + search_region["w"],
            ]
            match_result = cv2.matchTemplate(img_search, img_temp, match_method)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(match_result)  # find the best match
            if match_method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
                top_left = min_loc
            else:
                top_left = max_loc
            new_x = search_region["x"] + top_left[1]
            new_y = search_region["y"] + top_left[0]

            # update the template
            x_obs = new_x
            y_obs = new_y
            z = np.array([x_obs, y_obs], dtype=float)
            kalman.update(z)
            x_pred = kalman.predict()
            print(f"==>> x_pred: {[int(x) for x in x_pred]}")

            template["x"] = new_x
            template["y"] = new_y
            template["x"] = int(x_pred[0])
            template["y"] = int(x_pred[1])
            img_temp = img[template["x"] : template["x"] + template["h"], template["y"] : template["y"] + template["w"]]

            iou.append(calcu_iou(template, gt))
            center_dist.append(calcu_center_dist(template, gt))

            # visualize the result
            if demo_output_type == "plt":
                draw_in_plt(img, template, gt, search_region=search_region, match_result=match_result)
            if demo_output_type == "video":
                demo_img = img.copy()
                demo_img = draw_demo_img(
                    demo_img, template, gt, search_region=search_region, iou=iou[-1], center_dist=center_dist[-1]
                )
                video_out.write(demo_img)
                cv2.imshow("demo", demo_img)
                if realtime:
                    cv2.waitKey(1)
                else:
                    cv2.waitKey(0)

        # result post-processing
        if demo_output_type == "video":
            video_out.release()
        iou = np.array(iou)
        center_dist = np.array(center_dist)
        iou_sum = np.sum(iou)
        center_dist_sum = np.sum(center_dist)
        iou_mean = np.mean(iou)
        center_dist_mean = np.mean(center_dist)
        iou_std = np.std(iou)
        center_dist_std = np.std(center_dist)
        print(f"iou_sum: {iou_sum}, center_dist_sum: {center_dist_sum}")
        print(f"iou_mean: {iou_mean}, center_dist_mean: {center_dist_mean}")
        print(f"iou_std: {iou_std}, center_dist_std: {center_dist_std}")
        _, ax = plt.subplots(1, 2)
        ax[0].plot(iou, label="iou", color="r")
        ax[0].legend()
        ax[1].plot(center_dist, label="center_dist", color="b")
        ax[1].legend()
        plt.show()
        if write_log:
            result_json = {
                "iou_sum": iou_sum,
                "center_dist_sum": center_dist_sum,
                "iou_mean": iou_mean,
                "center_dist_mean": center_dist_mean,
                "iou_std": iou_std,
                "center_dist_std": center_dist_std,
                "param": param["task1"][seq]["kalman"],
            }
            time_stamp = time.strftime("%m-%d-%H-%M-%S", time.localtime())
            open(os.path.join(sys.path[0], "../temp/task1/kalman", seq + f"_{time_stamp}.json"), "w").write(
                json.dumps(result_json, indent=2)
            )


def tm_adap(data_all, param, demo_output_type="video", write_log=False, realtime=True):
    dt = 1 / 30
    kalman = KalmanFilter(
        x=np.array([0, 0, 0, 0], dtype=float),
        P=np.eye(4),
        A=np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float),  # x(k+1) = x(k) + v(k)*dt
        H=np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float),
        # Q is greater than R means the model is more trustful than the measurement
        Q=np.eye(4),
        R=np.eye(2) * 1,
    )
    kalman4adap = KalmanFilter(
        x=np.array([0, 0, 0, 0], dtype=float),
        P=np.eye(4),
        A=np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float),  # x(k+1) = x(k) + v(k)*dt
        H=np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float),
        # Q is greater than R means the model is more trustful than the measurement
        Q=np.eye(4),
        R=np.eye(2) * 1,
    )  # predict the template size change

    for seq in data_all.keys():  # traverse all the sequences
        adap_delta = param["task1"][seq]["adap_delta"]
        data = data_all[seq]
        imgs, gts, fir_track = data["img"], data["ground_truth"], data["fir_track"]

        if demo_output_type == "video":
            file_name = seq + ".avi"
            file_path = os.path.join(sys.path[0], "../temp/", file_name)
            video_out = cv2.VideoWriter(
                file_path, cv2.VideoWriter_fourcc(*"MJPG"), 30, (imgs[0].shape[1], imgs[0].shape[0])
            )

        temp_delta = param["task1"][seq]["kalman"]["temp_delta"]
        search_region_delta = param["task1"][seq]["kalman"]["search_region_delta"]

        # template = {"x": fir_track[1], "y": fir_track[0], "w": fir_track[2], "h": fir_track[3]}
        template = {
            "x": fir_track[1] + temp_delta["dx"],
            "y": fir_track[0] + temp_delta["dy"],
            "w": fir_track[2] + temp_delta["dw"],
            "h": fir_track[3] + temp_delta["dh"],
        }
        tp_adap_size_delta = [
            [0, 0],
            # [0, adap_delta["dw"]],
            # [adap_delta["dh"], 0],
            # [0, -adap_delta["dw"]],
            # [-adap_delta["dh"], 0],
            # [adap_delta["dh"], adap_delta["dw"]],
            # [-adap_delta["dh"], -adap_delta["dw"]],
            # [adap_delta["dh"], -adap_delta["dw"]],
            # [-adap_delta["dh"], adap_delta["dw"]],
        ]
        tp_adap_scale = [0.5, 0.75, 1, 1.25, 1.5]
        iou = []
        center_dist = []
        Q = param["task1"][seq]["kalman"]["Q"]
        R = param["task1"][seq]["kalman"]["R"]
        vx = param["task1"][seq]["kalman"]["init"]["vx"]
        vy = param["task1"][seq]["kalman"]["init"]["vy"]
        kalman.Q = np.array(Q)
        kalman.R = np.array(R)
        kalman.x = np.array([template["x"], template["y"], vx, vy], dtype=float)
        kalman4adap.Q = np.diag([1, 1, 0.1, 0.1])
        kalman4adap.R = np.diag([10, 10])
        kalman4adap.x = np.array([template["w"], template["h"], 0, 0], dtype=float)

        # img_temp = imgs[0][template["x"] : template["x"] + template["h"], template["y"] : template["y"] + template["w"]]
        img_temp = []
        for i in range(len(tp_adap_size_delta)):
            x2 = template["x"] + template["h"] + tp_adap_size_delta[i][0]
            x2 = np.clip(x2, 0, imgs[0].shape[0])
            y2 = template["y"] + template["w"] + tp_adap_size_delta[i][1]
            y2 = np.clip(y2, 0, imgs[0].shape[1])
            img_temp.append(imgs[0][template["x"] : x2, template["y"] : y2])

        for j, img in enumerate(imgs):  # traverse all the images in the sequence
            gt_list = gts[j]
            gt = {"x": gt_list[1], "y": gt_list[0], "w": gt_list[2], "h": gt_list[3]}

            search_region = {
                "x": template["x"] + search_region_delta["dx"],
                "y": template["y"] + search_region_delta["dy"],
                "w": template["w"] + search_region_delta["dw"],
                "h": template["h"] + search_region_delta["dh"],
            }
            restrain_bbox(search_region, img.shape)  # restrain search region according to the image size
            img_search = img[
                search_region["x"] : search_region["x"] + search_region["h"],
                search_region["y"] : search_region["y"] + search_region["w"],
            ]

            min_val_best = 1
            max_val_best = 0
            loc_best = None
            best_adap_idx = 0
            for i in range(len(tp_adap_size_delta)):  # try all the templates
                curr_img_temp = img_temp[i]
                for j in range(len(tp_adap_scale)):  # try different scales
                    width = int(curr_img_temp.shape[1] * tp_adap_scale[j])
                    height = int(curr_img_temp.shape[0] * tp_adap_scale[j])
                    if width >= img_search.shape[1] or height >= img_search.shape[0]:
                        continue
                    img_temp_scaled = cv2.resize(curr_img_temp, (width, height))
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
                            best_adap_idx = i
            print(f"max_val_best: {max_val_best}")
            new_x = search_region["x"] + loc_best[1]
            new_y = search_region["y"] + loc_best[0]

            iou.append(calcu_iou(template, gt))
            center_dist.append(calcu_center_dist(template, gt))

            # visualize the result
            if demo_output_type == "plt":
                draw_in_plt(img, template, gt, search_region=search_region, match_result=match_result)
            if demo_output_type == "video":
                demo_img = img.copy()
                demo_img = draw_demo_img(demo_img, template, gt, search_region=search_region)
                cv2.imshow("demo", demo_img)
                if realtime:
                    cv2.waitKey(1)
                else:
                    cv2.waitKey(0)
                if write_log:
                    video_out.write(demo_img)

            # update the template
            x_obs = new_x
            y_obs = new_y
            z = np.array([x_obs, y_obs], dtype=float)
            kalman.update(z)
            x_pred = kalman.predict()
            template["x"] = int(x_pred[0])
            template["y"] = int(x_pred[1])

            w_obs = template["w"] + tp_adap_size_delta[best_adap_idx][0]
            h_obs = template["h"] + tp_adap_size_delta[best_adap_idx][1]
            if max_val_best > 0.98:
                w_obs += 2
                h_obs += 2
            elif max_val_best < 0.92:
                w_obs -= 2
                h_obs -= 2
            template["w"] = w_obs
            template["h"] = h_obs

            # z = np.array([w_obs, h_obs], dtype=float)
            # kalman4adap.update(z)
            # x_pred = kalman4adap.predict()
            # template["w"] = int(x_pred[0])
            # template["h"] = int(x_pred[1])

            img_temp = []
            # img_temp = img[template["x"] : template["x"] + template["h"], template["y"] : template["y"] + template["w"]]
            for i in range(len(tp_adap_size_delta)):
                x2 = template["x"] + template["h"] + tp_adap_size_delta[i][0]
                x2 = np.clip(x2, 0, img.shape[0] - 1)
                y2 = template["y"] + template["w"] + tp_adap_size_delta[i][1]
                y2 = np.clip(y2, 0, img.shape[1] - 1)
                img_temp.append(
                    img[
                        template["x"] : x2,
                        template["y"] : y2,
                    ]
                )

        if demo_output_type == "video":
            video_out.release()
        # result post-processing
        iou = np.array(iou)
        center_dist = np.array(center_dist)
        iou_sum = np.sum(iou)
        center_dist_sum = np.sum(center_dist)
        iou_mean = np.mean(iou)
        center_dist_mean = np.mean(center_dist)
        iou_std = np.std(iou)
        center_dist_std = np.std(center_dist)
        print(f"iou_sum: {iou_sum}, center_dist_sum: {center_dist_sum}")
        print(f"iou_mean: {iou_mean}, center_dist_mean: {center_dist_mean}")
        print(f"iou_std: {iou_std}, center_dist_std: {center_dist_std}")

        # _, ax = plt.subplots(1, 2)
        # ax[0].plot(iou, label="iou", color="r")
        # ax[0].legend()
        # ax[1].plot(center_dist, label="center_dist", color="b")
        # ax[1].legend()
        # plt.show()

        if write_log:
            result_json = {
                "iou_sum": iou_sum,
                "center_dist_sum": center_dist_sum,
                "iou_mean": iou_mean,
                "center_dist_mean": center_dist_mean,
                "iou_std": iou_std,
                "center_dist_std": center_dist_std,
                "param": param["task1"][seq]["kalman"],
            }
            time_stamp = time.strftime("%m-%d-%H-%M-%S", time.localtime())
            open(os.path.join(sys.path[0], "results/adap_temp", seq + f"_{time_stamp}.json"), "w").write(
                json.dumps(result_json, indent=2)
            )


def tm_final(data_all, param, demo_output_type="video", write_log=False, realtime=True, log_prefix=""):
    dt = 1 / 30
    temp_adj_max = param["task1"]["temp_size_adjust_ratio_limit"]
    temp_adj_coef = param["task1"]["temp_adjust_coef"]

    kalman = KalmanFilter(
        x=np.array([0, 0, 0, 0], dtype=float),
        P=np.eye(4),
        A=np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float),  # x(k+1) = x(k) + v(k)*dt
        H=np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float),
        Q=np.eye(4),
        R=np.eye(2) * 1,
    )
    kalman4size = KalmanFilter(
        x=np.array([0, 0, 0, 0], dtype=float),
        P=np.eye(4),
        A=np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float),
        H=np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float),
        Q=np.eye(4),
        R=np.eye(2) * 1,
    )

    for seq in data_all.keys():  # traverse all the sequences
        data = data_all[seq]
        imgs, gts, fir_track = data["img"], data["ground_truth"], data["fir_track"]

        if demo_output_type == "video":
            if not os.path.exists(os.path.join(sys.path[0], f"./results/{log_prefix}/")):
                os.makedirs(os.path.join(sys.path[0], f"./results/{log_prefix}/"))
            file_name = seq + ".avi"
            file_path = os.path.join(sys.path[0], f"./results/{log_prefix}/", file_name)
            video_out = cv2.VideoWriter(
                file_path, cv2.VideoWriter_fourcc(*"MJPG"), 30, (imgs[0].shape[1], imgs[0].shape[0])
            )

        # load seq related params
        enable_hsv_finetune = param["task1"][seq]["enable_hsv_finetune"]
        enable_kalman = param["task1"][seq]["enable_kalman"]
        enable_multi_scale = param["task1"][seq]["enable_multi_scale"]
        if enable_kalman:
            if enable_hsv_finetune:
                temp_delta = {"dx": 0, "dy": 0, "dw": 0, "dh": 0}
                search_region_delta = {"dx": -50, "dy": -50, "dw": 50, "dh": 50}
            else:
                temp_delta = param["task1"][seq]["kalman"]["temp_delta"]
                search_region_delta = param["task1"][seq]["kalman"]["search_region_delta"]
        else:
            temp_delta = param["task1"][seq]["temp_delta"]
            search_region_delta = param["task1"][seq]["search_region_delta"]
        if enable_multi_scale:
            tp_adap_scale = (0.9, 0.95, 1, 1.05, 1.1)
        else:
            tp_adap_scale = [1]
        obj_hsv = param["task1"][seq]["hsv"]

        template = {
            "x": fir_track[1] + temp_delta["dx"],
            "y": fir_track[0] + temp_delta["dy"],
            "w": fir_track[2] + temp_delta["dw"],
            "h": fir_track[3] + temp_delta["dh"],
        }
        init_temp = copy.deepcopy(template)
        img_temp = imgs[0][template["x"] : template["x"] + template["h"], template["y"] : template["y"] + template["w"]]
        x, y, w, h = get_obj_in_img_hsv(img_temp, obj_hsv)
        obj_in_temp_ratio = {
            "w": w / template["w"],
            "h": h / template["h"],
        }  # the initial ratio of the object size to the template size
        print(f"obj_in_temp_ratio: {obj_in_temp_ratio}")

        Q = param["task1"][seq]["kalman"]["Q"]
        R = param["task1"][seq]["kalman"]["R"]
        vx = param["task1"][seq]["kalman"]["init"]["vx"]
        vy = param["task1"][seq]["kalman"]["init"]["vy"]
        kalman.Q = np.array(Q)
        kalman.R = np.array(R)
        kalman.x = np.array([template["x"], template["y"], vx, vy], dtype=float)
        kalman4size.Q = np.diag([1, 1, 0.01, 0.01])
        kalman4size.R = np.diag([100, 100])
        kalman4size.x = np.array([template["w"], template["h"], 0, 0], dtype=float)

        iou = []
        center_dist = []
        for j, img in enumerate(imgs):  # traverse all the images in the sequence
            gt = {"x": gts[j][1], "y": gts[j][0], "w": gts[j][2], "h": gts[j][3]}

            search_region = {
                "x": template["x"] + search_region_delta["dx"],
                "y": template["y"] + search_region_delta["dy"],
                "w": template["w"] + search_region_delta["dw"] * 2,
                "h": template["h"] + search_region_delta["dh"] * 2,
            }
            restrain_bbox(search_region, img.shape)  # restrain search region according to the image size

            img_search = img[
                search_region["x"] : search_region["x"] + search_region["h"],
                search_region["y"] : search_region["y"] + search_region["w"],
            ]

            min_val_best = 1
            max_val_best = 0
            loc_best = None
            for i in range(len(tp_adap_scale)):  # try different scales
                width = int(img_temp.shape[1] * tp_adap_scale[i])
                height = int(img_temp.shape[0] * tp_adap_scale[i])
                if width > img_search.shape[1] or height > img_search.shape[0]:
                    continue
                img_temp_scaled = cv2.resize(img_temp, (width, height))
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
            new_w = template["w"]
            new_h = template["h"]

            temp_before = copy.deepcopy(template)

            # adjust template to center the object
            if enable_hsv_finetune:
                img_temp_tmp = img[new_x : new_x + new_h, new_y : new_y + new_w]
                delta_center, delta_size = calcu_temp_delta_hsv(img_temp_tmp, obj_hsv, size_ratio=obj_in_temp_ratio)
                delta_center["x"] *= temp_adj_coef["pos"]
                delta_center["y"] *= temp_adj_coef["pos"]
                delta_size["w"] *= temp_adj_coef["size"]
                delta_size["h"] *= temp_adj_coef["size"]
                new_x += delta_center["x"]
                new_y += delta_center["y"]
                new_w += delta_size["w"]
                new_h += delta_size["h"]
                max_w = temp_adj_max["max"] * init_temp["w"]
                min_w = temp_adj_max["min"] * init_temp["w"]
                max_h = temp_adj_max["max"] * init_temp["h"]
                min_h = temp_adj_max["min"] * init_temp["h"]
                new_w = np.clip(new_w, min_w, max_w)
                new_h = np.clip(new_h, min_h, max_h)

            # update the template using kalman
            if enable_kalman:
                z = np.array([new_x, new_y], dtype=float)
                kalman.update(z)
                pos_pred = kalman.predict()
                new_x = int(pos_pred[0])
                new_y = int(pos_pred[1])
                if enable_hsv_finetune:
                    z = np.array([new_w, new_h], dtype=float)
                    kalman4size.update(z)
                    # size_pred = [new_w, new_h]
                    size_pred = kalman4size.predict()
                    new_w = int(size_pred[0])
                    new_h = int(size_pred[1])

            template["x"] = int(new_x)
            template["y"] = int(new_y)
            template["w"] = int(new_w)
            template["h"] = int(new_h)
            restrain_bbox(template, img.shape)

            img_temp = img[template["x"] : template["x"] + template["h"], template["y"] : template["y"] + template["w"]]

            iou.append(calcu_iou(template, gt))
            center_dist.append(calcu_center_dist(template, gt))

            # visualize the result
            if demo_output_type == "plt":
                draw_in_plt(img, temp_before, gt, search_region=search_region, match_result=match_result)
            if demo_output_type == "video":
                demo_img = img.copy()
                demo_img = draw_demo_img(
                    demo_img, temp_before, gt, search_region=search_region, iou=iou[-1], center_dist=center_dist[-1]
                )
                cv2.imshow("demo", demo_img)
                video_out.write(demo_img)
                if realtime:
                    cv2.waitKey(1)
                else:
                    cv2.waitKey(0)

        # result post-processing
        if demo_output_type == "video":
            video_out.release()
        iou = np.array(iou)
        center_dist = np.array(center_dist)
        iou_sum = np.sum(iou)
        center_dist_sum = np.sum(center_dist)
        iou_mean = np.mean(iou)
        center_dist_mean = np.mean(center_dist)
        iou_std = np.std(iou)
        center_dist_std = np.std(center_dist)
        print(f"iou_sum: {iou_sum}, center_dist_sum: {center_dist_sum}")
        print(f"iou_mean: {iou_mean}, center_dist_mean: {center_dist_mean}")
        print(f"iou_std: {iou_std}, center_dist_std: {center_dist_std}")

        # _, ax = plt.subplots(1, 2)
        # ax[0].plot(iou, label="iou", color="r")
        # ax[0].legend()
        # ax[1].plot(center_dist, label="center_dist", color="b")
        # ax[1].legend()
        # plt.show()

        if write_log:
            result_json = {
                "iou_sum": iou_sum,
                "center_dist_sum": center_dist_sum,
                "iou_mean": iou_mean,
                "center_dist_mean": center_dist_mean,
                "iou_std": iou_std,
                "center_dist_std": center_dist_std,
                "param": param["task1"][seq],
            }
            time_stamp = time.strftime("%m-%d-%H-%M-%S", time.localtime())
            open(os.path.join(sys.path[0], f"results/{log_prefix}", seq + f"_{time_stamp}.json"), "w").write(
                json.dumps(result_json, indent=2)
            )


match_method = cv2.TM_CCOEFF_NORMED


if __name__ == "__main__":
    demo_output_type = "video"  # "plt" or "video"

    param = yaml.load(open(os.path.join(sys.path[0], "../config/param.yaml"), "r"), Loader=yaml.FullLoader)

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--seq", type=int, default=[1], nargs="+", help="sequence number")
    args, _ = parser.parse_known_args("--seq")
    seq_to_run = args.seq

    seq_to_run = [1, 2, 3, 4, 5]
    data_all = load_data(seq_to_run)

    if 0:
        tm(data_all, param, demo_output_type=demo_output_type, realtime=False, write_log=False)
    if 0:
        tm_kalman(data_all, param, demo_output_type="video", realtime=False, write_log=False)
    if 0:
        tm_adap(data_all, param, demo_output_type=demo_output_type, realtime=False, write_log=False)
    if 1:
        tm_final(
            data_all, param, demo_output_type=demo_output_type, realtime=True, write_log=False, log_prefix="no_kalman"
        )
