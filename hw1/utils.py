import os
import sys

import cv2
import matplotlib
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output, display


def load_data(seq_to_run=[1]):
    """
    the image coordinate I use is ↓x, →y, which alignes to the array index
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


# Metrics Calculation ----------------------------------
def calcu_iou(rec1: dict, rec2: dict):
    """rec= {"x":, "y":, "w":, "h":}"""
    inter_area = max(0, min(rec1["x"] + rec1["h"], rec2["x"] + rec2["h"]) - max(rec1["x"], rec2["x"])) * max(
        0, min(rec1["y"] + rec1["w"], rec2["y"] + rec2["w"]) - max(rec1["y"], rec2["y"])
    )
    union_area = rec1["w"] * rec1["h"] + rec2["w"] * rec2["h"] - inter_area
    iou = inter_area / union_area
    return iou


def print_dict(d, indent=0):
    for key, value in d.items():
        if isinstance(value, dict):
            print("    " * indent + str(key) + ": ")
            print_dict(value, indent + 1)
        else:
            print("    " * indent + str(key) + ": " + str(value))


def calcu_center_dist(rec1: dict, rec2: dict):
    """rec= {"x":, "y":, "w":, "h":}"""
    center1 = (rec1["x"] + rec1["w"] / 2, rec1["y"] + rec1["h"] / 2)
    center2 = (rec2["x"] + rec2["w"] / 2, rec2["y"] + rec2["h"] / 2)
    dist = np.sqrt((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2)
    return dist


def calcu_ade(predicted, gt):
    """
    predicted(list): [center_x, center_y, center_z, boundingbox_x, boundingbox_y, boundingbox_z, heading, vel_x, vel_y, valid]
    """
    ade = 0
    for i in range(len(predicted)):
        ade += np.sqrt((predicted[i][0] - gt[i][0]) ** 2 + (predicted[i][1] - gt[i][1]) ** 2)
    return ade / len(predicted)


def calcu_fde(predicted, gt):
    """
    predicted(list): [center_x, center_y, center_z, boundingbox_x, boundingbox_y, boundingbox_z, heading, vel_x, vel_y, valid]
    """

    return np.sqrt((predicted[-1][0] - gt[-1][0]) ** 2 + (predicted[-1][1] - gt[-1][1]) ** 2)


def calcu_p_norm(track, bb_gt):
    """
    center_dist: distance between the center of the template and the ground truth
    bb_gt: bounding box of the ground truth

    refer: https://github.com/SilvioGiancola/TrackingNet-devkit/blob/master/metrics.py
    Note that the image coordinate I use is ↓x, →y, while the above link is →x, ↓y
    """
    center1 = (track["y"] + track["w"] / 2, track["x"] + track["h"] / 2)
    center2 = (bb_gt["y"] + bb_gt["w"] / 2, bb_gt["x"] + bb_gt["h"] / 2)
    delta_x = center1[0] - center2[0]
    delta_y = center1[1] - center2[1]
    return np.sqrt((delta_x / bb_gt["h"]) ** 2 + (delta_y / bb_gt["w"]) ** 2)


# Visualization ----------------------------------
def draw_polygon(track_bb: dict, ax, color="g"):  # bb means bounding box
    """_summary_

    Args:
        track_bb (dict): bbox in format of (x, y, w, h), expressed in opencv coordinate system
        ax (_type_): _description_
        color (str, optional): _description_. Defaults to "g".
    """
    polygon = patches.Polygon(  # width is x, height is y. reverse to opencv
        [
            (track_bb["y"], track_bb["x"]),  # x1, y1
            (track_bb["y"], track_bb["x"] + track_bb["h"]),  # x2, y2
            (track_bb["y"] + track_bb["w"], track_bb["x"] + track_bb["h"]),
            (track_bb["y"] + track_bb["w"], track_bb["x"]),
        ],
        True,
        linewidth=1,
        edgecolor=color,
        facecolor="none",
    )
    ax.add_patch(polygon)


def restrain_bbox(bbox: dict, img_size):
    """
    bbox: {"x":, "y":, "h":, "w":}
    img_size: (height, width)"""
    bbox["x"] = max(0, bbox["x"])
    bbox["y"] = max(0, bbox["y"])
    # bbox["x2"] = min(img_size[0], bbox["x"]+bbox["h"])
    # bbox["y2"] = min(img_size[1], bbox["y"]+bbox["w"])
    bbox["h"] = min(img_size[0] - bbox["x"], bbox["h"])
    bbox["w"] = min(img_size[1] - bbox["y"], bbox["w"])
    return bbox


def draw_in_plt(fig, ax, img, template, gt, search_region, match_result, in_nb=False):
    ax[0].cla()
    ax[0].imshow(img)
    draw_polygon(template, ax[0], color="r")
    draw_polygon(gt, ax[0], color="b")

    temp_img = img[template["x"] : template["x"] + template["h"], template["y"] : template["y"] + template["w"]]
    ax[1].cla()
    ax[1].imshow(temp_img)
    ax[1].set_title("template")

    ax[2].cla()
    ax[2].imshow(match_result, cmap="gray")
    ax[2].set_title("match result")

    search_img = img[search_region["x1"] : search_region["x2"], search_region["y1"] : search_region["y2"]]
    ax[3].cla()
    ax[3].imshow(search_img)
    ax[3].set_title("search region")

    if in_nb:  # in notebook
        display(fig)
        clear_output(wait=True)
        plt.close()
    else:
        plt.pause(0.1)


def draw_demo_img(img_demo, template, gt, iou=None, center_dist=None, search_region=None):
    temp = template  # for compactness
    sr = search_region
    cv2.rectangle(img_demo, (temp["y"], temp["x"]), (temp["y"] + temp["w"], temp["x"] + temp["h"]), (255, 0, 0), 2)
    cv2.rectangle(img_demo, (gt["y"], gt["x"]), (gt["y"] + gt["w"], gt["x"] + gt["h"]), (0, 0, 255), 2)
    if sr:
        # cv2.rectangle(img_demo, (sr["y1"], sr["x1"]), (sr["y2"], sr["x2"]), (0, 255, 0), 2)
        cv2.rectangle(img_demo, (sr["y"], sr["x"]), (sr["y"] + sr["w"], sr["x"] + sr["h"]), (0, 255, 0), 2)
    cv2.putText(img_demo, "Template", (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(img_demo, "Ground Truth", (0, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(img_demo, "Search Region", (0, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    if iou:
        cv2.putText(img_demo, f"iou: {iou:.2f}", (0, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    if center_dist:
        cv2.putText(
            img_demo, f"center_dist: {center_dist:.2f}", (0, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2
        )
    return cv2.cvtColor(img_demo, cv2.COLOR_RGB2BGR)


def draw_road_polylines(fig, ax, road_polylines):
    for polyline in road_polylines:
        map_type = polyline[0, 6]
        if map_type == 6:
            ax.plot(polyline[:, 0], polyline[:, 1], "w", linestyle="dashed", linewidth=1)
        elif map_type == 7:
            ax.plot(polyline[:, 0], polyline[:, 1], "w", linestyle="solid", linewidth=1)
        elif map_type == 8:
            ax.plot(polyline[:, 0], polyline[:, 1], "w", linestyle="solid", linewidth=1)
        elif map_type == 9:
            ax.plot(polyline[:, 0], polyline[:, 1], "xkcd:yellow", linestyle="dashed", linewidth=1)
        elif map_type == 10:
            ax.plot(polyline[:, 0], polyline[:, 1], "xkcd:yellow", linestyle="dashed", linewidth=1)
        elif map_type == 11:
            ax.plot(polyline[:, 0], polyline[:, 1], "xkcd:yellow", linestyle="solid", linewidth=1)
        elif map_type == 12:
            ax.plot(polyline[:, 0], polyline[:, 1], "xkcd:yellow", linestyle="solid", linewidth=1)
        elif map_type == 13:
            ax.plot(polyline[:, 0], polyline[:, 1], "xkcd:yellow", linestyle="dotted", linewidth=1)
        elif map_type == 15:
            ax.plot(polyline[:, 0], polyline[:, 1], "k", linewidth=1)
        elif map_type == 16:
            ax.plot(polyline[:, 0], polyline[:, 1], "k", linewidth=1)


def draw_pred_traj(fig, ax, traj, gt, road_polylines, margin=20):
    fig.set_facecolor("xkcd:white")
    ax.set_facecolor("xkcd:grey")
    draw_road_polylines(fig, ax, road_polylines)
    (traj1,) = ax.plot(traj[:, 0], traj[:, 1], "r", linestyle="solid", linewidth=2, label="Predicted Trajectory")
    (traj2,) = ax.plot(gt[:, 0], gt[:, 1], "b", linestyle="dashed", linewidth=1, label="Ground Truth Trajectory")
    ax.axis([-margin + gt[0][0], margin + gt[0][0], -margin + gt[0][1], margin + gt[0][1]])


def draw_pred_status_ego(fig, ax, agent_idx, time_idx, all_agent_trajs, tracks, road_polylines, margin=70):
    """
    Description: draw the predicted statuss in the ego vehicle's frame

    sdc_current_status: [center_x, center_y, center_z, boundingbox_x, boundingbox_y, boundingbox_z, heading, vel_x, vel_y, valid]
    """
    ax.cla()
    sdc_current_status = all_agent_trajs[tracks[agent_idx]][time_idx]
    print(f"==>> sdc_current_status: {sdc_current_status}")

    fig.set_facecolor("xkcd:grey")
    ax.set_facecolor("xkcd:grey")
    draw_road_polylines(fig, ax, road_polylines)
    ax.axis(
        [
            -margin + sdc_current_status[0],
            margin + sdc_current_status[0],
            -margin + sdc_current_status[1],
            margin + sdc_current_status[1],
        ]
    )
    rect = patches.Rectangle(
        (sdc_current_status[0] - 2, sdc_current_status[1] - 1),
        sdc_current_status[3],
        sdc_current_status[4],
        sdc_current_status[6] * 180 / np.pi,
        linewidth=1,
        edgecolor="r",
        facecolor="none",
    )
    ax.add_patch(rect)
    plt.pause(0.001)
    # plt.show()
    # filename = "./viz.png"
    # plt.savefig(filename)
    # plt.close()


def draw_pred_status_world(fig, ax, agent_idx, time_idx, all_agent_trajs, tracks, road_polylines):
    """
    Description: draw the predicted ego vehicle's trajectory in the world frame
    """
    sdc_current_status = all_agent_trajs[tracks[agent_idx]][time_idx]
    print(f"==>> sdc_current_status: {sdc_current_status}")

    fig.set_facecolor("xkcd:grey")
    ax.set_facecolor("xkcd:grey")
    draw_road_polylines(fig, ax, road_polylines)
    ax.axis(  # set axis to the original status
        [
            -70 + all_agent_trajs[tracks[agent_idx]][0][0],
            70 + all_agent_trajs[tracks[agent_idx]][0][0],
            -70 + all_agent_trajs[tracks[agent_idx]][0][1],
            70 + all_agent_trajs[tracks[agent_idx]][0][1],
        ]
    )
    rect = patches.Rectangle(
        (sdc_current_status[0] - 2, sdc_current_status[1] - 1),
        sdc_current_status[3],
        sdc_current_status[4],
        sdc_current_status[6] * 180 / np.pi,
        linewidth=1,
        edgecolor="r",
        facecolor="none",
    )
    ax.add_patch(rect)
    plt.pause(0.001)
    ax.cla()


# Image Processing ----------------------------------
def get_obj_in_img_hsv(img, obj_hsv, draw=False):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    img_bin = cv2.inRange(img_hsv, np.array(obj_hsv["low"]), np.array(obj_hsv["high"]))
    img_erode = cv2.erode(img_bin, None, iterations=1)
    img_dil = cv2.dilate(img_erode, None, iterations=2)
    img_contour, _ = cv2.findContours(img_dil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_area = 0
    for contour in img_contour:
        area = cv2.contourArea(contour.squeeze())
        if area > max_area:
            max_area = area
            max_contour = contour
    x, y, w, h = cv2.boundingRect(max_contour)
    if draw:
        cv2.imshow("img_temp_bin", img_bin)
        cv2.imshow("dilation", img_dil)
        contour = cv2.drawContours(img.copy(), [max_contour], -1, (0, 255, 0), 3)
        contour = cv2.rectangle(contour, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imshow("contour", contour)
    return x, y, w, h


def calcu_temp_delta_hsv(img_temp, hsv, size_ratio={"w": 0.9, "h": 0.5}):
    """
    img_temp: the template image
    hsv: the hsv range of the object
    size_ratio: the specified ratio of the object size to the template size
    draw: whether to draw the result
    """
    x, y, w, h = get_obj_in_img_hsv(img_temp, hsv, draw=False)

    delta_w = int(w - img_temp.shape[1] * size_ratio["w"])
    delta_h = int(h - img_temp.shape[0] * size_ratio["h"])
    delta_size = {"w": delta_w, "h": delta_h}

    center_of_temp = [img_temp.shape[1] // 2, img_temp.shape[0] // 2]
    cX = x + w // 2
    cY = y + h // 2
    delta_center = {
        "x": cY - center_of_temp[1],
        "y": cX - center_of_temp[0],
    }  # note the transformation from opencv to array

    return delta_center, delta_size


def calcu_temp_delta_seg(img_temp, seg, draw=False):
    pass
