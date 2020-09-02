from pathlib import Path
import imageio
from imageio import imwrite
from skimage.color import hsv2rgb
import numpy as np
from tqdm import tqdm

from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

import h5py
import copy
import cv2
import sys

from util import *



skip_frames = 1
offset_img_idx = 1968
total_num_frames = 26605
sensor_size  = np.array([260, 346])
padding      = np.array([2, 5])
cropped_size = sensor_size - 2 * padding
camIntrinsic = np.array([[223.9940010790056, 0, 170.7684322973841], [0, 223.61783486959376, 128.18711828338436], [0, 0, 1]])



events_path =   "/mnt/Data1/dataset/evflow-data/outdoor_day2/h5_events/outdoor_day2_data.h5"
# image_ts_path = "/mnt/Data1/dataset/evflow-data/outdoor_day2/outdoor_day2_data.hdf5" # outdoor2
gt_path =       "/mnt/Data1/dataset/evflow-data/outdoor_day2/outdoor_day2_gt.hdf5"  # outdoor2
pre_gen_flow =  "/mnt/Data3/outdoor_day2_tf_output_trim_skip1.h5"
# pre_gen_flow =  "/mnt/Data2/EV_FlowNet_daniilidis/outdoor_day2_tf_output_trim_skip4.h5"




# gt_trans = np.empty([1, 3])
gt_camera_frame = []
predict_camera_frame = []

with h5py.File(pre_gen_flow, "r") as h5_file:
    for i in range(1, total_num_frames-skip_frames, skip_frames):

        if i < 400:   continue
        if i > 440:    break

        # Get events timestamp
        start_t = h5_file["prev_images"]["image{:09d}".format(i)].attrs['timestamp']
        end_t = h5_file["next_images"]["image{:09d}".format(i)].attrs['timestamp']
        start_ev_idx = binary_search_h5_timestamp(events_path, 0, None, start_t)
        end_ev_idx = binary_search_h5_timestamp(events_path, 0, None, end_t)

        # Get flow estimates
        flow = np.array(h5_file["flows"]["flow{:09d}".format(i)])
        flow_img = vis_flow(flow)
        flow_img[-50:, -50:] = draw_color_wheel_np(50, 50)
        flow_mask = (np.abs(flow[:, :, 0]) > 0.01) & (np.abs(flow[:, :, 1]) > 0.01)

        # Get events
        events = get_events_by_idx(events_path, start_ev_idx, end_ev_idx)
        pos = get_pos_events(events)
        neg = get_neg_events(events)
        pos = crop_car_events(crop_center_events(pos, sensor_size, trim_size=padding))
        neg = crop_car_events(crop_center_events(neg, sensor_size, trim_size=padding))

        # Move events forward and backward
        pos_backward = move_flow_with_time(flow, pos, cropped_size, forward=False)
        pos_forward = move_flow_with_time(flow, pos, cropped_size, forward=True)
        neg_backward = move_flow_with_time(flow, neg, cropped_size, forward=False)
        neg_forward = move_flow_with_time(flow, neg, cropped_size, forward=True)

        # Get correpsondence points and R, t
        p1, p2 = get_random_selected_correspondence_points(pos_backward, pos_forward, num_track_points=100, mask=flow_mask)
        # n1, n2 = get_random_selected_correspondence_points(neg_backward, neg_forward, num_track_points=200, mask=flow_mask)
        # pt1 = np.vstack([p1, n1])
        # pt2 = np.vstack([p2, n2])

        E, mask = cv2.findEssentialMat(p1, p2, cameraMatrix=camIntrinsic, method=cv2.RANSAC, prob=0.999, threshold=1.5)
        points, R, t, mask = cv2.recoverPose(E, p1, p2, mask=mask)

        gt_pt1_idx = binary_search_h5_gt_timestamp(gt_path, 0, None, start_t, side='right')
        gt_pt2_idx = binary_search_h5_gt_timestamp(gt_path, 0, None, end_t, side='right')

        # Get ground truth
        with h5py.File(gt_path, "r") as gt_file:
            gt_pose = gt_file['davis']['left']['pose']
            gt_ts = gt_file['davis']['left']['pose_ts']

            gt_pt1_interp_begin_t   = gt_ts[gt_pt1_idx]
            gt_pt1_interp_end_t     = gt_ts[gt_pt1_idx + 1]
            gt_pt2_interp_begin_t   = gt_ts[gt_pt2_idx]
            gt_pt2_interp_end_t     = gt_ts[gt_pt2_idx + 1]

            gt_pose1_interp_begin   = gt_pose[gt_pt1_idx]
            gt_pose1_interp_end     = gt_pose[gt_pt1_idx + 1]
            gt_pose2_interp_begin   = gt_pose[gt_pt2_idx]
            gt_pose2_interp_end     = gt_pose[gt_pt2_idx + 1]

        ratio1 = (gt_pt1_interp_end_t - start_t) / (gt_pt1_interp_end_t - gt_pt1_interp_begin_t)
        ratio2 = (gt_pt2_interp_end_t - end_t) / (gt_pt2_interp_end_t - gt_pt2_interp_begin_t)
        twc1 = ratio1 * gt_pose1_interp_begin[0:3, 3] + (1 - ratio1) * gt_pose1_interp_end[0:3, 3]
        twc2 = ratio2 * gt_pose2_interp_begin[0:3, 3] + (1 - ratio2) * gt_pose2_interp_end[0:3, 3]

        twc1 = interp_rigid_matrix(gt_pose1_interp_begin, 
                                        gt_pose1_interp_end, 
                                        gt_pt1_interp_begin_t, 
                                        gt_pt1_interp_end_t, 
                                        start_t)

        twc2 = interp_rigid_matrix(gt_pose2_interp_begin, 
                                        gt_pose2_interp_end, 
                                        gt_pt2_interp_begin_t, 
                                        gt_pt2_interp_end_t, 
                                        end_t)

        t_c1_c2 = np.linalg.inv(twc1) @ twc2
        gt_camera_frame.append(t_c1_c2)

        # scale t
        t *= np.linalg.norm(twc2 - twc1)

        if i < 405:
            print(np.linalg.norm(twc2 - twc1))

        S = np.eye(4)
        S[0:3, 0:3] = R
        S[0:3, 3]   = np.squeeze(t)
        predict_camera_frame.append(S)

        # raise

        # Visualize images, flow and event images
        rgb_img = np.array(h5_file["prev_images"]["image{:09d}".format(i)])
        rgb_img = np.tile(rgb_img[..., np.newaxis], [1, 1, 3])

        pos_original = vis_events(pos, cropped_size)
        pos_backward_img = vis_events(pos_backward, cropped_size)
        pos_forward_img = vis_events(pos_forward, cropped_size)

        # neg_original = vis_events(neg, cropped_size)
        # neg_backward_img = vis_events(neg_backward, cropped_size)
        # neg_forward_img = vis_events(neg_forward, cropped_size)
        top = np.hstack([pos_original, rgb_img, flow_img])
        bot = np.hstack([pos_backward_img, pos_forward_img, pos_backward_img - pos_forward_img])

        # Draw correspondence points
        for pt1, pt2 in zip(p1, p2):
            x1, y1 = pt1
            x2, y2 = pt2

            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2 + cropped_size[1])
            y2 = int(y2)
            
            color = np.random.randint(0, 255, size=(1, 3), dtype=np.uint8).squeeze()
            color = (int(color [0]), int(color[1]), int(color[2]))
            bot = cv2.circle(bot, (x1, y1), radius=3, color=color, thickness=1)
            bot = cv2.circle(bot, (x2, y2), radius=3, color=color, thickness=1)
            bot = cv2.line(bot, (x1,y1), (x2,y2), color=color, thickness=1)

        img = np.vstack([top, bot])
        cv2.imshow("img", img)
        cv2.waitKey(1)



gt_camera_frame = np.array(gt_camera_frame)
predict_camera_frame = np.array(predict_camera_frame)

for i in range(len(gt_camera_frame)-1):

    q_i     = gt_camera_frame[i]
    q_i_1   = gt_camera_frame[i+1]
    q_i_inv = np.linalg.inv(q_i)

    p_i     = predict_camera_frame[i]
    p_i_1   = predict_camera_frame[i+1]
    p_i_inv = np.linalg.inv(p_i)

    ei = np.linalg.inv(q_i_inv @ q_i_1) @ (p_i_inv @ p_i_1)

    trans = ei[0:3, 3]
    print(ei)
    print()
    norm_trans = np.linalg.norm(trans)

    # print(i, norm_trans)
    if i > 4:
        raise



raise












# Convert to world coordinate
coord_list = []
total_trans = np.eye(4)
last_location = np.eye(4)
for i, p in enumerate(inv_pose_list):
    
    total_trans = total_trans @ p
    location = total_trans[:, 3]
    coord_list.append(location)

    # diff = np.linalg.norm(location - last_location)
    # print(diff)
    # last_location = total_trans[:, 3]

coord_list = np.array(coord_list)


# Visualize path 
x = coord_list[:, 0]
y = coord_list[:, 1]
# z = np.zeros_like(x)
z = coord_list[:, 2]


# x = ground_truth_trans[1:, 0]
# y = ground_truth_trans[1:, 1]
# z = ground_truth_trans[1:, 2]
idx = np.arange(len(x))

fig = plt.figure()
ax = plt.axes(projection="3d")

ax.plot3D(x, y, z, 'gray')
ax.scatter3D(x, y, z, c=idx, cmap='hsv')
plt.show()

# fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(5, 3))
# axes[0].plot(x1, y1)
# axes[1].plot(x2, y2)
# fig.tight_layout()