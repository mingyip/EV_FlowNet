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



inv_pose_list = []
with h5py.File(pre_gen_flow, "r") as h5_file:
    for i in range(1, total_num_frames-skip_frames, skip_frames):

        if i < 100:   continue
        if i > 800:    break

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
        p1, p2 = get_random_selected_correspondence_points(pos_backward, pos_forward, num_track_points=200, mask=flow_mask)
        # n1, n2 = get_random_selected_correspondence_points(neg_backward, neg_forward, num_track_points=200, mask=flow_mask)
        # pt1 = np.vstack([p1, n1])
        # pt2 = np.vstack([p2, n2])

        E, mask = cv2.findEssentialMat(p1, p2, cameraMatrix=camIntrinsic, method=cv2.RANSAC, prob=0.999, threshold=5.0)
        points, R, t, mask = cv2.recoverPose(E, p1, p2, mask=mask)

        S = np.eye(4)
        S[0:3, 0:3] = np.transpose(R)
        S[0:3, 3]   = - np.transpose(R) @ np.squeeze(t)
        inv_pose_list.append(S)


        gt_start_idx = binary_search_h5_gt_timestamp(gt_path, 0, None, start_t, side='right')
        gt_end_idx = binary_search_h5_gt_timestamp(gt_path, 0, None, end_t, side='right')

        # Get ground truth
        with h5py.File(gt_path, "r") as h5_file:
            gt_pose = h5_file['davis']['left']['pose']
            gt_ts = h5_file['davis']['left']['pose_ts']

            gt_start_t1 = gt_ts[gt_start_idx]
            gt_start_t2 = gt_ts[gt_start_idx + 1]
            gt_end_t1 = gt_ts[gt_end_idx]
            gt_end_t2 = gt_ts[gt_end_idx + 1]

            gt_start_p1 = gt_pose[gt_start_idx]
            gt_start_p2 = gt_pose[gt_start_idx + 1]
            gt_end_p1 = gt_pose[gt_end_idx]
            gt_end_p2 = gt_pose[gt_end_idx + 1]

            print(gt_start_t1, start_t, gt_start_t2)
            print(gt_end_t1, end_t, gt_end_t2)

        

        raise


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
z = np.zeros_like(x)
# z = coord_list[:, 2]
idx = np.arange(len(x))

fig = plt.figure()
ax = plt.axes(projection="3d")

ax.plot3D(x, y, z, 'gray')
ax.scatter3D(x, y, z, c=idx, cmap='hsv')
plt.show()






# with h5py.File(gt_path, "r") as h5_file:
#     gt_pose = h5_file['davis']['left']['pose']
#     gt_ts = h5_file['davis']['left']['pose_ts']


# with h5py.File(image_ts_path, "r") as h5_file:
#     img_ts = h5_file['davis']['left']['image_raw_ts']
#     img = h5_file['davis']['left']['image_raw']
#     img = np.array(img[offset_img_idx])
