from skimage.color import hsv2rgb


import numpy as np
import h5py
import cv2

from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp


def binary_search_h5_dset(dset, x, l=None, r=None, side='left'):
    l = 0 if l is None else l
    r = len(dset)-1 if r is None else r
    while l <= r:
        mid = l + (r - l)//2
        midval = dset[mid]
        if midval == x:
            return mid
        elif midval < x:
            l = mid + 1
        else:
            r = mid - 1
    if side == 'left':
        return l
    return r

def binary_search_h5_gt_timestamp(hdf_path, l, r, x, side='left'):
    f = h5py.File(hdf_path, 'r')
    return binary_search_h5_dset(f['davis']['left']['pose_ts'], x, l=l, r=r, side=side)

def binary_search_h5_timestamp(hdf_path, l, r, x, side='left'):
    f = h5py.File(hdf_path, 'r')
    return binary_search_h5_dset(f['events/ts'], x, l=l, r=r, side=side)

def get_events_by_idx(path, start_idx, end_idx):
    with h5py.File(path, "r") as h5_file:
        xs = h5_file['events/xs'][start_idx:end_idx]
        ys = h5_file['events/ys'][start_idx:end_idx]
        ts = h5_file['events/ts'][start_idx:end_idx]
        ps = h5_file['events/ps'][start_idx:end_idx]

    return xs, ys, ts, ps 

def vis_flow(flow):
    mag = np.linalg.norm(flow, axis=2)
    a_mag = np.min(mag)
    b_mag = np.max(mag)

    ang = np.arctan2(flow[...,0], flow[...,1])
    ang += np.pi
    ang *= 180. / np.pi / 2.
    ang = ang.astype(np.uint8)
    hsv = np.zeros(list(flow.shape[:2]) + [3], dtype=np.uint8)
    hsv[:, :, 0] = ang
    hsv[:, :, 1] = 255
    hsv[:, :, 2] = np.clip(mag, 0, 255)
    hsv[:, :, 2] = ((mag - a_mag).astype(np.float32) * (255. / (b_mag - a_mag + 1e-32))).astype(np.uint8)
    flow_rgb = hsv2rgb(hsv)
    return 255 - (flow_rgb * 255).astype(np.uint8)

def draw_color_wheel_np(width, height):
    color_wheel_x = np.linspace(-width / 2.,
                                 width / 2.,
                                 width)
    color_wheel_y = np.linspace(-height / 2.,
                                 height / 2.,
                                 height)
    color_wheel_X, color_wheel_Y = np.meshgrid(color_wheel_x, color_wheel_y)
    flow = np.dstack([color_wheel_X, color_wheel_Y])
    color_wheel_rgb = vis_flow(flow)
    return color_wheel_rgb

def vis_events(events, imsize):
    res = np.zeros(imsize, dtype=np.uint8).ravel()
    x, y = map(lambda x: x.astype(int), events[:2])
    i = np.ravel_multi_index([y, x], imsize)
    np.maximum.at(res, i, np.full_like(x, 255, dtype=np.uint8))
    return np.tile(res.reshape(imsize)[..., None], (1, 1, 3))

def collage(flow_rgb, events_rgb):
    flow_rgb = flow_rgb[::-1]

    orig_h, orig_w, c = flow_rgb[0].shape
    h = orig_h + flow_rgb[1].shape[0]
    w = orig_w + events_rgb.shape[1]

    res = np.zeros((h, w, c), dtype=events_rgb.dtype)
    res[:orig_h, :orig_w] = flow_rgb[0]
    res[:orig_h, orig_w:] = events_rgb

    k = 0
    for img in flow_rgb[1:]:
        h, w = img.shape[:2]
        l = k + w
        res[orig_h:orig_h+h, k:l] = img
        k = l
    return res

def crop_car_events(events):
    x, y, p, t = events

    x = x[y < 190]
    p = p[y < 190]
    t = t[y < 190]
    y = y[y < 190]

    return x, y, p, t


def get_random_selected_correspondence_points(e1, e2, num_track_points=100, mask=None):

    num_events = len(e1[0])

    if mask is not None:
        track_idx = np.array([], dtype=int)

        while len(track_idx) < num_track_points:
            idx = np.random.randint(0, num_events * 0.3)
            x1, y1 = int(e1[0][idx]), int(e1[1][idx])
            x2, y2 = int(e2[0][idx]), int(e2[1][idx])

            if mask[y1, x1] and mask[y2, x2]:
                track_idx = np.append(track_idx, idx)

    else:
        track_idx = np.random.randint(0, len(e1[0]), size=num_track_points)


    x = e1[0][track_idx]
    y = e1[1][track_idx]
    x_ = e2[0][track_idx]
    y_ = e2[1][track_idx]

    points1 = np.dstack([x, y]).squeeze()
    points2 = np.dstack([x_, y_]).squeeze()

    return points1, points2



def vis_events_flow(flow, events, event_forward, event_backward, imsize):

    # visualization
    flow_img = vis_flow(flow)
    flow_img[-50:, -50:] = draw_color_wheel_np(50, 50)
    event_img = vis_events(events, imsize)
    forward_img = vis_events(event_forward, imsize)
    backward_img = vis_events(event_backward, imsize)
    diff_img = np.abs(forward_img - backward_img)

    # Add labels to images
    cv2.putText(flow_img, "Flow Prediction", (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(event_img, "Event Image", (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(forward_img, "Forward", (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(backward_img, "Backward", (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(diff_img, "diff", (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    img_top = np.hstack([flow_img, event_img, np.zeros_like(event_img)])
    img_bot = np.hstack([forward_img, backward_img, diff_img])
    img = np.vstack([img_top, img_bot])

    return img



def move_flow(flow, events, imsize, forward=True):

    x, y, t, p = events

    x_idx = x.astype(int)
    y_idx = y.astype(int)

    if forward:
        x_ = x + flow[y_idx, x_idx, 1]
        y_ = y + flow[y_idx, x_idx, 0]
    else:
        x_ = x - flow[y_idx, x_idx, 1]
        y_ = y - flow[y_idx, x_idx, 0]

    x_ = np.clip(x_, 0, imsize[1]-1)
    y_ = np.clip(y_, 0, imsize[0]-1)

    return (x_, y_, t, p)


def move_flow_with_time(flow, events, imsize, forward=True):

    x, y, t, p = events

    if forward:
        t = (t[-1] - t) / (t[-1] - t[0])
    else:
        t = (t[0] - t) / (t[-1] - t[0])

    x_idx = x.astype(int)
    y_idx = y.astype(int)

    y_ = y + t * flow[y_idx, x_idx, 1]
    x_ = x + t * flow[y_idx, x_idx, 0]

    x_ = np.clip(x_, 0, imsize[1]-1)
    y_ = np.clip(y_, 0, imsize[0]-1)

    return (x_, y_, t, p)

def crop_center_events(events, imsize, trim_size=(2, 5)):
    x, y, t, p = events

    mask = (x >= trim_size[1]) & (x < imsize[1]-trim_size[1]) &  \
            (y >= trim_size[0]) & (y < imsize[0]-trim_size[0])

    x = x[mask] - trim_size[1]
    y = y[mask] - trim_size[0]
    t = t[mask]
    p = p[mask]

    return [x, y, t, p]

def get_pos_events(events):
    x, y, t, p = events

    x = x[p == 1.0]
    y = y[p == 1.0]
    t = t[p == 1.0]
    p = p[p == 1.0]

    # x = x[y < 200]
    # t = t[y < 200]
    # p = p[y < 200]
    # y = y[y < 200]

    return [x, y, t, p]

def get_neg_events(events):
    x, y, t, p = events

    x = x[p == 0.0]
    y = y[p == 0.0]
    t = t[p == 0.0]
    p = p[p == 0.0]

    return [x, y, t, p]

def get_new_track_points_idx(track_pts, events, num_events_sample=2000):

    xs, ys, ts, ps = track_pts
    x_, y_, t_, p_ = events

    # filter out eariler events
    x_ = x_[-num_events_sample:]
    y_ = y_[-num_events_sample:]
    t_ = t_[-num_events_sample:]
    p_ = p_[-num_events_sample:]

    # create list of absolute index of positive and negative events
    idx_list = np.arange(len(x_)) + len(events[0]) - num_events_sample

    new_events = []
    for x, y, t, p in zip(xs, ys, ts, ps):

        mask = p_ == p
        new_x = x_[mask]
        new_y = y_[mask]

        dist = (x - new_x)**2  + (y - new_y)**2
        idx = np.argmin(dist)
        abs_idx = idx_list[idx]

        new_events.append(abs_idx)

    return np.array(new_events, dtype=np.int)


def interp_rotation_matrix(start_R, end_R, start_time, end_time, slerp_time):

    rotations = R.from_matrix([start_R, end_R])
    key_times = [start_time, end_time]

    slerp = Slerp(key_times, rotations)
    interp_rots = slerp([slerp_time])

    return interp_rots.as_matrix().squeeze()

def interp_rigid_matrix(start_R, end_R, start_time, end_time, slerp_time):

    ratio = (slerp_time - start_time) / (end_time - start_time)
    slerp_translation = ratio * start_R[0:3, 3] + (1 - ratio) * end_R[0:3, 3]
    slerp_rotation = interp_rotation_matrix(start_R[0:3, 0:3], end_R[0:3, 0:3], start_time, end_time, slerp_time)

    interp_rigid = np.eye(4, 4)
    interp_rigid[0:3, 0:3] = slerp_rotation
    interp_rigid[0:3, 3] = slerp_translation

    return interp_rigid


def inverse_rigid_matrix(matrix):

    inv_matrix = np.zeros((len(matrix), 4, 4))
    R = matrix[:, 0:3, 0:3]
    t = matrix[:, 0:3, 3]
    R_inv = R.transpose(0, 2, 1)


    for i, (ro, tn) in enumerate(zip(R_inv, t)):

        inv_matrix[i, 0:3, 0:3] = ro
        inv_matrix[i, 0:3, 3] = -ro @ tn
        inv_matrix[i, 3, 3] = 1

    return inv_matrix

def world_to_camera_frames(T_wc):

    T_c = []

    last_frame = np.eye(4)
    for T_ in T_wc:
        T_c.append(np.linalg.inv(T_) @ last_frame)

        last_frame = T_

    return np.array(T_c)

def camera_to_world_frames(T_c):

    T_wc = []

    total_trans = np.eye(4)
    for T in T_c:
        
        total_trans = total_trans @ np.linalg.inv(T)
        T_wc.append(total_trans)

    return np.array(T_wc)
    

# m = np.zeros((2, 4, 4))
# m[:, 0:3, 0:3] = R.random(2).as_matrix()
# m[:, 0:3, 3] = np.random.randint(5, size=(2, 3))
# m[:, 3, 3] = 1


# print(m)
# print()

# m = world_to_camera_frames(m)
# m = camera_to_world_frames(m)

# print(m)