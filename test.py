from pathlib import Path
import imageio
from imageio import imwrite
from skimage.color import hsv2rgb
import numpy as np
from tqdm import tqdm

import copy
import cv2

import sys
cur_path = Path(__file__).parent.resolve()
module_name = cur_path.name
sys.path.append(str(cur_path.parent))
OpticalFlow = __import__(module_name).OpticalFlow

data_base = cur_path/'data'/'events'
out_path = cur_path/'res'
out_path.mkdir(parents=True, exist_ok=True)

# events. Note that polarity values are in {-1, +1}
events = np.load(str(data_base/'outdoor.npy'))
# number of frames per second
fps = 30
# height and width of images
imsize = 280, 360
# window size in microseconds
dt = 1. / fps

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

def move_flow(flow, events, imsize, forward=True):

    x, y, t, p = events

    if forward:
        t = (t[-1] - t) / (t[-1] - t[0])
    else:
        t = (t[0] - t) / (t[-1] - t[0])

    x = x.astype(int)
    y = y.astype(int)

    y_ = y + t * flow[y, x, 0]
    x_ = x + t * flow[y, x, 1]

    x_ = np.clip(x_, 0, imsize[1]-1).astype(int)
    y_ = np.clip(y_, 0, imsize[0]-1).astype(int)

    return (x_, y_, t, p)

def get_pos_events(events):
    x, y, t, p = events

    x = x[p == 1.0]
    y = y[p == 1.0]
    t = t[p == 1.0]
    p = p[p == 1.0]

    x = x[y < 200]
    t = t[y < 200]
    p = p[y < 200]
    y = y[y < 200]

    return [x, y, t, p]

def get_neg_events(events):
    x, y, t, p = events

    x = x[p == 0.0]
    y = y[p == 0.0]
    t = t[p == 0.0]
    p = p[p == 0.0]

    return [x, y, t, p]

def get_events_by_idx(events, idx):
    x, y, t, p = events

    x = x[idx]
    y = y[idx]
    t = t[idx]
    p = p[idx]

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


of = OpticalFlow(imsize)

x, y, t, p = events

start_t = t[0]
stop_t = t[-1]
frame_ts = np.arange(start_t, stop_t, dt)
frame_ts = np.append(frame_ts, [frame_ts[-1] + dt])
num_frames = len(frame_ts) - 1
idx_array = np.searchsorted(t, frame_ts)


for k, (b, e) in tqdm(enumerate(zip(idx_array[:-1], idx_array[1:])), total = num_frames):

    if k < 3500:
        continue

    frame_events = get_pos_events([x[b:e] for x in events])
    flow = of([frame_events], [frame_ts[k]], [frame_ts[k+1]], return_all=True)
    flow = tuple(map(np.squeeze, flow))

    event_img = vis_events(frame_events, imsize)
    forward_move = move_flow(flow[3], frame_events, imsize, forward=True)
    backward_move = move_flow(flow[3], frame_events, imsize, forward=False)

    # visualization
    flow_img = vis_flow(flow[3])
    flow_img[-50:, -50:] = draw_color_wheel_np(50, 50)
    forward_img = vis_events(forward_move, imsize)
    backward_img = vis_events(backward_move, imsize)
    diff_img = np.abs(forward_img - backward_img)

    # Add labels to images
    cv2.putText(flow_img, "Flow Prediction", (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(event_img, "Event Image", (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(forward_img, "Forward", (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(backward_img, "Backward", (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(diff_img, "diff", (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    img_top = np.hstack([flow_img, event_img, np.zeros_like(event_img)])
    img_bot = np.hstack([forward_img, backward_img, diff_img])

    track_idx = np.random.randint(0, len(frame_events[0]), size=20)

    for i in track_idx:
        x = int(forward_move[0][i])
        y = int(forward_move[1][i])
        x_ = int(backward_move[0][i]) + 360
        y_ = int(backward_move[1][i])
        color = np.random.randint(0, 255, size=(1, 3), dtype=np.uint8).squeeze()
        color = ( int (color [ 0 ]), int (color [ 1 ]), int (color [ 2 ]))
        img_bot = cv2.circle(img_bot, (x, y), radius=3, color=color, thickness=-1)
        img_bot = cv2.circle(img_bot, (x_, y_), radius=3, color=color, thickness=-1)
        img_bot = cv2.line(img_bot, (x,y), (x_,y_), color=color, thickness=1)

    img = np.vstack([img_top, img_bot])
    cv2.imshow("", img)
    cv2.imwrite(f"{k}.png", img)

    cv2.waitKey(100000)
    raise

