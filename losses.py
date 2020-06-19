import torch
import torchvision.transforms.functional as functional
import cv2
import numpy as np


class TotalLoss(torch.nn.Module):
    def __init__(self, smoothness_weight, weight_decay_weight=1e-4, device=torch.device('cpu')):
        super(TotalLoss, self).__init__()
        self._smoothness_weight = smoothness_weight
        self._weight_decay_weight = weight_decay_weight
        self.device = device

    def forward(self):
        pass
        return loss


def sim_ev_loss(events, flow, image_size):

    H, W = image_size
    x, y, t = events
    num_events = len(x)
    t_sum = np.zeros(image_size, dtype=np.float)
    ev_count = np.zeros(image_size, dtype=np.float)

    # Calculate new position after flow
    # clip value ensures new coordinates are still on the image.
    x_ = x.astype(float) + (1-t) * flow[0,y,x]
    y_ = y.astype(float) + (1-t) * flow[1,y,x]
    x = np.clip(x_, 0, W-1).astype(np.int)
    y = np.clip(y_, 0, H-1).astype(np.int)

    for i in range(num_events):
        t_sum[y[i], x[i]] += t[i]
        ev_count[y[i], x[i]] += 1

    forward_loss = np.divide(t_sum, ev_count + 0.00001)
    backward_loss = 1 - forward_loss
    backward_loss[ev_count == 0] = 0
    loss = np.square(forward_loss) + np.square(backward_loss)
    return np.sum(loss)


def ev_loss(events, flow, image_size):
    
    H, W = image_size
    x, y, t = events
    num_events = len(x)
    denom = np.zeros(image_size, dtype=np.float)
    nom = np.zeros(image_size, dtype=np.float)


    # Calculate new position after flow
    # clip value ensures new coordinates are still on the image.
    x_ = x + (1-t) * flow[0,y,x]
    y_ = y + (1-t) * flow[1,y,x]
    x_ = np.clip(x_, 0, W-1)
    y_ = np.clip(y_, 0, H-1)

    # Calculate 4 interpolation corners.
    x_max = np.clip(np.ceil(x_), 0, W-1).astype(int)
    x_min = np.clip(np.floor(x_), 0, W-1).astype(int)
    y_max = np.clip(np.ceil(y_), 0, H-1).astype(int)
    y_min = np.clip(np.floor(y_), 0, H-1).astype(int)

    y_top_ratio = y_ - y_min
    y_bot_ratio = 1 - y_top_ratio
    x_top_ratio = x_ - x_min
    x_bot_ratio = 1 - x_top_ratio

    # Calculate denominator and nominator values according to the algo
    # in the paper.
    denom_ymax_xmax = y_top_ratio * x_top_ratio
    denom_ymax_xmin = y_top_ratio * x_bot_ratio
    denom_ymin_xmax = y_bot_ratio * x_top_ratio
    denom_ymin_xmin = y_bot_ratio * x_bot_ratio

    nom_ymax_xmax = denom_ymax_xmax * t
    nom_ymax_xmin = denom_ymax_xmin * t
    nom_ymin_xmax = denom_ymin_xmax * t
    nom_ymin_xmin = denom_ymin_xmin * t

    for i in range(num_events):
        denom[y_max[i], x_max[i]] += denom_ymax_xmax[i]
        denom[y_max[i], x_min[i]] += denom_ymax_xmin[i]
        denom[y_min[i], x_max[i]] += denom_ymin_xmax[i]
        denom[y_min[i], x_min[i]] += denom_ymin_xmin[i]

        nom[y_max[i], x_max[i]] += nom_ymax_xmax[i]
        nom[y_max[i], x_min[i]] += nom_ymax_xmin[i]
        nom[y_min[i], x_max[i]] += nom_ymin_xmax[i]
        nom[y_min[i], x_min[i]] += nom_ymin_xmin[i]

    forward_loss = np.divide(nom, denom + 0.00001)
    backward_loss = 1 - forward_loss
    backward_loss[denom < 0.00001] = 0
    loss = np.square(forward_loss) + np.square(backward_loss)
    return np.sum(loss)

def flow_loss(events, flow, img_size, idx=0):

    x, y, p, t = events
    t = (t - t[0]) / (t[-1] - t[0])
    n = p == False

    # flow = np.clip(flow, 0.000001, 30) 
    flow = np.clip(flow, 0.000001, None)

    last_p = 0
    last_n = 0

    last_p1 = 0
    last_n1 = 0

    old_sign2 = old_sign1 = True

    for i in range(400):

        flow_ = flow * i * 0.0001
        pos_loss = sim_ev_loss((x[p], y[p], t[p]), flow_, img_size)
        neg_loss = sim_ev_loss((x[n], y[n], t[n]), flow_, img_size)


        pos_loss1 = ev_loss((x[p], y[p], t[p]), flow_, img_size)
        neg_loss1 = ev_loss((x[n], y[n], t[n]), flow_, img_size)

        sign1 = (pos_loss-last_p)>=0
        sign2 = (pos_loss1-last_p1)>=0

        txt1 = "      " if sign1 == old_sign1 else "change"
        txt2 = "      " if sign2 == old_sign2 else "change"
        print(i, txt1, txt2)

        old_sign1 = sign1
        old_sign2 = sign2

        last_p = pos_loss
        last_n = neg_loss

        last_p1 = pos_loss1
        last_n1 = neg_loss1

    print(i1, i2, v1, v2)
    raise

    return forward_p_loss + forward_n_loss + backward_p_loss + backward_n_loss

if __name__ == "__main__":
    import h5py

    f = h5py.File('office.h5', 'r')
    H, W = f['images/image000000000'].attrs['size']

    xs = f['events/xs']
    ys = f['events/ys']
    ts = f['events/ts']
    ps = f['events/ps']

    print("Begin time: ", ts[0], "End time: ", ts[-1])
    for idx, (i_, f_) in enumerate(zip(f['images'], f['flow'])):

        if idx < 21:
            flow = f['flow/' + f_].value
            start_t = f['images/' + i_].attrs['timestamp']
            continue

        end_t = f['images/' + i_].attrs['timestamp']
        mask = (ts[()]<end_t) & (ts[()]>=start_t)
        x, y, p, t = xs[mask], ys[mask], ps[mask], ts[mask]

        if len(x) == 0:
            flow = f['flow/' + f_].value
            continue

        loss = flow_loss((x,y,p,t), flow, (H, W), idx)
        print(idx, loss)
    pass