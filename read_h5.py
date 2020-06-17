import time
import h5py
import numpy as np
import os
import cv2 
from skimage.color import hsv2rgb


def vis_flow(flow):
    assert flow.shape[0] == 2, "Convert flow to channel first flow"

    mag = np.linalg.norm(flow, axis=0)
    a_mag = np.min(mag)
    b_mag = np.max(mag)

    ang = np.arctan2(flow[0], flow[1])
    ang += np.pi
    ang *= 180. / np.pi / 2.
    ang = ang.astype(np.uint8)
    hsv = np.zeros(list(flow.shape[1:]) + [3], dtype=np.uint8)
    hsv[:, :, 0] = ang
    hsv[:, :, 1] = 255
    hsv[:, :, 2] = np.clip(mag, 0, 255)
    hsv[:, :, 2] = ((mag - a_mag).astype(np.float32) * (255. / (b_mag - a_mag + 1e-32))).astype(np.uint8)
    flow_rgb = hsv2rgb(hsv)
    return 255 - (flow_rgb * 255).astype(np.uint8)


def vis_events(events, imsize):
    res = np.zeros(imsize, dtype=np.uint8).ravel()
    x, y = map(lambda x: x.astype(int), events[:2])
    i = np.ravel_multi_index([y, x], imsize)
    np.maximum.at(res, i, np.full_like(x, 255, dtype=np.uint8))
    return np.tile(res.reshape(imsize)[..., None], (1, 1, 3))

def ev_pol_loss(events, flow, img_size, idx1=0, output=False):

    if not output:
        return 0

    tttt = np.array([1,-1])
    flow += tttt[:, None, None]

    cum_total_time = 0
    cum_op1_time = 0
    cum_op2_time = 0
    cum_op3_time = 0
    cum_op4_time = 0
    start_total_time = time.time()
    for idx in range(30):
        flow_ = flow * (0.0001 * idx)   #TODO: remove alpha

        start_op4_time = time.time()
        H, W = img_size
        x, y, t = events
        denom = np.zeros(img_size, dtype=np.float)
        nom = np.zeros((H, W), dtype=np.float)
        cum_op4_time += time.time() - start_op4_time
        flow_rgb = vis_flow(flow_)

        start_op2_time = time.time()
        x_ = x + (1-t) * flow_[0,y,x]
        y_ = y + (1-t) * flow_[1,y,x]

        x_max = np.clip(np.ceil(x_), 0, W-1).astype(int)
        x_min = np.clip(np.floor(x_), 0, W-1).astype(int)
        y_max = np.clip(np.ceil(y_), 0, H-1).astype(int)
        y_min = np.clip(np.floor(y_), 0, H-1).astype(int)

        y_top_ratio = y_ - y_min
        y_bot_ratio = 1 - y_top_ratio
        x_top_ratio = x_ - x_min
        x_bot_ratio = 1 - x_top_ratio

        denom_ymax_xmax = y_top_ratio * x_top_ratio
        denom_ymax_xmin = y_top_ratio * x_bot_ratio
        denom_ymin_xmax = y_bot_ratio * x_top_ratio
        denom_ymin_xmin = y_bot_ratio * x_bot_ratio

        nom_ymax_xmax = denom_ymax_xmax
        nom_ymax_xmin = denom_ymax_xmin
        nom_ymin_xmax = denom_ymin_xmax
        nom_ymin_xmin = denom_ymin_xmin
        cum_op2_time += time.time() - start_op2_time
        
        start_op1_time = time.time()
        for i in range(len(x)):
            denom[y_max[i], x_max[i]] += 1
            denom[y_max[i], x_min[i]] += 1
            denom[y_min[i], x_max[i]] += 1
            denom[y_min[i], x_min[i]] += 1

            nom[y_max[i], x_max[i]] += nom_ymax_xmax[i]
            nom[y_max[i], x_min[i]] += nom_ymax_xmin[i]
            nom[y_min[i], x_max[i]] += nom_ymin_xmax[i]
            nom[y_min[i], x_min[i]] += nom_ymin_xmin[i]
        cum_op1_time += time.time() - start_op1_time

        


        start_op3_time = time.time()
        x_ = np.clip(x_, 0, W-1)
        y_ = np.clip(y_, 0, H-1)
        
        loss = np.square(np.divide(nom, denom + 0.00001))

        
        loss_sum = np.sum(loss)
        denom_sum = np.sum(denom)
        nom_sum = np.sum(nom)

        cum_op3_time += time.time() - start_op3_time
        deblur_event_img = vis_events((x_, y_), (H, W))
        deblur_sum = np.sum(deblur_event_img)



        nom = (nom * 1000).astype(np.uint8)
        denom = (denom * 1000).astype(np.uint8)
        loss = (loss * 1000).astype(np.uint8)


        # deblur_event_img = cv2.cvtColor(deblur_event_img, cv2.COLOR_GRAY2RGB)
        nom = cv2.cvtColor(nom, cv2.COLOR_GRAY2RGB)
        denom = cv2.cvtColor(denom, cv2.COLOR_GRAY2RGB)
        loss = cv2.cvtColor(loss, cv2.COLOR_GRAY2RGB)

        cv2.putText(nom, str(nom_sum), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(denom, str(denom_sum), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(loss, str(loss_sum), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(deblur_event_img, str(deblur_sum), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)


        image1 = cv2.hconcat([deblur_event_img, loss])
        image2 = cv2.hconcat([nom, denom])
        image = cv2.vconcat([image1, image2])

        if output:
            cv2.imwrite("image_{:05}.png".format(idx), image)
        # loss = np.square(np.divide(nom, denom + 0.00001))
    cum_total_time = time.time() - start_total_time
    print(cum_total_time, cum_op1_time, cum_op2_time, cum_op3_time, cum_op4_time)

    return np.sum(loss)




def flow_loss(events, flow, img_size, idx=0):

    x, y, p, t = events
    t = (t - t[0]) / (t[-1] - t[0])
    n = p == False

    forward_p_loss = ev_pol_loss((x[p], y[p], t[p]), flow, img_size, idx, False)
    forward_n_loss = ev_pol_loss((x[n], y[n], t[n]), flow, img_size, idx, False)

    backward_p_loss = ev_pol_loss((x[p], y[p], 1-t[p]), -flow, img_size, idx, True)
    backward_n_loss = ev_pol_loss((x[n], y[n], 1-t[n]), -flow, img_size, idx, False)

    # print(forward_p_loss, forward_n_loss, backward_p_loss, backward_n_loss)
    return forward_p_loss, forward_n_loss, backward_p_loss, backward_n_loss




f = h5py.File('office.h5', 'r')
H, W = f['images/image000000000'].attrs['size']

xs = f['events/xs']
ys = f['events/ys']
ts = f['events/ts']
ps = f['events/ps']



print("Begin time: ", ts[0], "End time: ", ts[-1])

sum = 0
start_t = 0
for idx, (i_, f_) in enumerate(zip(f['images'], f['flow'])):
    # print(idx)

    if idx < 20:
        flow = f['flow/' + f_].value
        start_t = f['images/' + i_].attrs['timestamp']
        continue

    # flow = f['flow/' + f_].value
    end_t = f['images/' + i_].attrs['timestamp']
    mask = (ts[()]<end_t) & (ts[()]>=start_t)
    x, y, p, t = xs[mask], ys[mask], ps[mask], ts[mask]

    if len(x) == 0:
        flow = f['flow/' + f_].value
        continue


    l1, l2, l3, l4 = flow_loss((x,y,p,t), flow, (H, W), idx)
    print(idx, l1, l2, l3, l4)
    raise

    # x = x[:len(x)//4]
    # y = y[:len(y)//4]
    # t = t[:len(t)//4]

    # t = (t - t[0]) / (t[-1] - t[0])
    # flow *= 0.0245
    # x_ = x - t * flow[0, y, x]
    # y_ = y - t * flow[1, y, x]
    # x_ = np.clip(x_, 0, W-1)
    # y_ = np.clip(y_, 0, H-1)


    # flow_rgb = vis_flow(flow)
    # event_img = vis_events((x, y), (H, W))
    # deblur_event_img = vis_events((x_, y_), (H, W))
    # image = cv2.hconcat([event_img, flow_rgb, deblur_event_img])

    # raw_count = np.count_nonzero(event_img)
    # deblur_count = np.count_nonzero(deblur_event_img)
    # print("event_raw: ", raw_count, "  event_deblur: ", deblur_count, " diff:", raw_count-deblur_count)

    # cv2.imwrite('flow{:05}.png'.format(idx), flow_rgb)
    # cv2.imwrite('image_new_half{:05}.png'.format(idx), deblur_event_img)
    flow = f['flow/' + f_].value
    sum += len(x)
    start_t = end_t



mask = ts[()]>=end_t
x = xs[mask]
y = ys[mask]
print(len(x), len(y))
sum += len(x)

print(sum)
print(len(xs))




