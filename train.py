from pathlib import Path
import imageio
from imageio import imwrite
from skimage.color import hsv2rgb
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.tensorboard import SummaryWriter

import sys
import numpy as np




def main():
    print("Training Started")

    device = torch.device('cuda:0') if torch.cuda.is_available() else torchl.device('cpu')

    cur_path = Path(__file__).parent.resolve()
    module_name = cur_path.name
    sys.path.append(str(cur_path.parent))
    OpticalFlow = __import__(module_name).OpticalFlow

    data_base = cur_path/'data'/'events'
    out_path = cur_path/'res'
    out_path.mkdir(parents=True, exist_ok=True)

    events = np.load(str(data_base/'outdoor.npy'))
    
    
    # number of frames per second
    fps = 5
    # height and width of images
    imsize = 480, 640
    # window size in microseconds
    dt = 1. / fps

    of = OpticalFlow(imsize)

    x, y, t, p = events

    start_t = t[0]
    stop_t = t[-1]
    frame_ts = np.arange(start_t, stop_t, dt)
    frame_ts = np.append(frame_ts, [frame_ts[-1] + dt])
    num_frames = len(frame_ts) - 1

    # TODO: change the learning rate
    optimizer = torch.optim.Adam(of._net.parameters(), lr=3e-4)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.9)
    # loss_fun = TotalLoss(args.smoothness_weight, device=device)

    iteration = 0
    size = 0
    epoch_loss = 0.0
    running_loss = 0.0
    of._net.train()




if __name__ == "__main__":

    main()