from argparse import ArgumentParser
import os
import cv2
import numpy as np

import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer, loggers
from torchsummary import summary
import torch.nn.functional as F

from autoencoder import Autoencoder
from harbour_datamodule import list_frames_in_dir

def play_thermal(view, crop, set, encoder=None, decoder=None, norm=True, save=True, n_channels=1):
    frames = list_frames_in_dir(os.path.join('data/',view,crop,set), 'png')
    if frames is None:
        return

    if save:
        output_dir = 'output'
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        view_dir = os.path.join(output_dir,view)
        if not os.path.exists(view_dir):
            os.mkdir(view_dir)

        set_dir = os.path.join(view_dir,set)
        if not os.path.exists(set_dir):
            os.mkdir(set_dir)

    input, rec, file, loss = [], [], [], []
    print(len(frames))
    for i, path in enumerate(frames):
        img = cv2.imread(path)
        input.append(img)
        if n_channels == 3:
            intensity, dx, dy = cv2.split(img)

        if model:
            img = img.transpose((2, 0, 1))
            img = img / 255.0
            if norm:
                img[0] = (img[0] - 0.5)/0.5
                img[1] = (img[1] - 0.5)/0.5
                img[2] = (img[2] - 0.5)/0.5
            img = torch.from_numpy(img)
            img = img.float()
            z = encoder(img.unsqueeze(0))
            rec = decoder(z)[0]
            loss = F.mse_loss(rec, img)
            if norm:
                rec[0] = rec[0] * 0.5 - 0.5
                rec[1] = rec[1] * 0.5 - 0.5
                rec[2] = rec[2] * 0.5 - 0.5
            #rec = rec.mul(255).permute(1, 2, 0).byte().numpy()
            input.append(img)
            intensity_, dx_, dy_ = cv2.split(rec)
            vis_org = np.concatenate((intensity, dx, dy), axis=1)
            vis_reg = np.concatenate((intensity_, dx_, dy_), axis=1)
            vis = np.concatenate((vis_org, vis_reg), axis=0)
        else:
            vis = np.concatenate((intensity, dx, dy), axis=1)
        vis = cv2.putText(vis, str(i).zfill(4), (5,10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, 255, 1, cv2.LINE_AA)
        vis = cv2.resize(vis, (vis.shape[1]*3,vis.shape[0]*3), interpolation = cv2.INTER_AREA)
        cv2.imshow(set,vis)
        if save:
            cv2.imwrite("output/{}.png".format(str(i).zfill(5)),vis)
        key = cv2.waitKey(0)
        if key == 27:
            break

def test(hparams, path):
    #model = Autoencoder.load_from_checkpoint(path)

    #model = Autoencoder(hparams)
    encoder = torch.load(torch.load('encoder.pt'))
    decoder = torch.load(torch.load('decoder.pt'))

    model.eval()

    #play(data_root = hparams.data_root, set = 'test_val', folder = 'aligned_out_item00_image00', model = model)
    play_thermal(view = 'view1_nc1', crop = 'crop0', set = 'test',
                 encoder=encoder, decoder=decoder, n_channels=hparams.nc)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_root", type=str, default="data/view1_nc1/crop0", help="View root directory")
    parser.add_argument("--log_dir", type=str, default="logs", help="Logging directory")
    parser.add_argument("--num_workers", type=int, default=4, help="num_workers > 0 turns on multi-process data loading")
    parser.add_argument("--image_size", type=int, default=64, help="Spatial size of training images")
    parser.add_argument("--max_epochs", type=int, default=10, help="Number of maximum training epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size during training")
    parser.add_argument("--nc", type=int, default=1, help="Number of channels in the training images")
    parser.add_argument("--norm", type=int, default=0, help="Normalize or not")
    parser.add_argument("--nz", type=int, default=8, help="Size of latent vector z")
    parser.add_argument("--nfe", type=int, default=32, help="Size of feature maps in encoder")
    parser.add_argument("--nfd", type=int, default=32, help="Size of feature maps in decoder")
    parser.add_argument("--lr", type=float, default=0.0002, help="Learning rate for optimizer")
    parser.add_argument("--beta1", type=float, default=0.9, help="Beta1 hyperparameter for Adam optimizer")
    parser.add_argument("--beta2", type=float, default=0.999, help="Beta2 hyperparameter for Adam optimizer")
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs. Use 0 for CPU mode")

    args = parser.parse_args()
    model_path = 'logs/dadata/view1_nc1/crop0_is64_nc1/version_6/checkpoints/epoch=6.ckpt'

    test(args,model_path)
    #play(view = 'view1', crop = 'crop0', set = 'test')
