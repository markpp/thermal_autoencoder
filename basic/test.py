from argparse import ArgumentParser
import os
import cv2
import numpy as np
from glob import glob

import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer, loggers
from torchsummary import summary
import torch.nn.functional as F

from autoencoder import Autoencoder
from views import views
from plotter import Plotter


def play_thermal(frame_list, hparams, output_dir, encoder=None, decoder=None, norm=True, show=True, save=True, n_channels=1):

    if save or show:
        img = cv2.imread(frame_list[0])[:,:,0]
        img_w, img_h = img.shape[:2]
        scale = 3
        # Create a plotter class object
        plot = Plotter(img_h*scale*2, img_w*scale*2)

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    inputs, recs, files, losses, latent = [], [], [], [], []
    print(len(frame_list))
    for i, frame_path in enumerate(frame_list):
        crop = cv2.imread(frame_path, -1)#[:,:,0] #[:,:,:1]
        #crop = cv2.resize(crop, (64,192), interpolation=cv2.INTER_LINEAR)

        inputs.append(crop)
        files.append(frame_path)
        img = crop / 255.0
        img = torch.from_numpy(img)
        img = img.float()
        x = img.unsqueeze(0).unsqueeze(0)
        if encoder is not None:
            z = encoder(x)
        if decoder is not None:
            rec = decoder(z)[0][0]
        loss = F.mse_loss(rec, img)
        rec = rec.mul(255).byte().numpy()
        recs.append(rec)
        losses.append(loss.item())
        latent.append(z[0].detach().numpy().flatten())
        if save or show:
            vis = np.concatenate((inputs[-1], recs[-1]), axis=0)
            vis = cv2.resize(vis, (vis.shape[1]*scale,vis.shape[0]*scale), interpolation = cv2.INTER_NEAREST)
            plot.plot(loss*100000)
            output = np.concatenate((cv2.merge((vis,vis,vis)), plot.plot_canvas), axis=1)
            output = cv2.putText(output, str(i).zfill(4), (4,16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1, cv2.LINE_AA)
            if save:
                l = "{:.6f}".format(losses[-1])[-6:]
                cv2.imwrite(os.path.join(output_dir,"{}_l-{}.png".format(str(i).zfill(5),l)),vis)
            if show:
                cv2.imshow("test",output)
                key = cv2.waitKey()
                if key == 27:
                    break

    np.save(os.path.join(output_dir,'inputs.npy'), inputs)
    np.save(os.path.join(output_dir,'recs.npy'), recs)
    np.save(os.path.join(output_dir,'files.npy'), files)
    np.save(os.path.join(output_dir,'losses.npy'), losses)
    np.save(os.path.join(output_dir,'latent.npy'), latent)

def test(hparams):

    model = Autoencoder(hparams)

    model.encoder = torch.load("trained_models/train_all/encoder.pt")
    model.decoder = torch.load("trained_models/train_all/decoder.pt")

    #print(model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    #from torchsummary import summary
    #summary(model, (1, 64, 192))

    model.encoder.eval()
    model.decoder.eval()

    output_dir = "output/{}".format(os.path.basename(hparams.image_list).split('.')[0])

    with open(hparams.image_list) as f:
        image_files = f.read().splitlines()
        play_thermal(image_files, hparams, output_dir, encoder=model.encoder.to('cpu'), decoder=model.decoder.to('cpu'), norm=hparams.norm, n_channels=hparams.nc, show=False, save=False)
    if not len(image_files)>0:
        print("did not find any files")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--image_list", type=str, default="output/test_normal.txt", help="View root directory")
    parser.add_argument("--nc", type=int, default=1, help="Number of channels in the training images")
    parser.add_argument("--norm", type=int, default=0, help="Normalize or not")
    parser.add_argument("--nz", type=int, default=8, help="Size of latent vector z")
    parser.add_argument("--nfe", type=int, default=32, help="Size of feature maps in encoder")
    parser.add_argument("--nfd", type=int, default=32, help="Size of feature maps in decoder")
    parser.add_argument("--lr", type=float, default=0.0002, help="Learning rate for optimizer")
    parser.add_argument("--beta1", type=float, default=0.9, help="Beta1 hyperparameter for Adam optimizer")
    parser.add_argument("--beta2", type=float, default=0.999, help="Beta2 hyperparameter for Adam optimizer")

    args = parser.parse_args()

    test(args)
