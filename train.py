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
from harbour_datamodule import HarbourDataModule

def train(hparams, dm):
    logger = loggers.TensorBoardLogger(hparams.log_dir, name=f"da{hparams.data_root}_is{hparams.image_size}_nc{hparams.nc}")
    model = Autoencoder(hparams)
    # print detailed summary with estimated network size
    summary(model, (hparams.nc, hparams.image_size, hparams.image_size), device="cpu")
    trainer = Trainer(logger=logger, gpus=hparams.gpus, max_epochs=hparams.max_epochs)
    trainer.fit(model, dm)
    #trainer.test(model)
    torch.save(model.encoder, "encoder.pt")
    torch.save(model.decoder, "decoder.pt")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_root", type=str, default="data/view1_normal/crop0/train", help="View root directory")
    parser.add_argument("--log_dir", type=str, default="logs", help="Logging directory")
    parser.add_argument("--num_workers", type=int, default=12, help="num_workers > 0 turns on multi-process data loading")
    parser.add_argument("--image_size", type=int, default=64, help="Spatial size of training images")
    parser.add_argument("--max_epochs", type=int, default=20, help="Number of maximum training epochs")
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

    dm = HarbourDataModule(data_dir=args.data_root, batch_size=args.batch_size)
    dm.setup()

    train(args, dm)
