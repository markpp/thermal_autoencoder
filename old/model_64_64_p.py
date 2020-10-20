import torch
import torch.nn as nn

def create_encoder(hparams, train=True):
    return nn.Sequential(
        # input (nc) x 64 x 64
        nn.Conv2d(hparams['nc'], hparams['nfe'], 4, 2, 1, bias=False),
        nn.BatchNorm2d(hparams['nfe']),
        nn.LeakyReLU(True),
        # input (nfe) x 32 x 32
        nn.Conv2d(hparams['nfe'], hparams['nfe'] * 2, 4, 2, 1, bias=False),
        nn.BatchNorm2d(hparams['nfe'] * 2),
        nn.LeakyReLU(True),
        # input (nfe*2) x 16 x 16
        nn.Conv2d(hparams['nfe'] * 2, hparams['nfe'] * 4, 4, 2, 1, bias=False),
        nn.BatchNorm2d(hparams['nfe'] * 4),
        nn.LeakyReLU(True),
        # input (nfe*4) x 8 x 8
        nn.Conv2d(hparams['nfe'] * 4, hparams['nfe'] * 8, 4, 2, 1, bias=False),
        nn.BatchNorm2d(hparams['nfe'] * 8),
        nn.LeakyReLU(True),
        # input (nfe*8) x 4 x 4
        nn.Conv2d(hparams['nfe'] * 8, hparams['nz'], 4, 1, 0, bias=False),
        nn.BatchNorm2d(hparams['nz']),
        nn.LeakyReLU(True)
        # output (nz) x 1 x 1
    )

def create_decoder(hparams):
    return nn.Sequential(
        # input (nz) x 1 x 1
        nn.ConvTranspose2d(hparams['nz'], hparams['nz'] * 8, 4, 1, 0, bias=False),
        nn.BatchNorm2d(hparams['nz'] * 8),
        nn.ReLU(True),
        # input (nfd*8) x 4 x 4
        nn.ConvTranspose2d(hparams['nz'] * 8, hparams['nz'] * 4, 4, 2, 1, bias=False),
        nn.BatchNorm2d(hparams['nz'] * 4),
        nn.ReLU(True),
        # input (nfd*4) x 8 x 8
        nn.ConvTranspose2d(hparams['nz'] * 4, hparams['nz'] * 2, 4, 2, 1, bias=False),
        nn.BatchNorm2d(hparams['nz'] * 2),
        nn.ReLU(True),
        # input (nfd*2) x 16 x 16
        nn.ConvTranspose2d(hparams['nz'] * 2, hparams['nz'], 4, 2, 1, bias=False),
        nn.BatchNorm2d(hparams['nz']),
        nn.ReLU(True),
        # input (nfd) x 32 x 32
        nn.ConvTranspose2d(hparams['nz'], hparams['nc'], 4, 2, 1, bias=False),
        nn.Tanh()
        # output (nc) x 64 x 64
    )
