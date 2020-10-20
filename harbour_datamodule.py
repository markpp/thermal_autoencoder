import torch
import torchvision
from torch.utils.data import DataLoader, Subset
import pytorch_lightning as pl

import os
from glob import glob
import cv2


class FromFolderDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, crop_size=64):
        self.crop_size = crop_size
        self.image_files = [y for x in os.walk(data_dir) for y in glob(os.path.join(x[0], '*.png'))]
        if not len(self.image_files)>0:
            print("did not find any files")

    def load_sample(self, image_path):
        img = cv2.imread(image_path, -1)
        self.image_h, self.image_w = img.shape[:2]
        return img

    def __getitem__(self, idx):
        img = self.load_sample(self.image_files[idx])
        #img = img.transpose((2, 0, 1))
        img = img / 255.0
        img = torch.from_numpy(img)
        img = img.float()
        img = torch.unsqueeze(img, 0)
        return img

    def __len__(self):
        return len(self.image_files)


class HarbourDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

    #def prepare_data():
        #download, unzip here. anything that should not be done distributed
    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.data = FromFolderDataset(self.data_dir)
            n_sample = len(self.data)
            end_train_idx = int(n_sample * 0.9)
            self.data_train = Subset(self.data, range(0, end_train_idx))
            self.data_val = Subset(self.data, range(end_train_idx + 1, n_sample))

    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.data_val, batch_size=self.batch_size, shuffle=False)


if __name__ == '__main__':

    dm = HarbourDataModule(data_dir='data/view1_normal/crop0/train',
                           batch_size=16)

    dm.setup()

    # cleanup output dir
    import os, shutil
    output_root = "output/"
    if os.path.exists(output_root):
        shutil.rmtree(output_root)
    os.makedirs(output_root)

    sample_idx = 0
    for batch_id, batch in enumerate(dm.val_dataloader()):
        imgs = batch
        for img in imgs:
            #img = img.mul(255).permute(1, 2, 0).byte().numpy()
            img = img.mul(255).byte().numpy()
            output_dir = os.path.join(output_root,str(batch_id).zfill(6))
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            filename = "id-{}.png".format(str(sample_idx).zfill(6))
            cv2.imwrite(os.path.join(output_dir,filename),img)
            sample_idx = sample_idx + 1
