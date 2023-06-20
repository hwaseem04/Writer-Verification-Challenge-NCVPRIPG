import torch
from torch.utils.data import DataLoader, Dataset, Subset
import cv2
import numpy as np
import pandas as pd
from torchvision import transforms
import os
from config import Config
from empatches import EMPatches


class Read_patch(Dataset):
    """
    After creating patches and storing it in two directories,
    it is then read using this class
    """
    def __init__(self, base_dir, patch_size):
        self.base_dir = base_dir
        self.path1 = os.path.join(base_dir, 'data_1')
        self.path2 = os.path.join(base_dir, 'data_2')
        self.pair = os.listdir(self.path1)
        self.patch_size = patch_size
        self.emp = EMPatches()

    def __getitem__(self, idx):
        img_name = self.pair[idx]
        label, img1, img2 = self.read_pair(img_name)

        p = self.patch_size
        patch_img1 = self.get_patches(img1).reshape(-1, p*p)
        patch_img2 = self.get_patches(img2).reshape(-1, p*p)
        idx = len(patch_img1)
        concatenated_patches = np.concatenate([patch_img1, patch_img2], axis=0)

        return concatenated_patches, label, idx
    
    def __len__(self):
        return len(self.pair)
    
    def get_patches(self, img):
        p = self.patch_size
        prow = (img.shape[0]//p + 1) * p
        pcol = (img.shape[1]//p + 1) * p 
        drow = prow - img.shape[0]
        dcol = pcol - img.shape[1]
        pimg = np.zeros((prow,pcol))
    
        up_margin = drow // 2
        bot_margin = drow - drow//2
        l_margin = dcol // 2
        r_margin = dcol - dcol//2
        
        pimg[up_margin:pimg.shape[0]-bot_margin, l_margin:pimg.shape[1]-r_margin] = img
        img_patches, indices = self.emp.extract_patches(pimg, patchsize=(p), overlap=0)

        return np.array(img_patches)

    def read_pair(self, name):
        """
        Read data
        """
        img1 = cv2.imread(os.path.join(self.path1, name), 0)
        img2 = cv2.imread(os.path.join(self.path2, name), 0)
        label = int(name.split('.')[0][-1])

        # img1 = torch.from_numpy(img1).to(torch.float32)
        # img2 = torch.from_numpy(img2).to(torch.float32)
        return label, img1, img2


def custom_collate(List):
    input_list = []
    label_list = []
    mask_tkn = []
    max_length = max([ele[0].shape[0] for ele in List])
    index = []
    pad_idx = []
    for input, label, idx in List:
        
        mask = np.concatenate([np.ones(input.shape[0]), np.zeros(max_length - len(input))])
        ele = np.concatenate([input, np.zeros((max_length - len(input), input.shape[1]))])
        
        mask_tkn.append(mask)
        input_list.append(ele)
        label_list.append(label)
        index.append(idx)
        pad_idx.append(max_length - len(input))

    input_list = np.array(input_list)
    mask_tkn = np.array(mask_tkn)

    
    return torch.tensor(mask_tkn, dtype=torch.float32), torch.tensor(input_list, dtype=torch.float32), torch.tensor(label_list, dtype=torch.float32), (index, pad_idx)

"""

"""

def create_dataloader():
    """
    Create 2 dataloader
    - For train
    - For validation

    - Testing will be done seperately on entire image
    """

    cfg = Config().parse()
    dataset = Read_patch(cfg.data_path, cfg.patch_size)

    index = torch.randperm(len(dataset))
    train_size = cfg.train_size
    len_train = int(len(dataset) * train_size)

    train_dataset = Subset(dataset, index[:len_train])
    valid_dataset = Subset(dataset, index[len_train:])

    train_loader = DataLoader(train_dataset, collate_fn=custom_collate, batch_size=cfg.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, collate_fn=custom_collate, batch_size=cfg.batch_size)

    return train_loader, valid_loader


# train_load, _ = create_dataloader()
# for mask, x, y in train_load:
#     print(mask)
#     break


    



