import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import glob
import numpy as np
import os

from lyft_dataset_sdk.lyftdataset import LyftDataset
from dataset import BEVImageDataset
from model import get_unet_model
from config import cfg
from tqdm import tqdm
from evaluate import evaluate_map

data_path = cfg.DATA.DATA_PATH
json_path = cfg.DATA.TRAIN_JSON_PATH
l5d = LyftDataset(data_path = data_path,json_path=json_path,verbose=True)

train_data_folder = cfg.DATA.TRAIN_DATA_FOLDER

input_filepaths = sorted(glob.glob(os.path.join(train_data_folder, "*_input.png")))
target_filepaths = sorted(glob.glob(os.path.join(train_data_folder, "*_target.png")))

train_dataset = BEVImageDataset(input_filepaths, target_filepaths)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

classes = cfg.DATA.CLASSES

class_weights = torch.from_numpy(np.array([0.2] + [1.0]*len(classes), dtype=np.float32))
class_weights = class_weights.to(device)

batch_size = cfg.TRAIN.BATCH_SIZE
epochs = cfg.TRAIN.NUM_EPOCHS
bev_shape = cfg.DATA.BEV_SHAPE

model = get_unet_model(num_output_classes = len(cfg.DATA.CLASSES) + 1,in_channels=bev_shape[-1])
model = model.to(device)

optim = torch.optim.Adam(model.parameters(), lr=1e-3)
dataloader = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=True, num_workers=os.cpu_count()*2)

all_losses = []

for epoch in range(1, epochs+1):
    print("Epoch", epoch)
    
    epoch_losses = []
    progress_bar = tqdm(dataloader)
    
    for ii, (X, target, sample_ids) in enumerate(progress_bar):
        X = X.to(device)  # [N, 3, H, W]
        target = target.to(device)  # [N, H, W] with class indices (0, 1)
        prediction = model(X)  # [N, 2, H, W]
        loss = F.cross_entropy(prediction, target, weight=class_weights)

        optim.zero_grad()
        loss.backward()
        optim.step()
        
        epoch_losses.append(loss.detach().cpu().numpy())
    
    print("Loss:", np.mean(epoch_losses))
    all_losses.extend(epoch_losses)
    
    checkpoint_filename = "unet_checkpoint_epoch_{}.pth".format(epoch)
    checkpoint_filepath = os.path.join(cfg.DATA.ARTIFACTS_FOLDER, checkpoint_filename)
    torch.save(model.state_dict(), checkpoint_filepath)

    if epoch % 3 == 0:

        val_map = evaluate_map(cfg.DATA.VAL_DATA_FOLDER,epoch,l5d)
        print('VALIDATION SET mAP: ',val_map)


    
