import os
import os.path as osp
from easydict import EasyDict as edict

cfg = edict()


cfg.DATA = edict()
cfg.MODEL = edict()
cfg.TRAIN = edict()

# Data 

cfg.DATA.DATA_PATH
cfg.DATA.JSON_PATH

cfg.DATA.CLASSES = ["car", "motorcycle", "bus", "bicycle", "truck", "pedestrian", "other_vehicle", "animal", "emergency_vehicle"]
cfg.DATA.AVG_CAT_HEIGHTS = {'animal':0.51,'bicycle':1.44,'bus':3.44,'car':1.72,'emergency_vehicle':2.39,'motorcycle':1.59,'other_vehicle':3.23,'pedestrian':1.78,'truck':3.44}

cfg.DATA.IND_TO_NAME = {i+1:cfg.DATA.CLASSES[i] for i in range(len(cfg.DATA.CLASSES))}


cfg.DATA.ARTIFACTS_FOLDER = './artifacts/'

cfg.DATA.TRAIN_DATA_FOLDER = osp.join(cfg.DATA.ARTIFACTS_FOLDER,'bev_train_data')
cfg.DATA.VAL_DATA_FOLDER = osp.join(cfg.DATA.ARTIFACTS_FOLDER,'bev_val_data')

cfg.DATA.VOXEL_SIZE = (0.4,0.4,1.5)
cfg.DATA.Z_OFFSET = -2.0
cfg.DATA.BEV_SHAPE = (336,336,3)
cfg.DATA.NUM_SWEEPS = 10
cfg.DATA.MIN_DISTANCE = 1.0

cfg.DATA.BOX_SCALE = 0.8
cfg.DATA.NUM_WORKERS = os.cpu_count()

# Train

cfg.TRAIN.BATCH_SIZE = 8
cfg.TRAIN.NUM_EPOCHS = 15
