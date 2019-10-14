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

cfg.DATA.ARTIFACTS_FOLDER = './artifacts/'

cfg.DATA.TRAIN_DATA_FOLDER = osp.join(cfg.DATA.ARTIFACTS_FOLDER,'bev_train_data')
cfg.DATA.VAL_DATA_FOLDER = osp.join(cfg.DATA.ARTIFACTS_FOLDER,'bev_val_data')

cfg.DATA.VOXEL_SIZE = (0.4,0.4,1.5)
cfg.DATA.Z_OFFSET = -2.0
cfg.DATA.BEV_SHAPE = (336,336,3)

cfg.DATA.BOX_SCALE = 0.8
cfg.DATA.NUM_WORKERS = os.cpu_count()

# Train

cfg.TRAIN.BATCH_SIZE = 8
cfg.TRAIN.NUM_EPOCHS = 15