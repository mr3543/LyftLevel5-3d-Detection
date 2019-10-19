from datetime import datetime
from functools import partial
import glob
from multiprocessing import Pool

# Disable multiprocesing for numpy/opencv. We already multiprocess ourselves, this would mean every subprocess produces
# even more threads which would lead to a lot of context switching, slowing things down a lot.
import os
os.environ["OMP_NUM_THREADS"] = "1"

import sys
import gc
import pandas as pd
import cv2
from PIL import Image
import numpy as np
from tqdm import tqdm
import scipy
import scipy.ndimage
import scipy.special
import pickle
from scipy.spatial.transform import Rotation as R

from lyft_dataset_sdk.lyftdataset import LyftDataset
from lyft_dataset_sdk.utils.data_classes import LidarPointCloud, Box, Quaternion
from lyft_dataset_sdk.utils.geometry_utils import view_points, transform_matrix
from data_augmentation import augment_pc

from config import cfg

def prepare_training_data_for_scene(first_sample_token, output_folder, bev_shape, voxel_size, z_offset, 
                                    box_scale,num_sweeps,min_distance,classes,level5data,box_db,augment=True):
    """
    Given a first sample token (in a scene), output rasterized input volumes and targets in birds-eye-view perspective.

    """
    sample_token = first_sample_token
   
    while sample_token:
        
        sample = level5data.get("sample", sample_token)

        sample_lidar_token = sample["data"]["LIDAR_TOP"]
        lidar_data = level5data.get("sample_data", sample_lidar_token)

        ego_pose = level5data.get("ego_pose", lidar_data["ego_pose_token"])
        calibrated_sensor = level5data.get("calibrated_sensor", lidar_data["calibrated_sensor_token"])

        car_from_sensor = transform_matrix(calibrated_sensor['translation'], Quaternion(calibrated_sensor['rotation']),
                                            inverse=False)

        try:
            lidar_pointcloud = LidarPointCloud.from_file_multisweep(level5data,sample,'LIDAR_TOP','LIDAR_TOP',num_sweeps=num_sweeps,min_distance=min_distance)[0]
            lidar_pointcloud.transform(car_from_sensor)
        except Exception as e:
            print ("Failed to load Lidar Pointcloud for {}: {}:".format(sample_token, e))
            sample_token = sample["next"]
            continue
        
        lidar_points = lidar_pointcloud.points
        boxes = level5data.get_boxes(sample_lidar_token)
        #augment data here
        if augment:
            lidar_points,boxes = augment_pc(lidar_points,boxes,box_db)


        bev = create_voxel_pointcloud(lidar_points, bev_shape, voxel_size=voxel_size, z_offset=z_offset)
        bev = normalize_voxel_intensities(bev)

        target = np.zeros_like(bev)

        move_boxes_to_car_space(boxes, ego_pose)
        scale_boxes(boxes, box_scale)
        draw_boxes(target, voxel_size, boxes=boxes, classes=classes, z_offset=z_offset)

        bev_im = np.round(bev*255).astype(np.uint8)
        target_im = target[:,:,0] # take one channel only

        cv2.imwrite(os.path.join(output_folder, "{}_input.png".format(sample_token)), bev_im)
        cv2.imwrite(os.path.join(output_folder, "{}_target.png".format(sample_token)), target_im)
        
        sample_token = sample["next"]


if __name__ == '__main__':

    data_path = cfg.DATA.DATA_PATH
    json_path = cfg.DATA.TRAIN_JSON_PATH

    l5d = LyftDataset(data_path= data_path,json_path = json_path,verbose = True)
   
    os.makedirs(cfg.DATA.ARTIFACTS_FOLDER,exist_ok=True)

    records = [(l5d.get('sample', record['first_sample_token'])['timestamp'], record) for record in l5d.scene]

    entries = []

    for start_time, record in sorted(records):
        start_time = l5d.get('sample', record['first_sample_token'])['timestamp'] / 1000000

        token = record['token']
        name = record['name']
        date = datetime.utcfromtimestamp(start_time)
        host = "-".join(record['name'].split("-")[:2])
        first_sample_token = record["first_sample_token"]

        entries.append((host, name, date, token, first_sample_token))
                
    df = pd.DataFrame(entries, columns=["host", "scene_name", "date", "scene_token", "first_sample_token"])
    validation_hosts = ["host-a007", "host-a008", "host-a009"]

    validation_df = df[df["host"].isin(validation_hosts)]
    vi = validation_df.index
    train_df = df[~df.index.isin(vi)]

    train_data_folder = cfg.DATA.TRAIN_DATA_FOLDER
    validation_data_folder = cfg.DATA.VAL_DATA_FOLDER
    num_workers = cfg.DATA.NUM_WORKERS

    voxel_size = cfg.DATA.VOXEL_SIZE
    z_offset = cfg.DATA.Z_OFFSET
    bev_shape = cfg.DATA.BEV_SHAPE
    box_scale = cfg.DATA.BOX_SCALE
    num_sweeps = cfg.DATA.NUM_SWEEPS
    min_distance = cfg.DATA.MIN_DISTANCE
    classes = cfg.DATA.CLASSES
    augment = True
    box_db = pickle.load(open(cfg.DATA.BOX_DB_FILE,'rb'))
    print('PICKLE LOADED')
    sys.exit(1)

    for df, data_folder in [(train_df, train_data_folder), (validation_df, validation_data_folder)]:
        print("Preparing data into {} using {} workers".format(data_folder, num_workers))
        first_samples = df.first_sample_token.values

        os.makedirs(data_folder, exist_ok=True)
    
        process_func = partial(prepare_training_data_for_scene,
                           output_folder=data_folder, bev_shape=bev_shape, voxel_size=voxel_size, 
                           z_offset=z_offset, box_scale=box_scale,
                           num_sweeps=num_sweeps,min_distance=min_distance,
                           classes=classes,level5data=l5d,augment=augment)

        pool = Pool(num_workers)
        for _ in tqdm(pool.imap_unordered(process_func, first_samples), total=len(first_samples)):
            pass
        
        pool.close()
        del pool
        augment = False
        del box_db
        gc.collect()
