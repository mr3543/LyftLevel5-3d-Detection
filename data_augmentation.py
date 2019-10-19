from datetime import datetime
from functools import partial
import glob
from multiprocessing import Pool

import os
import sys
import pandas as pd
import cv2
import pickle
from PIL import Image
import numpy as np
import random
from tqdm import tqdm
import scipy
import scipy.ndimage
import scipy.special
from scipy.spatial.transform import Rotation as R
from shapely.geometry import Polygon


from lyft_dataset_sdk.lyftdataset import LyftDataset
from lyft_dataset_sdk.utils.data_classes import LidarPointCloud, Box, Quaternion
from lyft_dataset_sdk.utils.geometry_utils import view_points, transform_matrix, points_in_box
from box_utils import scale_boxes,move_boxes_to_car_space
from config import cfg


def make_box_db(train_df,l5d):
    
    first_samples = train_df.first_sample_token.values
    box_db = {}
    box_scale = cfg.DATA.BOX_SCALE
    num_sweeps = cfg.DATA.NUM_SWEEPS
    min_distance = cfg.DATA.MIN_DISTANCE

    for sample_token in tqdm(first_samples):
        
        while sample_token:

            sample = l5d.get('sample',sample_token)
            sample_lidar_token = sample['data']['LIDAR_TOP']
            lidar_data = l5d.get('sample_data',sample_lidar_token)
            
            ego_pose = l5d.get("ego_pose", lidar_data["ego_pose_token"])
            calibrated_sensor = l5d.get("calibrated_sensor", lidar_data["calibrated_sensor_token"])
            car_from_sensor = transform_matrix(calibrated_sensor['translation'], Quaternion(calibrated_sensor['rotation']),
                                                inverse=False)

            try:
                lidar_pointcloud = LidarPointCloud.from_file_multisweep(l5d,sample,'LIDAR_TOP','LIDAR_TOP',num_sweeps=num_sweeps,min_distance=min_distance)[0]
                lidar_pointcloud.transform(car_from_sensor)
            except Exception as e:
                print ("Failed to load Lidar Pointcloud for {}: {}:".format(sample_token, e))
                sample_token = sample["next"]
                continue


            boxes = l5d.get_boxes(sample_lidar_token)
            move_boxes_to_car_space(boxes, ego_pose)
            scale_boxes(boxes, box_scale)
            for box in boxes:
                point_mask = points_in_box(box,lidar_pointcloud.points[:3,:])
                box_points = lidar_pointcloud.points[:,point_mask]
                if box.name not in box_db:
                    box_db[box.name] = [{'lidar':box_points,'box':box}]
                else:
                    box_db[box.name].append({'lidar':box_points,'box':box})

            sample_token = sample['next']
        
    pickle.dump(box_db,open(cfg.DATA.BOX_DB_FILE,'wb'))


def augment_pc(pc,boxes,box_db):

    classes = cfg.DATA.CLASSES
    sample_dict = cfg.DATA.RANDOM_SAMPLES
    for c in classes:
        if not sample_dict[c] or not box_db[c]: continue
        to_sample = sample_dict[c]
        sample_inds = random.sample(range(0,len(box_db[c])),to_sample)
        samples_to_add = [box_db[c][i] for i in sample_inds]

        for sample in samples_to_add:

            sample_box = sample['box']
            # we don't add the sample_box to the lidar cloud if it intersects with another 
            # object in the cloud
            # we use xy intersection of the bottom corners of the 3d box to check for overlap
            # this method is not perfect and could be refactored - however it is unlikely that
            # two objects would intersect on xy yet be completely separated - objects would need to stacked
            # on top of each other
            sample_polygon = Polygon([sample_box.bottom_corners[:2,:]])
            if not any([sample_polygon.intersection(Polygon(box.bottom_corners[:2,:])).area for box in boxes]):
                # apply transformations to box and points, then add to lidar
                pc.append(sample['lidar'])
                boxes.append(sample_box)
    
    for box in boxes:
        # apply random rotation and translation to the 
        # ground truth boxes and their points
        random_translation = .25 * np.random.randn(3)
        box.center += random_translation

        points_mask = points_in_box(box,pc[:3,:])
        box_points = pc[:,points_mask]
        box_t = np.transpose(box_points[:3,:])
        box_t += random_translation

        random_angle = np.random.uniform(-np.pi,np.pi)
        quat = Quaternion(axis=[0,0,1],angle=random_angle)
        box.orientation *= quat
        rot_matrix = quat.rotation_matrix
        box_points = np.dot(rot_matrix,box_points)

    return pc,boxes

if __name__ == '__main__':

    data_path = cfg.DATA.DATA_PATH
    json_path = cfg.DATA.TRAIN_JSON_PATH

    l5d = LyftDataset(data_path= data_path,json_path = json_path,verbose = True)

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
    
    make_box_db(train_df,l5d)
