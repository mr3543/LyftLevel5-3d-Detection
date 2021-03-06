import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import cv2
import os
import json
import numpy as np
import glob 
from tqdm import tqdm 

from lyft_dataset_sdk.lyftdataset import LyftDataset
from lyft_dataset_sdk.utils.data_classes import LidarPointCloud, Box, Quaternion
from lyft_dataset_sdk.utils.geometry_utils import view_points, transform_matrix
from lyft_dataset_sdk.eval.detection.mAP_evaluation import Box3D, recall_precision, get_average_precisions

from dataset import BEVImageDataset,BEVTestImageDataset
from config import cfg
from model import get_unet_model

def eval_for_kaggle(l5d,test_data_folder,epoch_to_load):
    
    batch_size = cfg.TRAIN.BATCH_SIZE
    
    input_filepaths = sorted(glob.glob(test_data_folder, "*_input.png"))

    sample_tokens = [filename.split("/")[-1].replace("_input.png","") for filename in input_filepaths]

    num_inputs = len(input_filepaths)

    test_dataset = BEVTestImageDataset(input_filepaths)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size, shuffle=False, num_workers=os.cpu_count())
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_unet_model(num_output_classes=1+len(classes))
    model = model.to(device)

    checkpoint_filename = "unet_checkpoint_epoch_{}.pth".format(epoch_to_load)
    checkpoint_filepath = os.path.join(cfg.DATA.ARTIFACTS_FOLDER, checkpoint_filename)
    model.load_state_dict(torch.load(checkpoint_filepath))

    boxes = make_prediction_boxes(model,test_dataloader,num_inputs,batch_size,l5d)
    write_submission(boxes)
    

def load_groundtruth_boxes(level5data, sample_tokens):
    gt_box3ds = []

    # Load annotations and filter predictions and annotations.
    for sample_token in tqdm(sample_tokens):

        sample = level5data.get('sample', sample_token)
        sample_annotation_tokens = sample['anns']

        sample_lidar_token = sample["data"]["LIDAR_TOP"]
        lidar_data = level5data.get("sample_data", sample_lidar_token)
        ego_pose = level5data.get("ego_pose", lidar_data["ego_pose_token"])
        ego_translation = np.array(ego_pose['translation'])
        
        for sample_annotation_token in sample_annotation_tokens:
            sample_annotation = level5data.get('sample_annotation', sample_annotation_token)
            sample_annotation_translation = sample_annotation['translation']
            
            class_name = sample_annotation['category_name']
            
            box3d = Box3D(
                sample_token=sample_token,
                translation=sample_annotation_translation,
                size=sample_annotation['size'],
                rotation=sample_annotation['rotation'],
                name=class_name
            )
            gt_box3ds.append(box3d)
            
    return gt_box3ds

def write_for_map(boxes,filepath,gt=False):
    l = [{'sample_token':box.sample_token,
      'translation':box.translation,
      'size':box.size,
      'rotation':box.rotation,
      'name':box.name} for box in boxes]

    if not gt:
        for i in range(len(boxes)):
            l[i]['score'] = boxes[i].score

    with open(filepath,'w') as out:
        json.dump(l,out)
    
def write_submission(boxes):
    sub = {}
    for i in range(len(boxes)):
        yaw = 2*np.arccos(boxes[i].rotation[0])
        pred = str(boxes[i].score/255) + ' ' + \
               str(boxes[i].center_x) + ' ' + \
               str(boxes[i].center_y) + ' ' + \
               str(boxes[i].center_z) + ' ' + \
               str(boxes[i].width) + ' ' + \
               str(boxes[i].length) + ' ' + \
               str(boxes[i].height) + ' ' + \
               str(yaw) + ' ' + \
               str(boxes[i].name) + ' '

        if boxes[i].sample_token in sub.keys():
            sub[boxes[i].sample_token] += pred
        else:
            sub[boxes[i].sample_token] = pred

    sub = pd.DataFrame(list(sub.items()))
    sub.columns = ['Id','PredictionString']
    sub.to_csv('lyft3d_pred.csv',index=False)

def evaluate_map(val_data_folder,epoch_to_load,level5data):
    
    batch_size = cfg.TRAIN.BATCH_SIZE
    classes = cfg.DATA.CLASSES

    input_filepaths = sorted(glob.glob(val_data_folder, "*_input.png"))
    target_filepaths = sorted(glob.glob(val_data_folder, "*_target.png"))

    sample_tokens = [filename.split("/")[-1].replace("_input.png","") for filename in input_filepaths]

    num_inputs = len(input_filepaths)

    validation_dataset = BEVImageDataset(input_filepaths, target_filepaths)
    validation_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size, shuffle=False, num_workers=os.cpu_count())
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_unet_model(num_output_classes=1+len(classes))
    model = model.to(device)

    checkpoint_filename = "unet_checkpoint_epoch_{}.pth".format(epoch_to_load)
    checkpoint_filepath = os.path.join(cfg.DATA.ARTIFACTS_FOLDER, checkpoint_filename)
    model.load_state_dict(torch.load(checkpoint_filepath))

    boxes = make_prediction_boxes(model,validation_dataloader,num_inputs,batch_size,level5data)
    gt_box3ds = load_groundtruth_boxes(level5data, sample_tokens)
    
    map_list = cfg.DATA.MAP_LIST

    return np.mean([get_average_precisions(gt_box3ds,boxes,cfg.DATA.CLASSES,iou) for iou in map_list])


def make_prediction_boxes(model,dataloader,num_inputs,batch_size,level5data):

    classes = cfg.DATA.CLASSES
    height = cfg.DATA.BEV_SHAPE[0]
    width = cfg.DATA.BEV_SHAPE[1]
    device = cfg.TRAIN.DEVICE
    bev_shape = cfg.DATA.BEV_SHAPE
    voxel_size = cfg.DATA.VOXEL_SIZE
    z_offset = cfg.DATA.Z_OFFSET

    # we quantize to uint8 here to conserve memory. we're allocating >20GB of memory otherwise.
    predictions = np.zeros((num_inputs, 1+len(classes), height, width), dtype=np.uint8) # [N,C,H,W]

    sample_tokens = []
    progress_bar = tqdm(dataloader)

    # evaluate samples with loaded model - predictions are gathered in 'predictions'
    with torch.no_grad():
        model.eval()
        for ii, (X, target, batch_sample_tokens) in enumerate(progress_bar):

            offset = ii*batch_size
            sample_tokens.extend(batch_sample_tokens)
            
            X = X.to(device)  # [N, 1, H, W]
            prediction = model(X)  # [N, 2, H, W]
            
            prediction = F.softmax(prediction, dim=1)
            
            prediction_cpu = prediction.cpu().numpy()
            predictions[offset:offset+batch_size] = np.round(prediction_cpu*255).astype(np.uint8)
            
    predictions_non_class0 = 255 - predictions[:,0] # [N,H,W]
    background_threshold = 255//2

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    predictions_opened = np.zeros((predictions_non_class0.shape), dtype=np.uint8) # [N,H,W]

    for i, p in enumerate(tqdm(predictions_non_class0)):
        thresholded_p = (p > background_threshold).astype(np.uint8)
        predictions_opened[i] = cv2.morphologyEx(thresholded_p, cv2.MORPH_OPEN, kernel) # [H,W]

    detection_boxes = []
    detection_scores = []
    detection_classes = []

    for i in tqdm(range(len(predictions))):
        prediction_opened = predictions_opened[i] # [H,W]
        probability_non_class0 = predictions_non_class0[i] # [H,W]
        class_probability = predictions[i] # [C,H,W]

        sample_boxes = []
        sample_detection_scores = []
        sample_detection_classes = []
        
        contours, hierarchy = cv2.findContours(prediction_opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
        
        for cnt in contours:
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            
            # Let's take the center pixel value as the confidence value
            box_center_index = np.int0(np.mean(box, axis=0))
            
            for class_index in range(len(classes)):
                box_center_value = class_probability[class_index+1, box_center_index[1], box_center_index[0]]
                
                # Let's remove candidates with very low probability
                if box_center_value < 0.01:
                    continue
                
                box_center_class = classes[class_index]

                box_detection_score = box_center_value
                sample_detection_classes.append(box_center_class)
                sample_detection_scores.append(box_detection_score)
                sample_boxes.append(box)
            
        
        detection_boxes.append(np.array(sample_boxes))
        detection_scores.append(sample_detection_scores)
        detection_classes.append(sample_detection_classes)


    pred_box3ds = []
    height_dict = cfg.DATA.AVG_CAT_HEIGHT

    # This could use some refactoring..
    for (sample_token, sample_boxes, sample_detection_scores, sample_detection_class) in tqdm(zip(sample_tokens, detection_boxes, detection_scores, detection_classes), total=len(sample_tokens)):
        sample_boxes = sample_boxes.reshape(-1, 2) # (N, 4, 2) -> (N*4, 2)
        sample_boxes = sample_boxes.transpose(1,0) # (N*4, 2) -> (2, N*4)

        # Add Z dimension
        sample_boxes = np.vstack((sample_boxes, np.zeros(sample_boxes.shape[1]),)) # (2, N*4) -> (3, N*4)

        sample = level5data.get("sample", sample_token)
        sample_lidar_token = sample["data"]["LIDAR_TOP"]
        lidar_data = level5data.get("sample_data", sample_lidar_token)
        lidar_filepath = level5data.get_sample_data_path(sample_lidar_token)
        ego_pose = level5data.get("ego_pose", lidar_data["ego_pose_token"])
        ego_translation = np.array(ego_pose['translation'])

        global_from_car = transform_matrix(ego_pose['translation'],
                                           Quaternion(ego_pose['rotation']), inverse=False)

        car_from_voxel = np.linalg.inv(create_transformation_matrix_to_voxel_space(bev_shape, voxel_size, (0, 0, z_offset)))


        global_from_voxel = np.dot(global_from_car, car_from_voxel)
        sample_boxes = transform_points(sample_boxes, global_from_voxel)

        # We don't know at where the boxes are in the scene on the z-axis (up-down), let's assume all of them are at
        # the same height as the ego vehicle.
        sample_boxes[2,:] = ego_pose["translation"][2]


        # (3, N*4) -> (N, 4, 3)
        sample_boxes = sample_boxes.transpose(1,0).reshape(-1, 4, 3)

        box_height = [height_dict[name] for name in sample_detection_class]

        # Note: Each of these boxes describes the ground corners of a 3D box.
        # To get the center of the box in 3D, we'll have to add half the height to it.
        sample_boxes_centers = sample_boxes.mean(axis=1)
        sample_boxes_centers[:,2] += box_height/2

        # Width and height is arbitrary - we don't know what way the vehicles are pointing from our prediction segmentation
        # It doesn't matter for evaluation, so no need to worry about that here.
        # Note: We scaled our targets to be 0.8 the actual size, we need to adjust for that
        sample_lengths = np.linalg.norm(sample_boxes[:,0,:] - sample_boxes[:,1,:], axis=1) * 1/box_scale
        sample_widths = np.linalg.norm(sample_boxes[:,1,:] - sample_boxes[:,2,:], axis=1) * 1/box_scale
        
        sample_boxes_dimensions = np.zeros_like(sample_boxes_centers) 
        sample_boxes_dimensions[:,0] = sample_widths
        sample_boxes_dimensions[:,1] = sample_lengths
        sample_boxes_dimensions[:,2] = box_height

        for i in range(len(sample_boxes)):
            translation = sample_boxes_centers[i]
            size = sample_boxes_dimensions[i]
            class_name = sample_detection_class[i]
            ego_distance = float(np.linalg.norm(ego_translation - translation))
        
            
            # Determine the rotation of the box
            v = (sample_boxes[i,0] - sample_boxes[i,1])
            v /= np.linalg.norm(v)
            r = R.from_dcm([
                [v[0], -v[1], 0],
                [v[1],  v[0], 0],
                [   0,     0, 1],
            ])
            quat = r.as_quat()
            # XYZW -> WXYZ order of elements
            quat = quat[[3,0,1,2]]
            
            detection_score = float(sample_detection_scores[i])

            
            box3d = Box3D(
                sample_token=sample_token,
                translation=list(translation),
                size=list(size),
                rotation=list(quat),
                name=class_name,
                score=detection_score
            )
            pred_box3ds.append(box3d)

    return pred_box3ds

if __name__ == '__main__':
    
    # sym links in data dir need to be changed before running this script

    data_path = cfg.DATA.DATA_PATH
    
    json_path = cfg.DATA.TEST_JSON_PATH
    
    l5d = LyftDataset(data_path = data_path,json_path=json_path)
    
    eval_for_kaggle(l5d)   



