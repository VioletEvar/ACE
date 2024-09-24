import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .utils_detectron2 import DefaultPredictor_Lazy
from pathlib import Path
import cv2
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from tqdm import tqdm
from .vitpose_model import ViTPoseModel
# import ipdb
import os



def rat(cfg):
    return RelativeAttentionTokenization(
        input_dim=cfg.MODEL.RAT.INPUT_DIM,
        hidden_size_1=cfg.MODEL.RAT.HIDDEN_SIZE_1,
        hidden_size_2=cfg.MODEL.RAT.HIDDEN_SIZE_2,
        hidden_size_3=cfg.MODEL.RAT.HIDDEN_SIZE_3,
        output_dim=cfg.MODEL.RAT.OUTPUT_DIM,
        t = cfg.MODEL.RAT.TAU,
    )


def get_postion_map(bbox, H = 16, W = 12) -> float:
    '''
     using the images with bounding boxes to calculate the position map of the two hands
    '''
    x_min, y_min, x_max, y_max = bbox
    cx = (x_max + x_min) / 2
    cy = (y_max + y_min) / 2
    sx = (x_max - x_min)
    sy = (y_max - y_min)
    C = np.zeros((H, W, 2))
    for i in range(W):
        for j in range(H):
            C[j, i] = [cx + (2 * i - W) * sx / (2 * W), cy + (2 * j - H) * sy / (2 * H)]
        
    return C



def get_dis_tok(lefthand_p_map: float, righthand_p_map: float, scale, t):
    '''
    Using the position map to calculate the relative distance token by activation function, the hyperparameter should be gain from the parser
    '''

    rel_distance = t * (righthand_p_map - lefthand_p_map) / scale
    dis_tok = torch.sigmoid(torch.tensor(rel_distance))
    return dis_tok

def inside(patch, Bbox):
    edge_x = Bbox[0], Bbox[2]
    edge_y = Bbox[1], Bbox[3]
    if (patch[0] < edge_x[1]) & (patch[0] > edge_x[0]) & (patch[1] < edge_y[1]) & (patch[1] > edge_y[0]):
        return 1
    else:
        return -1



def get_overlapping_map(pos_map, Bbox):
    '''
    using the image with bounding box to gain if some part of the hand is overlapped by others
    '''
    H, W, _ = pos_map.shape
    O_map = torch.zeros((H, W, 1))
    for i in range(W):
        for j in range(H):
            O_map[j, i] = inside(pos_map[j, i], Bbox)
    return O_map



# def process_img(image_tensor):
    """
    Process a single image tensor to detect hand bounding boxes and keypoints.

    Args:
        image_tensor (torch.Tensor): Input image tensor with shape [H, W, C].

    Returns:
        boxes (np.array): Bounding boxes for detected hands.
        right (np.array): Array indicating which boxes correspond to the right hand.
    """
    # Convert the tensor to a numpy array if necessary
    # if isinstance(image_tensor, torch.Tensor):
    #     image = image_tensor.permute(1, 2, 0).cpu().numpy()  # Convert from [C, H, W] to [H, W, C]
    # else:
    #     image = image_tensor
    image = image_tensor
    detectron2_cfg = model_zoo.get_config('new_baselines/mask_rcnn_regnety_4gf_dds_FPN_400ep_LSJ.py', trained = True)
    detectron2_cfg.model.roi_heads.box_predictor.test_score_thresh = 0.5
    detectron2_cfg.model.roi_heads.box_predictor.test_nms_thresh = 0.4
    detector = DefaultPredictor_Lazy(detectron2_cfg)
    cpm = ViTPoseModel(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    
    # Detecting objects and keypoints
    det_img = detector(image)
    img = image.copy()[:, :, ::-1]  # Convert to RGB format

    det_instances = det_img['instances']
    valid_idx = (det_instances.pred_classes == 0) & (det_instances.scores > 0.5)
    pred_bboxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
    pred_scores = det_instances.scores[valid_idx].cpu().numpy()

    vitposes_out = cpm.predict_pose(
        img,
        [np.concatenate([pred_bboxes, pred_scores[:, None]], axis=1)],
    )

    bboxes = []
    is_right = []

    # Use hands based on hand keypoint detections
    for vitposes in vitposes_out:
        left_hand_keyp = vitposes['keypoints'][-42:-21]
        right_hand_keyp = vitposes['keypoints'][-21:]

        # Rejecting not confident detections
        keyp = left_hand_keyp
        valid = keyp[:, 2] > 0.5
        if sum(valid) > 3:
            bbox = [keyp[valid, 0].min(), keyp[valid, 1].min(), keyp[valid, 0].max(), keyp[valid, 1].max()]
            bboxes.append(bbox)
            is_right.append(0)
        keyp = right_hand_keyp
        valid = keyp[:, 2] > 0.5
        if sum(valid) > 3:
            bbox = [keyp[valid, 0].min(), keyp[valid, 1].min(), keyp[valid, 0].max(), keyp[valid, 1].max()]
            bboxes.append(bbox)
            is_right.append(1)

    if len(bboxes) == 0:
        return None, None

    boxes = np.stack(bboxes)
    right = np.stack(is_right)

    return boxes, right

def convert_tensor_to_numpy(tensor):
    # Ensure the tensor is detached and on the CPU
    if tensor.requires_grad:
        tensor = tensor.detach()
    if tensor.is_cuda:
        tensor = tensor.cpu()
    # Convert to numpy array
    numpy_array = tensor.numpy()
    return numpy_array

def process_img(img_cv2):
    detectron2_cfg = model_zoo.get_config('new_baselines/mask_rcnn_regnety_4gf_dds_FPN_400ep_LSJ.py', trained=True)
    detectron2_cfg.model.roi_heads.box_predictor.test_score_thresh = 0.5
    detectron2_cfg.model.roi_heads.box_predictor.test_nms_thresh = 0.4
    detector = DefaultPredictor_Lazy(detectron2_cfg)
    cpm = ViTPoseModel(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    # Process the single image
    det_img = detector(img_cv2)
    img = img_cv2.copy()[:, :, ::-1]

    det_instances = det_img['instances']
    valid_idx = (det_instances.pred_classes == 0) & (det_instances.scores > 0.5)
    pred_bboxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
    pred_scores = det_instances.scores[valid_idx].cpu().numpy()

    vitposes_out = cpm.predict_pose(
        img_cv2,
        [np.concatenate([pred_bboxes, pred_scores[:, None]], axis=1)],
    )

    bboxes = []
    is_right = []

    # Use hands based on hand keypoint detections
    for vitposes in vitposes_out:
        left_hand_keyp = vitposes['keypoints'][-42:-21]
        right_hand_keyp = vitposes['keypoints'][-21:]

        # Rejecting not confident detections
        keyp = left_hand_keyp
        valid = keyp[:, 2] > 0.5
        if sum(valid) > 3:
            bbox = [keyp[valid, 0].min(), keyp[valid, 1].min(), keyp[valid, 0].max(), keyp[valid, 1].max()]
            bboxes.append(bbox)
            is_right.append(0)
        keyp = right_hand_keyp
        valid = keyp[:, 2] > 0.5
        if sum(valid) > 3:
            bbox = [keyp[valid, 0].min(), keyp[valid, 1].min(), keyp[valid, 0].max(), keyp[valid, 1].max()]
            bboxes.append(bbox)
            is_right.append(1)

    if len(bboxes) == 0:
        return None, None

    boxes = np.stack(bboxes)
    right = np.stack(is_right)

    # Free up GPU memory by deleting model instances and clearing cache
    del detector
    del cpm
    torch.cuda.empty_cache()

    # Returning the bounding boxes and the indication of right hand
    return boxes, right


def concat_maps(dis_tok_r2l, dis_tok_l2r, O_map, is_right_hand: bool):
    if is_right_hand:
        arrays = []
        arrays.append(dis_tok_r2l) 
        arrays.append(dis_tok_l2r)
        arrays.append(O_map)
        concated_map = np.concatenate(arrays, axis=-1)
        concated_map = torch.from_numpy(concated_map).float()
        return concated_map
    else:
        arrays = []
        arrays.append(dis_tok_l2r)
        arrays.append(dis_tok_r2l)
        arrays.append(O_map)
        concated_map = np.concatenate(arrays, axis=-1)
        concated_map = torch.from_numpy(concated_map).float()
        return concated_map
        
            


class Mlp(nn.Module):
    def __init__(self, input_dim: int, hidden_size_1: int, hidden_size_2: int, hidden_size_3: int, output_dim: int):
        super().__init__()
        if input_dim < 0:
            raise ValueError(f"input_dim should be positive, but got {input_dim}")
        if output_dim < 0:
            raise ValueError(f"output_dim should be positive, but got {output_dim}")
        
        '''
        Gradual expansion of the feature dimensions with an additional transition layer.
        Layer1: Linear(input_dim, hidden_size_1) -> ReLU
        Layer2: Linear(hidden_size_1, hidden_size_2) -> ReLU
        Layer3: Linear(hidden_size_2, hidden_size_3) -> ReLU
        Layer4: Linear(hidden_size_3, output_dim) -> ReLU
        '''
        
        self.fc1 = nn.Linear(input_dim, hidden_size_1)
        self.fc2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.fc3 = nn.Linear(hidden_size_2, hidden_size_3)
        self.fc4 = nn.Linear(hidden_size_3, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))   # Expand from input_dim to hidden_size_1
        x = self.relu(self.fc2(x))   # Expand to hidden_size_2
        x = self.relu(self.fc3(x))   # Expand to hidden_size_3
        x = self.relu(self.fc4(x))   # Finally expand to output_dim
        return x


class RelativeAttentionTokenization(nn.Module):
    def __init__(self, input_dim: int, hidden_size_1: int, hidden_size_2: int, hidden_size_3: int, output_dim: int, t):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2
        self.hidden_size_3 = hidden_size_3
        self.output_dim = output_dim
        self.t = t

    # def forward(self, image, fixed_value_bbox=(0, 0, 0, 0)):
    def forward(self, lh_box, rh_box):

        p_map_l, p_map_r = get_postion_map(lh_box), get_postion_map(rh_box)

        tau = self.t
        rh_scale = [rh_box[2], rh_box[3]]
        lh_scale = [lh_box[2], lh_box[3]]
        # rh_scale = (rh_box[2] - rh_box[0]) * (rh_box[3] - rh_box[1])
        # lh_scale = (lh_box[2] - lh_box[0]) * (lh_box[3] - lh_box[1])

        r2l_dis_tok, l2r_dis_tok = get_dis_tok(p_map_r, p_map_l, rh_scale, tau), get_dis_tok(p_map_l, p_map_r, lh_scale, tau)
        # Replace NaN values with a default value, e.g., 0.0
        r2l_dis_tok = torch.nan_to_num(r2l_dis_tok, nan=0.0)
        l2r_dis_tok = torch.nan_to_num(l2r_dis_tok, nan=0.0)

        r2l_O_map, l2r_O_map = get_overlapping_map(p_map_r, lh_box), get_overlapping_map(p_map_l, rh_box)


        all_map_r2l = concat_maps(r2l_dis_tok, l2r_dis_tok, r2l_O_map, is_right_hand=1)

        all_map_l2r = concat_maps(r2l_dis_tok, l2r_dis_tok, l2r_O_map, is_right_hand=0)
        # Replace NaN values with a default value, e.g., 0.0
        all_map_r2l = torch.nan_to_num(all_map_r2l, nan=0.0)
        all_map_l2r = torch.nan_to_num(all_map_l2r, nan=0.0)

        mlp_in = self.input_dim
        mlp_hidden_1 = self.hidden_size_1
        mlp_hidden_2 = self.hidden_size_2
        mlp_hidden_3 = self.hidden_size_3
        mlp_out = self.output_dim

        tokenization_mlp = Mlp(mlp_in, mlp_hidden_1, mlp_hidden_2, mlp_hidden_3, mlp_out)

        r2l_token = tokenization_mlp(all_map_r2l)
        l2r_token = tokenization_mlp(all_map_l2r)

        return r2l_token, l2r_token



        



