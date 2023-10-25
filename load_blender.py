import datetime
import json
import os
import time

import imageio.v2 as imageio
import cv2
import torch
import numpy as np

down_sample = 32

import helper


def load_blender(base_dir):
    splits = ['train', 'val', 'test']

    metas = {}
    for s in splits:
        with open(os.path.join(base_dir, 'transforms_{}.json'.format(s)), 'r') as fp:
            metas[s] = json.load(fp)
    # only use train dataset
    meta = metas['train']
    images = []

    poses = []

    for frame in meta['frames']:
        frame_name = os.path.join(base_dir, frame['file_path'] + '.png')
        images.append(imageio.imread(frame_name))
        poses.append(frame['transform_matrix'])

    images = (np.array(images) / 255.).astype(np.float32)
    poses = torch.tensor(np.array(poses), device='cuda').float()
    camera_angle_x = torch.tensor(meta['camera_angle_x'])

    image_height, image_width = images[0].shape[:2]

    image_height = image_height // down_sample
    image_width = image_width // down_sample

    after_down_sample_images = []
    for image in images:
        # cur_img = images[i]
        after_down_sample_images.append(cv2.resize(image, (image_width, image_height), interpolation=cv2.INTER_AREA))
    focal_length = 0.5 * image_width / torch.tan(0.5 * camera_angle_x)

    images = (torch.tensor(after_down_sample_images, device='cuda')).float()
    cx = image_width / 2
    cy = image_height / 2
    camera_to_world = poses[:, :3]
    # helper.render_picture(images[0])
    return focal_length, image_width, image_height, images, poses, camera_to_world

# def get_ground_truth_image(image_index, images):
