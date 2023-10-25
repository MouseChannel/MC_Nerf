from nerf_model import *
from load_blender import *
from tqdm import tqdm, trange
from helper import *
import numpy as np
import torch.nn.functional as F

iters = 2000


def train(model, focal_length, image_width, image_height, images, poses, camera_to_world):
    model_optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(model_optimizer, milestones=[2, 4, 8], gamma=0.5)

    for epoch in range(50):
        pbar = trange(iters)
        for i in pbar:
            
            cur_img_index = np.random.randint(0, images.size()[0] - 1)

            ray_origin, ray_dir = get_ray(image_height, image_width, focal_length, camera_to_world[cur_img_index])
            # inference_image format = [image_width, image_height, 3(rgb)]
            inference_image = NerfModel.get_expect_color(model, ray_origin, ray_dir, 2, 6)
            ground_truth_image = images[cur_img_index]
            # a = ground_truth_image[...,:3] - inference_image
            loss = F.mse_loss(inference_image, ground_truth_image[..., :3], reduction='sum')
            model_optimizer.zero_grad()
            loss.backward()
            model_optimizer.step()
            if i%100 == 0:
                helper.render_picture(inference_image, f'iter_{i}_inferencr')
                helper.render_picture(ground_truth_image, f'iter_{i}_ground_truth')
            pbar.set_description("loss = "+str(loss.item()))

        scheduler.step()


if __name__ == '__main__':
    model = NerfModel()

    focal_length, image_width, image_height, images, poses, camera_to_world = load_blender('data/lego')
    train(model, focal_length, image_width, image_height, images, poses, camera_to_world[:-1])
