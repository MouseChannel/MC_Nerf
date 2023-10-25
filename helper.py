import datetime

import torch
import imageio.v2 as imageio
import numpy as np


def get_ray(height, width, focal_length, c2w):
    i, j = torch.meshgrid(
        torch.linspace(0, width - 1, width),
        torch.linspace(0, height - 1, height)
    )
    i = i.t().to('cuda')
    j = j.t().to('cuda')
    K = torch.tensor([
        [focal_length, 0, 0.5 * width],
        [0, focal_length, 0.5 * height],
        [0, 0, 1]]
    )
    camera_axis_dir = torch.stack(
        [(i - K[0][2]) / K[0][0],
         -(j - K[1][2]) / K[1][1],
         -torch.ones_like(i)],
        -1)
    world_axis_dir = torch.sum(torch.unsqueeze(camera_axis_dir, -2) * c2w[:3, :3], -1)
    world_axis_origin = c2w[:3, -1].expand(world_axis_dir.shape)
    return world_axis_origin, world_axis_dir


def render_picture(image,image_name):
    cur_time = datetime.datetime.now()

    image_name ='output/'+image_name+'_'+str(cur_time.hour) + '_' + str(cur_time.minute) + '_' +str( cur_time.second)+'.png'
    imageio.imwrite(image_name, (image.cpu().detach().numpy() * 255.).astype(np.uint8))
