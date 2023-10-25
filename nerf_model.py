import torch
import torch.nn as nn
import torch.nn.functional as F
import tinycudann as tcnn



class NerfModel(nn.Module):

    def __init__(self, embedding_pos_dim=60, embedding_dir_dim=24, net_width=256):
        """
        
        Args:
            embedding_pos_dim: camera pos after embedding dim ,default **60**  
            embedding_dir_dim: camera dir after embedding dim , default **24**  
            net_width: default **8**  
        """
        super(NerfModel, self).__init__()
        self.block1 = nn.Sequential(
            nn.Linear(embedding_pos_dim + 3, net_width),
            nn.ReLU(),
            nn.Linear(net_width, net_width),
            nn.ReLU(),
            nn.Linear(net_width, net_width),
            nn.ReLU(),
            nn.Linear(net_width, net_width),
            nn.ReLU()
        )
        
        self.block2 = nn.Sequential(
            nn.Linear(embedding_pos_dim + 3 + net_width, net_width),
            nn.ReLU(),
            nn.Linear(net_width, net_width),
            nn.ReLU(),
            nn.Linear(net_width, net_width),
            nn.ReLU(),
            nn.Linear(net_width, 1 + net_width)
        )
        self.block3 = nn.Sequential(
            nn.Linear(
                embedding_dir_dim + net_width + 3,
                net_width // 2
            ),
            nn.ReLU()
        )
        self.block4 = nn.Sequential(
            nn.Linear(net_width // 2, 3),
            nn.Sigmoid()
        )
        self.embedding_dir_dim = embedding_dir_dim
        self.embedding_pos_dim = embedding_pos_dim
        self.to('cuda')
        self.relu = nn.ReLU()

    @staticmethod
    def positional_encoding(x, L):
        out = [x]
        # temp = torch.linspace( 0 , L-1,L)
        # temp = torch.full_like(temp,2)** temp
        for j in range(L):
            out.append(torch.sin(2 ** j * x))
            out.append(torch.cos(2 ** j * x))
        return torch.cat(out, dim=1)

    def forward(self, origin, dir):
        '''
        we set L = 10 for γ(x) and L = 4 for γ(d).
        Args:
            origin: 
            dir: 

        Returns: inferenced color,Sigma

        '''
        embedding_origin_pos = NerfModel.positional_encoding(origin, 10)
        embedding_direction = NerfModel.positional_encoding(dir, 4)
        rgb = self.block1(embedding_origin_pos)
        rgb1 = self.block2(
            torch.cat((
                embedding_origin_pos, rgb), 1))
        rgb1 = rgb1[:, :-1]

        # sigma = F.relu(rgb1)
        sigma = F.relu(rgb1[:, -1])

        rgb2 = self.block3(torch.cat((rgb1, embedding_direction), 1))
        rgb3 = self.block4(rgb2)
        return rgb3, sigma

    @staticmethod
    def compute_accumulated_transmittance(alphas):
        accumulated_transmittance = torch.cumprod(alphas, -1)
        return torch.cat((torch.ones(
            (accumulated_transmittance.shape[0], accumulated_transmittance.shape[1], 1),
            device=alphas.device),
                          accumulated_transmittance[..., :-1]), dim=-1)

    @staticmethod
    def get_expect_color(model, ray_origin, ray_dir, near_plane, far_plane, step=128):
        device = ray_origin.device
        t = torch.linspace(near_plane, far_plane, step, device=device)
        t = t.expand(ray_origin.shape[0], ray_origin.shape[1], step)
        start = t[..., 1:]
        end = t[..., :-1]
        mid = (start + end) / 2
        lower = torch.cat((t[..., :1], mid), -1)
        upper = torch.cat((mid, t[..., -1:]), -1)
        U = torch.rand(t.shape, device=device)
        sample_t = lower + (upper - lower) * U
        delta = torch.cat(
            (sample_t[..., 1:] - sample_t[..., :-1], torch.tensor([1e10], device=device).expand(
                ray_origin.shape[0], ray_origin.shape[1], 1)), -1)
        current_pos = ray_origin.unsqueeze(-2) + sample_t.unsqueeze(-1) * ray_dir.unsqueeze(-2)
        ray_dir = ray_dir.expand(step, ray_dir.shape[0], ray_dir.shape[1], 3).transpose(0, 1).transpose(1, 2)
        rgb, sigma = model(current_pos.reshape(-1, 3),
                           ray_dir.reshape(-1, 3))
        rgb = rgb.reshape(current_pos.shape)
        sigma = sigma.reshape(current_pos.shape[:-1])
        alpha = 1 - torch.exp(-sigma * delta)
        weights = NerfModel.compute_accumulated_transmittance(1 - alpha).unsqueeze(-1) * alpha.unsqueeze(-1)
        # sum all point value in a ray
        C = (weights * rgb).sum(-2)

        return C
