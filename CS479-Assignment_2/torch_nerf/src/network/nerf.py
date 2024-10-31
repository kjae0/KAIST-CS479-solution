"""
Pytorch implementation of MLP used in NeRF (ECCV 2020).
"""

from typing import Tuple
from typeguard import typechecked

from jaxtyping import Float, jaxtyped
import torch
import torch.nn as nn


class NeRF(nn.Module):
    """
    A multi-layer perceptron (MLP) used for learning neural radiance fields.

    For architecture details, please refer to 'NeRF: Representing Scenes as
    Neural Radiance Fields for View Synthesis (ECCV 2020, Best paper honorable mention)'.

    Attributes:
        pos_dim (int): Dimensionality of coordinate vectors of sample points.
        view_dir_dim (int): Dimensionality of view direction vectors.
        feat_dim (int): Dimensionality of feature vector within forward propagation.
    """

    def __init__(
        self,
        pos_dim: int,
        view_dir_dim: int,
        feat_dim: int = 256,
    ) -> None:
        """
        Constructor of class 'NeRF'.
        """
        super().__init__()

        # TODO
        self.body = nn.ModuleList()
        self.act_fn = nn.ReLU()
        for i in range(8):
            if i == 0:
                in_dim = pos_dim
            elif i == 4:
                in_dim = feat_dim + pos_dim    # for residual connection (concat)
            else:
                in_dim = feat_dim
            
            self.body.append(nn.Linear(in_dim, feat_dim))
        
        self.rgb = nn.Sequential(
            nn.Linear(feat_dim + view_dir_dim, feat_dim // 2),
            self.act_fn,
            nn.Linear(feat_dim // 2, 3),
        )
        self.sigma = nn.Linear(feat_dim, 1)

    @jaxtyped
    @typechecked
    def forward(
        self,
        pos: Float[torch.Tensor, "num_sample pos_dim"],
        view_dir: Float[torch.Tensor, "num_sample view_dir_dim"],
    ) -> Tuple[Float[torch.Tensor, "num_sample 1"], Float[torch.Tensor, "num_sample 3"]]:
        """
        Predicts color and density.

        Given sample point coordinates and view directions,
        predict the corresponding radiance (RGB) and density (sigma).

        Args:
            pos: The positional encodings of sample points coordinates on rays.
            view_dir: The positional encodings of ray directions.

        Returns:
            sigma: The density predictions evaluated at the given sample points.
            radiance: The radiance predictions evaluated at the given sample points.
        """

        # TODO
        hs = pos
        for i, layer in enumerate(self.body):
            if i == 4:
                hs = torch.cat([hs, pos], dim=-1)
            hs = layer(hs)
            hs = self.act_fn(hs)
            
        rgb = self.rgb(torch.concat([hs, view_dir], dim=-1))
        sigma = self.sigma(hs)
        
        return sigma, rgb
    
    
if __name__ == "__main__":
    device = 'cuda:0'
    
    model = NeRF(60, 24)
    x_coords = torch.randn(32, 60).to(device)
    view_dirs = torch.randn(32, 24).to(device)
    
    model = model.to(device)
    x_coords = x_coords.to(device)
    view_dirs = view_dirs.to(device)
    
    sigma, rgb = model(x_coords, view_dirs)
    
    print(sigma.shape, rgb.shape)
