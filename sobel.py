"""
Implements a sobel filter, which I tried to use to preprocess the images in the
datasets. I ended up not using this at all, since it couldn't extract more 
details
"""
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class SobelFilter(nn.Module):

    def __init__(self, k_sobel=3, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
        super().__init__()
        sobel_2D = self.get_sobel_kernel(k_sobel)

        self.register_buffer(
            'sobel_filter_x',
            torch.tensor(sobel_2D.tolist()).view(1, 1, k_sobel, k_sobel))
        self.register_buffer(
            'sobel_filter_y',
            torch.tensor(sobel_2D.T.tolist()).view(1, 1, k_sobel, k_sobel))

        self.padding = nn.ReflectionPad2d(k_sobel // 2)

        self.register_buffer(
            'rgb_weight',
            torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(std).view(1, -1, 1, 1))
        self.register_buffer('mean', torch.tensor(mean).view(1, -1, 1, 1))

    def apply(self, fn):
        return

    def rgb2gray(self, tensor):
        return torch.sum(tensor * self.rgb_weight, 1, keepdim=True)

    @staticmethod
    def get_sobel_kernel(k=3):
        # get range
        range = np.linspace(-(k // 2), k // 2, k)
        # compute a grid the numerator and the axis-distances
        x, y = np.meshgrid(range, range)
        sobel_2D_numerator = x
        sobel_2D_denominator = (x**2 + y**2)
        sobel_2D_denominator[:, k // 2] = 1  # avoid division by zero
        sobel_2D = sobel_2D_numerator / sobel_2D_denominator
        return sobel_2D

    def get_num_channels(self):
        return 2

    def forward(self, x):
        x = self.rgb2gray(x + self.mean / self.std)
        x = self.padding(x)
        grad_x = F.conv2d(x, self.sobel_filter_x)
        grad_y = F.conv2d(x, self.sobel_filter_y)
        return torch.cat([grad_x, grad_y], dim=1)