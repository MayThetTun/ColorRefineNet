import math
import pytorch_ssim
from VGG_loss import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import exp


class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = self.create_window(window_size).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    def create_window(self, window_size, channel=1):
        def gaussian(window_size, sigma):
            gauss = torch.Tensor(
                [exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
            return gauss / gauss.sum()

        _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = torch.mm(_1D_window, _1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def forward(self, x, y):
        # Ensure that the window is created with the correct number of channels
        (_, channel, _, _) = x.size()
        if channel != self.channel or self.window.data.type() != x.data.type():
            self.window = self.create_window(self.window_size, channel=channel).to(x.device)
            self.channel = channel

        mu_x = F.conv2d(x, self.window, padding=self.window_size // 2, groups=channel)
        mu_y = F.conv2d(y, self.window, padding=self.window_size // 2, groups=channel)

        mu_x_sq = mu_x ** 2
        mu_y_sq = mu_y ** 2
        mu_x_mu_y = mu_x * mu_y

        sigma_x_sq = F.conv2d(x * x, self.window, padding=self.window_size // 2, groups=channel) - mu_x_sq
        sigma_y_sq = F.conv2d(y * y, self.window, padding=self.window_size // 2, groups=channel) - mu_y_sq
        sigma_xy = F.conv2d(x * y, self.window, padding=self.window_size // 2, groups=channel) - mu_x_mu_y

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_n = (2 * mu_x_mu_y + C1) * (2 * sigma_xy + C2)
        ssim_d = (mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2)
        ssim_map = ssim_n / ssim_d

        if self.size_average:
            return 1 - ssim_map.mean()
        else:
            return 1 - ssim_map.mean(1).mean(1).mean(1)


class combinedloss2(nn.Module):
    def __init__(self, config):
        super(combinedloss2, self).__init__()
        # Load pre-trained VGG19 model with batch normalization
        vgg = models.vgg19_bn(pretrained=True)
        print("VGG model is loaded")
        # Initialize VGG loss with the loaded VGG model and provided configuration
        self.vggloss = VGG_loss(vgg, config)
        # Freeze parameters of the VGG loss to prevent training
        for param in self.vggloss.parameters():
            param.requires_grad = False
        # Initialize Mean Squared Error (MSE) loss and L1 loss, moving them to the specified device
        self.mseloss = nn.MSELoss().to(config['device'])
        self.l1loss = nn.L1Loss().to(config['device'])
        self.ssimloss = SSIMLoss().to(config['device'])

    def forward(self, out, label):
        # Compute VGG loss for the generated output and the ground truth label
        inp_vgg = self.vggloss(out)
        label_vgg = self.vggloss(label)
        # Calculate mean squared error loss
        mse_loss = self.mseloss(out, label)
        # Compute L1 loss between VGG features of output and label
        vgg_loss = self.l1loss(inp_vgg, label_vgg)
        # Compute SSIM loss
        ssim_loss = self.ssimloss(out, label)  # SSIM returns a similarity score, so we subtract from 1 for loss
        # Total loss is the sum of mean squared error loss, VGG loss, and SSIM loss
        total_loss = mse_loss + vgg_loss + ssim_loss
        return total_loss, mse_loss, vgg_loss, ssim_loss
