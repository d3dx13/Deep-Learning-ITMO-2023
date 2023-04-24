import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

IMAGE_SIZE = (28, 28)
INPUT_SIZE = (32, 32)

class ImageFilter:
    def __init__(self):
        kernel_Roberts_x = torch.FloatTensor([
            [1, -1],
            [0, 0]
        ]).unsqueeze(0).unsqueeze(0).cuda()
        kernel_Roberts_y = torch.FloatTensor([
            [1, 0],
            [-1, 0]
        ]).unsqueeze(0).unsqueeze(0).cuda()
        self.weight_Roberts_x = nn.Parameter(data=kernel_Roberts_x, requires_grad=False)
        self.weight_Roberts_y = nn.Parameter(data=kernel_Roberts_y, requires_grad=False)
        
        kernel_Roberts_diag_x = torch.FloatTensor([
            [1, 0],
            [0, -1]
        ]).unsqueeze(0).unsqueeze(0).cuda()
        kernel_Roberts_diag_y = torch.FloatTensor([
            [0, 1],
            [-1, 0]
        ]).unsqueeze(0).unsqueeze(0).cuda()
        self.weight_Roberts_diag_x = nn.Parameter(data=kernel_Roberts_diag_x, requires_grad=False)
        self.weight_Roberts_diag_y = nn.Parameter(data=kernel_Roberts_diag_y, requires_grad=False)
    
    def apply_Roberts(self, x, mux=1):
        res_x = F.conv2d(x, mux * self.weight_Roberts_x, padding=3)
        res_y = F.conv2d(x, mux * self.weight_Roberts_y, padding=3)
        res = torch.sqrt(torch.pow(res_x, 2) + torch.pow(res_y, 2))
        return res[:,:,1:,1:]
    
    def apply_Roberts_diag(self, x, mux=1):
        res_x = F.conv2d(x, mux * self.weight_Roberts_diag_x, padding=3)
        res_y = F.conv2d(x, mux * self.weight_Roberts_diag_y, padding=3)
        res = torch.sqrt(torch.pow(res_x, 2) + torch.pow(res_y, 2))
        return res[:,:,1:,1:]
    
    def apply_Roberts_combined(self, x, bidirectional=False):
        res = torch.pow(F.conv2d(x, 1 * self.weight_Roberts_x, padding=3), 2)
        res += torch.pow(F.conv2d(x, 1 * self.weight_Roberts_y, padding=3), 2)
        if bidirectional:
            res += torch.pow(F.conv2d(x, -1 * self.weight_Roberts_x, padding=3), 2)
            res += torch.pow(F.conv2d(x, -1 * self.weight_Roberts_y, padding=3), 2)
        res += torch.pow(F.conv2d(x, 1 * self.weight_Roberts_diag_x, padding=3), 2)
        res += torch.pow(F.conv2d(x, 1 * self.weight_Roberts_diag_y, padding=3), 2)
        if bidirectional:
            res += torch.pow(F.conv2d(x, -1 * self.weight_Roberts_diag_x, padding=3), 2)
            res += torch.pow(F.conv2d(x, -1 * self.weight_Roberts_diag_y, padding=3), 2)
        return torch.sqrt(res[:,:,1:,1:])

class SimpleNet(nn.Module):
    def __init__(self, num_classes):
        super(SimpleNet, self).__init__()
        self.image_filter = ImageFilter()
        
        self.conv1 = nn.Conv2d(1, 32, kernel_size=7, padding=4, stride=3)
        self.dropout1 = nn.Dropout(.5)
        self.conv2 = nn.Conv2d(32, 128, kernel_size=5, padding=3, stride=3)
        self.dropout2 = nn.Dropout(.5)
        self.conv3 = nn.Conv2d(128, 64, kernel_size=3, padding=2, stride=3)
        self.dropout3 = nn.Dropout(.5)
        self.conv4 = nn.Conv2d(64, 32, kernel_size=2, padding=1)
        self.fc1 = nn.Linear(512, 150)
        self.fc2 = nn.Linear(150, num_classes)
    
    def prepare(self, x):
        x = self.image_filter.apply_Roberts_combined(x)
        return x
    
    def forward(self, x):
        x = self.prepare(x)
        
        x = nn.functional.relu(self.conv1(x))
        x = self.dropout1(x)
        x = nn.functional.relu(self.conv2(x))
        x = self.dropout2(x)
        x = nn.functional.relu(self.conv3(x))
        x = self.dropout3(x)
        x = nn.functional.relu(self.conv4(x))
        x = x.view(-1, 512)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x

