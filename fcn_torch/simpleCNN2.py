# -*- coding: utf-8 -*-
"""
Created on Sat Jan 27 13:45:14 2024

@author: Ashish
run/compile the keras equavalant onnx model on TI board, for debugging purpose
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import onnx
import onnxruntime
import torch


class Reshape(nn.Module):
    def __init__(self):
        super(Reshape, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)

class fcn(nn.Module):
    def __init__(self, image_width, image_height, num_classes):
        super(fcn, self).__init__()

        # First conv block.
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)

        # Second conv block.
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)

        # Reshape layer.
        self.reshape = Reshape()

        # Fully connected layers.
        self.dense1 = nn.Linear((image_width // 4) * (image_height // 4) * 64, 64)
        self.dropout = nn.Dropout(0.2)
        self.dense2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool2(x)

        x = self.reshape(x)
        x = self.dense1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.dense2(x)

        # Apply softmax activation to get probabilities.
        x = F.softmax(x, dim=1)

        return x

# %% ONNX prediction using dummy data
"""
run the onnx inference on dummy data
"""
dummy_input = torch.randn(1,1,128,32)
ep_list= ['CPUExecutionProvider']
session= onnxruntime.InferenceSession('fcn_torch_updated.onnx',providers = ep_list)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
dummy_input_np = dummy_input.numpy()
pred = session.run([output_name],{input_name:dummy_input_np})
print(pred)