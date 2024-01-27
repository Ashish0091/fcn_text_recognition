# -*- coding: utf-8 -*-
"""
Created on Sat Jan 27 13:44:01 2024

@author: Ashish 
just compile and check for the errors on TI board
"""
# SimpleCNN model test-1
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import onnx
import torch
import onnxruntime

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)

        # Max pooling layers
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 16 * 4, 512)
        self.fc2 = nn.Linear(512, 39)  # Assuming 10 classes for classification

    def forward(self, x):
        # Convolutional layers with ReLU activation and max pooling
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        # Flatten the tensor before fully connected layers
        x = x.view(-1, 128 * 16 * 4)
        # Fully connected layers with ReLU activation
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

# # Instantiate the model
# model = SimpleCNN()
# # Print the model architecture
# print(model)
# torch.save(model.state_dict(),'/content/basic_cnn.pth')
# %% onnx prediction on dumy input
# onnxruntime prediction
dummy_input = torch.randn(1,1,128,32)
ep_list= ['CPUExecutionProvider']
session= onnxruntime.InferenceSession('simplecnn.onnx',providers = ep_list)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
dummy_input_np = dummy_input.numpy()
pred = session.run([output_name],{input_name:dummy_input_np})
print(pred)


