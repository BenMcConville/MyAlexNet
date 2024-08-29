import matplotlib.pyplot as plt
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np


from dep.MyAlexNet import MyAlexNet
from dep.SetCreator import *



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Running Device", device)

# Create instance of training/validation loader
train_loader, valid_loader = create_training_validation_set('./data')

# Create instance of test loader
test_loader = create_testing_set('./data')



num_classes = 10

use_Batch_Norm = True if str(input("Use BatchNorm Model (y/n): ")) == "y" else False

print("\nModel Loading...")
model = MyAlexNet(num_classes, use_Batch_Norm).to(device)
model.load_state_dict(torch.load("./model"))
model.eval()
print("Model Loaded...")


loader = {"train": train_loader, "valid": valid_loader, "test": test_loader}[str(input("Enter Testing Set (train/valid/test): "))]
# Test:
# m = nn.Softmax(dim=1)
correct = 0
total   = 0
for i in range(20):
    img, label = next(iter(loader))
    x = []
    with torch.no_grad():
        model.eval()
        x = model.forward(img).detach().numpy()
    lab = np.argmax(x, axis=1)
    for j in range(64):
        total += 1
        if lab[j] == label[j]:
            correct += 1

print(correct)
print(total)
print(correct/total * 100)

# def rgb2gray(rgb):
#     return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

# # conv1 = nn.Conv2d(3, 1, 3)
# weight = (model.layer1[1].weight.data.numpy() * (255/0.05)) + (255/2)

# for i in range(len(weight)):

#     print(weight[i].shape)
#     a, b, c = weight[i].shape
#     plt.imshow(rgb2gray(weight[i].reshape(b, c, a)))
#     plt.show()
