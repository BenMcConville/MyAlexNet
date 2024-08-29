import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


from dep.MyAlexNet import MyAlexNet
from dep.SetCreator import *


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Running Device", device)


# Create instance of training/validation loader
train_loader, valid_loader = create_training_validation_set('./data')

# Create instance of test loader
test_loader = create_testing_set('./data')




num_classes = 10
num_epochs = int(input("Enter Number of Training Epochs: "))
use_Batch_Norm = True if str(input("Use BatchNorm Model (y/n): ")) == "y" else False
batch_size = 64
learning_rate = 0.005



model = MyAlexNet(num_classes, use_Batch_Norm).to(device)


parameters = model.parameters()
print(sum(p.nelement() for p in parameters))
for p in parameters:
    p.requires_grad = True

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay = 0.005, momentum = 0.9)


# Train the model

total_step = len(train_loader)

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):

        # Move tensors to the configured device
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        # for p in parameters:
        #     p.grad = None
        loss.backward()
        optimizer.step()

    print('Epoch [{}], Step [{}]'.format(epoch+1, total_step))

print("Model Trained")
torch.save(model.state_dict(), "./model")
print("Model Saved")
