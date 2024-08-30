import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import argparse


from dep.MyAlexNet import MyAlexNet
from dep.SetCreator import *


parser = argparse.ArgumentParser()

parser.add_argument("--data_path", type=str, required=True, help="Data location")
parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
parser.add_argument("--compute", type=str, required=True, help="(cpu/gpu)")



def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.compute == 'cpu':
        device = torch.device('cpu')
    elif args.compute == 'gpu':
        device = torch.device('gpu')
    else:
        assert False, "Please select gpu or cpu"

    print("Running Device", device)

    # Create instance of training/validation loader
    train_loader, valid_loader = create_training_validation_set(args.data_path)

    # Create instance of test loader
    test_loader = create_testing_set(args.data_path)

    num_classes = 10
    num_epochs = 20 #int(input("Enter Number of Training Epochs: "))
    use_Batch_Norm = True #if str(input("Use BatchNorm Model (y/n): ")) == "y" else False
    batch_size = 64
    learning_rate = 0.005

    model = MyAlexNet(num_classes, use_Batch_Norm).to(device)


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
            loss.backward()
            optimizer.step()

        print('Epoch [{}], Step [{}]'.format(epoch+1, total_step))

    print("Model Trained")
    torch.save(model.state_dict(), f"{args.output_dir}")
    print("Model Saved")

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
    print("done!")
