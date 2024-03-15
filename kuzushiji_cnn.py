"""
Ronan Bottoms
AMATH 523 HW 5

This file contains a convolutional neural network (CNN) class and trains the model to the Kuzushiji dataset.

"KMNIST Dataset" (created by CODH), adapted from "Kuzushiji Dataset" (created by NIJL and others), doi:10.20676/00000341
https://www.kaggle.com/datasets/anokas/kuzushiji
Vesion 3
"""

import kuzushiji_data as data
from kuzushiji_fcn import format_loss_val_plot, train_model

import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn

import torch
from torch import nn
from torch.utils.data import DataLoader

seaborn.set_style("whitegrid")


def main():
    os.system("cls")
    
    # Create directories
    if not (os.path.isdir("Results")):
        os.mkdir("Results")
    if not (os.path.isdir("Results\\CNN")):
        os.mkdir("Results\\CNN")

    # Get gpu if avaliable
    device = torch.device("cuda:0") if torch.cuda.is_available else torch.device("cpu")

    # Initialize parameters
    learning_rate = 0.001
    epochs = 40
    loss_func = nn.CrossEntropyLoss()

    # Set batch sizes
    train_batch_size = 512
    test_batch_size = 256

    outputs = []
    f, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), layout="constrained")

    #
    # Train model for full dimension
    #

    # Get the data kept in image format
    train, val = data.get_train_dataset(reshape=False)
    test = data.get_test_dataset(reshape=False)

    # Split into batches
    train_batches = DataLoader(train, train_batch_size, True)
    val_batches = DataLoader(val, train_batch_size, True)
    test_batches = DataLoader(test, test_batch_size, True)

    # Training here
    print("Training Full Data (1/2):")
    model = CNN()
    model.to(device)
    print(model)
    optimizer = torch.optim.Adam(model.parameters(), learning_rate)

    loss, val_acc, test_acc, test_time = train_model(model, train_batches, val_batches, test_batches, epochs, loss_func, optimizer, device)
    outputs.append(["Full", loss[len(loss) - 1], val_acc[len(val_acc) - 1], test_acc, test_time])

    # Save in case plotting fails
    df = pd.DataFrame(outputs, columns=["Data", "Loss", "Val Accuracy", "Test Accuracy", "Test Time"])
    df = df.set_index("Data")
    print(df)
    df.to_csv("Results\\CNN\\output.csv", index=True, index_label="Data")

    dim = np.arange(1, len(loss) + 1, 1)
    ax1.plot(dim, loss, label = "Full")
    ax2.plot(dim, val_acc, label = "Full")
    
    #
    # Train model for reduced dimension (90% PCA energy)
    #

    # Reset model
    os.system("cls")
    print("Training Reduced Data (2/2):")
    model = CNN()
    model.to(device)
    print(model)
    optimizer = torch.optim.Adam(model.parameters(), learning_rate)

    # Get the data reshaped into vectors
    train, val = data.get_train_dataset(reshape=False, dim=138)
    test = data.get_test_dataset(reshape=False, dim=138)

    # Split into batches
    train_batches = DataLoader(train, train_batch_size, True)
    val_batches = DataLoader(val, train_batch_size, True)
    test_batches = DataLoader(test, test_batch_size, True)

    loss, val_acc, test_acc, test_time = train_model(model, train_batches, val_batches, test_batches, epochs, loss_func, optimizer, device)
    outputs.append(["Reduced", loss[len(loss) - 1], val_acc[len(val_acc) - 1], test_acc, test_time])

    # Save in case plotting fails
    df = pd.DataFrame(outputs, columns=["Data", "Loss", "Val Accuracy", "Test Accuracy", "Test Time"])
    df = df.set_index("Data")
    print(df)
    df.to_csv("Results\\CNN\\output.csv", index=True, index_label="Data")

    dim = np.arange(1, len(loss) + 1, 1)
    ax1.plot(dim, loss, label = "Reduced")
    ax2.plot(dim, val_acc, label = "Reduced")

    format_loss_val_plot(ax1, ax2, epochs, "Data type")
    plt.suptitle("Training Loss and Validation Accuracy\nfor Full and Reduced Data", fontsize=24)
    plt.savefig("Results\\CNN\\loss_val.png")


class CNN(nn.Module):
    """
    A fixed convolutional neural network.
    """

    def __init__(self):
        super(CNN, self).__init__()

        # output_dim = [(input_dim + 2*pad - kernel + 1) / stride] + 1
        # where output_dim and input_dim are pixel dimensions of the square images.

        self.conv1 = nn.Conv2d(1, 8, kernel_size=4, stride=1, padding=1)
        # output 27

        self.batch1 = nn.BatchNorm2d(8)

        #self.maxpool1 = nn.MaxPool2d(3, 3)
        # output 9

        self.conv2 = nn.Conv2d(8, 16, kernel_size=4, stride=1, padding=1)
        # output 8

        self.batch2 = nn.BatchNorm2d(16)

        #self.maxpool2 = nn.MaxPool2d(2, 2)
        # output 4
        
        self.fc1 = nn.Linear(26 * 26 * 16, 100)

        self.batch3 = nn.BatchNorm1d(100)

        self.fc2 = nn.Linear(100, 49)



    def forward(self, input):
        """
        Propogate input through the model.

        Parameters
        ----------
        input : 
            Input data of same dimensions as input_dimensions of the model.
        """
        x = input
        #print(x.shape)
        x = nn.functional.relu(self.batch1(self.conv1(x)))
        #x = self.maxpool1(x)
        #print(x.shape)
        x = nn.functional.relu(self.batch2(self.conv2(x)))
        #x = self.maxpool2(x)
        #print(x.shape)
        x = x.view(-1, 26 * 26 * 16)
        #print(x.shape)
        x = nn.functional.relu(self.batch3(self.fc1(x)))
        output = self.fc2(x)

        return output


if __name__ == "__main__":
    main()