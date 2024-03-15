"""
Ronan Bottoms
AMATH 523 HW 5

This file contains methods for creating a fully connected neural network (FCN) for the Kuzushiji dataset
at different dimensions, tuning the model based on the number of layers and number of neurons per layer, and
plotting and saving the output.

"KMNIST Dataset" (created by CODH), adapted from "Kuzushiji Dataset" (created by NIJL and others), doi:10.20676/00000341
https://www.kaggle.com/datasets/anokas/kuzushiji
Vesion 3
"""

import kuzushiji_data as data

import os
import time
import numpy as np
from tqdm import tqdm
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
    if not (os.path.isdir("Results\\FCN")):
        os.mkdir("Results\\FCN")
        os.mkdir("Results\\FCN\\Full")
        os.mkdir("Results\\FCN\\Full\\Neuron")
        os.mkdir("Results\\FCN\\Full\\Layer")
        os.mkdir("Results\\FCN\\138")
        os.mkdir("Results\\FCN\\138\\Neuron")
        os.mkdir("Results\\FCN\\138\\Layer")

    # Get gpu if avaliable
    device = torch.device("cuda:0") if torch.cuda.is_available else torch.device("cpu")

    # Initialize parameters
    learning_rate = 0.001
    epochs = 40
    loss_func = nn.CrossEntropyLoss()

    # Set batch sizes
    train_batch_size = 512
    test_batch_size = 256

    #
    # Train and tune model for full dimension
    #

    # Get the data reshaped into vectors
    train, val = data.get_train_dataset(reshape=True)
    test = data.get_test_dataset(reshape=True)

    # Split into batches
    train_batches = DataLoader(train, train_batch_size, True)
    val_batches = DataLoader(val, train_batch_size, True)
    test_batches = DataLoader(test, test_batch_size, True)

    tune_neurons([25, 50, 75, 100, 125, 150], 3, 784, train_batches, val_batches, test_batches, epochs, 
                 loss_func, torch.optim.Adam, learning_rate, device, "Results\\FCN\\Full\\Neuron")

    tune_layers([1, 2, 3, 4, 5], 100, 784, train_batches, val_batches, test_batches, epochs, 
                loss_func, torch.optim.Adam, learning_rate, device, "Results\\FCN\\Full\\Layer")
    
    #
    # Train and tune model for reduced dimension (90% PCA energy)
    #

    # Get the data reshaped into vectors
    train, val = data.get_train_dataset(reshape=True, dim=138)
    test = data.get_test_dataset(reshape=True, dim=138)

    # Split into batches
    train_batches = DataLoader(train, train_batch_size, True)
    val_batches = DataLoader(val, train_batch_size, True)
    test_batches = DataLoader(test, test_batch_size, True)

    tune_neurons([25, 50, 75, 100, 125, 150], 3, 138, train_batches, val_batches, test_batches, epochs, 
                 loss_func, torch.optim.Adam, learning_rate, device, "Results\\FCN\\138\\Neuron")

    tune_layers([1, 2, 3, 4, 5], 100, 138, train_batches, val_batches, test_batches, epochs, 
                loss_func, torch.optim.Adam, learning_rate, device, "Results\\FCN\\138\\Layer")
    
    #
    # Plot accuracy comparisons
    #

    # Neuron
    full_df = pd.read_csv("Results\\FCN\\Full\\Neuron\\neuron_tuning_output.csv").set_index("Neurons")
    reduced_df = pd.read_csv("Results\\FCN\\138\\Neuron\\neuron_tuning_output.csv").set_index("Neurons")

    f = plt.figure()
    full_df["Test Accuracy"].plot(label="Full")
    reduced_df["Test Accuracy"].plot(label="Reduced")
    plt.xticks(full_df.index)
    plt.xlabel("Number of Neurons")
    plt.ylabel("Testing Accuracy")
    plt.title("Number of Neurons vs Testing Accuracy")
    plt.legend()
    plt.savefig("Results\\FCN\\neuron_testing_acc_comparison.png")

    f = plt.figure()
    full_df["Test Time"].plot(label="Full")
    reduced_df["Test Time"].plot(label="Reduced")
    plt.xticks(full_df.index)
    plt.xlabel("Number of Neurons")
    plt.ylabel("Testing Time")
    plt.title("Number of Neurons vs Testing Time")
    plt.legend()
    plt.savefig("Results\\FCN\\neuron_testing_time_comparison.png")

    # Layer
    full_df = pd.read_csv("Results\\FCN\\Full\\Layer\\layer_tuning_output.csv").set_index("Layers")
    reduced_df = pd.read_csv("Results\\FCN\\138\\Layer\\layer_tuning_output.csv").set_index("Layers")

    f = plt.figure()
    full_df["Test Accuracy"].plot(label="Full")
    reduced_df["Test Accuracy"].plot(label="Reduced")
    plt.xticks(full_df.index)
    plt.xlabel("Number of Layers")
    plt.ylabel("Testing Accuracy")
    plt.title("Number of Layers vs Testing Accuracy")
    plt.legend()
    plt.savefig("Results\\FCN\\layer_testing_acc_comparison.png")

    f = plt.figure()
    full_df["Test Time"].plot(label="Full")
    reduced_df["Test Time"].plot(label="Reduced")
    plt.xticks(full_df.index)
    plt.xlabel("Number of Layers")
    plt.ylabel("Testing Time")
    plt.title("Number of Layers vs Testing Time")
    plt.legend()
    plt.savefig("Results\\FCN\\layer_testing_time_comparison.png")


def tune_neurons(num_neurons, num_layers, input_dim, train_batches, val_batches, 
                 test_batches, epochs, loss_func, optimizer, learning_rate, device, path) -> None:
    """
    Trains a FCN model for varying numbers of neurons per layer with a fixed number of layers.
    Plots the loss curves and validation accuracy curves for each number of neurons on one plot.

    Parameters
    ----------
    num_neurons : np.array
        Array of different numbers of neurons to use.
    num_layers : int
        Number of layers in the model.
    input_dim : int
        Dimension of the input data.
    train_batches : DataLoader
        Training data grouped into batches
    val_batches : DataLoader
        Validation data grouped into batches.
    test_batches : DataLoader
        Test data grouped into batches.
    epochs : int
        Number of epochs over which to train the model.
    loss_func : from torch.nn
        Loss function to be applied to the training.
    optimizer : from torch.optim
        Optimizer method to be applied to the training.
    learning_rate : float
        Learning rate of the optimizer.
    device : torch.Device
        Device to load the data on for computation.
    path : str
        Folder path to save the output to. Do not include an ending "\\".
    """

    f, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), layout="constrained")
    outputs = []
    for idx, num in enumerate(num_neurons):
        os.system("cls")
        print("Number of Neurons: " + str(num) + " (" + str(idx + 1) + "/" + str(len(num_neurons)) + ")")

        model = FCN(input_dim=input_dim, output_dim=49, num_hidden=num_layers, num_neurons=num, batch_norm=True)
        # model.apply(init_he_uniform)
        model.to(device)
        print(model)
        optimizer = torch.optim.Adam(model.parameters(), learning_rate)

        loss, val_acc, test_acc, test_time = train_model(model, train_batches, val_batches, test_batches, epochs, loss_func, optimizer, device)

        dim = np.arange(1, len(loss) + 1, 1)
        ax1.plot(dim, loss, label = str(num))
        ax2.plot(dim, val_acc, label = str(num))

        outputs.append([num, num_layers, loss[epochs - 1], val_acc[epochs - 1], test_acc, test_time])

    # Save data
    df = pd.DataFrame(outputs, columns=["Neurons", "Layers", "Loss", "Val Accuracy", "Test Accuracy", "Test Time"])
    df = df.set_index("Neurons")
    print("Results:")
    print(df)
    df.to_csv(str(path) + "\\neuron_tuning_output.csv", index=True, index_label="Neurons")

    # Plot loss & accuracy
    format_loss_val_plot(ax1, ax2, epochs, "Number of Neurons")
    plt.suptitle("Training Loss and Validation Accuracy\nvs Epochs", fontsize=24)
    plt.savefig(str(path) + "\\neuron_tuning_loss_acc.png")

    f = plt.figure()
    df["Test Accuracy"].plot()
    plt.xticks(df.index)
    plt.xlabel("Number of Neurons")
    plt.ylabel("Testing accuracy")
    plt.title("Number of Neurons vs Testing Accuracy")
    plt.savefig("Results\\FCN\\Full\\Neuron\\neuron_tuning_test_acc.png")

    # Plot testing time
    f = plt.figure()
    df["Test Time"].plot()
    plt.xticks(df.index)
    plt.xlabel("Number of Neurons")
    plt.ylabel("Testing Time (seconds)")
    plt.title("Number of Neurons vs Testing Time")
    plt.savefig("Results\\FCN\\Full\\Neuron\\neuron_tuning_test_time.png")


def tune_layers(num_layers, num_neurons, input_dim, train_batches, val_batches, 
                test_batches, epochs, loss_func, optimizer, learning_rate, device, path) -> None:
    """
    Trains a FCN model for varying numbers of layers with a fixed number of neurons.
    Plots the loss curves and validation accuracy curves for each number of layers on one plot.

    Parameters
    ----------
    num_layers : np.array
        Array of different numbers of hidden layers in the FCN to use.
    num_neurons : int
        Constant number of neurons in the model.
    input_dim : int
        Dimension of the input data.
    train_batches : DataLoader
        Training data grouped into batches
    val_batches : DataLoader
        Validation data grouped into batches.
    test_batches : DataLoader
        Test data grouped into batches.
    epochs : int
        Number of epochs over which to train the model.
    loss_func : from torch.nn
        Loss function to be applied to the training.
    optimizer : from torch.optim
        Optimizer method to be applied to the training.
    learning_rate : float
        Learning rate of the optimizer.
    device : torch.Device
        Device to load the data on for computation.
    path : str
        Folder path to save the output to. Do not include an ending "\\".
    """

    f, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), layout="constrained")
    outputs = []
    for idx, num in enumerate(num_layers):
        os.system("cls")
        print("Number of Layers: " + str(num) + " (" + str(idx + 1) + "/" + str(len(num_layers)) + ")")

        model = FCN(input_dim=input_dim, output_dim=49, num_hidden=num, num_neurons=num_neurons, batch_norm=True)
        # model.apply(init_he_uniform)
        model.to(device)
        print(model)
        optimizer = torch.optim.Adam(model.parameters(), learning_rate)

        loss, val_acc, test_acc, test_time = train_model(model, train_batches, val_batches, test_batches, epochs, loss_func, optimizer, device)

        dim = np.arange(1, len(loss) + 1, 1)
        ax1.plot(dim, loss, label = str(num))
        ax2.plot(dim, val_acc, label = str(num))

        outputs.append([num_neurons, num, loss[epochs - 1], val_acc[epochs - 1], test_acc, test_time])

    # Save data
    df = pd.DataFrame(outputs, columns=["Neurons", "Layers", "Loss", "Val Accuracy", "Test Accuracy", "Test Time"])
    df = df.set_index("Layers")
    print("Results:")
    print(df)
    df.to_csv(str(path) + "\\layer_tuning_output.csv", index=True, index_label="Layers")

    # Plot loss & accuracy
    format_loss_val_plot(ax1, ax2, epochs, "Number of Layers")
    plt.suptitle("Training Loss and Validation Accuracy\nvs Epochs", fontsize=24)
    plt.savefig(str(path) + "\\layer_tuning_loss_acc.png")

    # Plot testing accuracy
    f = plt.figure()
    df["Test Accuracy"].plot()
    plt.xticks(df.index)
    plt.xlabel("Number of Layers")
    plt.ylabel("Testing accuracy")
    plt.title("Number of Layers vs Testing Accuracy")
    plt.savefig(str(path) + "\\layer_tuning_test_acc.png")

    # Plot testing time
    f = plt.figure()
    df["Test Time"].plot()
    plt.xticks(df.index)
    plt.xlabel("Number of Layers")
    plt.ylabel("Testing Time (seconds)")
    plt.title("Number of Layers vs Testing Time")
    plt.savefig(str(path) + "\\layer_tuning_test_time.png")
        

def format_loss_val_plot(ax1, ax2, epochs, legend_title) -> None:
    """
    Formats axes to show the loss curve and validation accuracy curve.

    Parameters
    ----------
    ax1 : Axis
        Axis loss is plotted to.
    ax2 : Axis
        Axis validation accuracy is plotted to.
    epochs : int
        Number of epochs the data is trained over.
    legend_title : str
        Title of the legend of the plot.
    """

    # Adjust ax1 width
    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    # Adjust ax2 width
    box = ax2.get_position()
    ax2.set_position([box.x0, box.y0, box.width * 0.8, box.height + box.height * 0.1])

    # Put a legend to the right of the axes
    ax1.legend(title=legend_title, loc='center left', bbox_to_anchor=(1, -0.1), fancybox=True)
  
    # Set x tick marks
    ax1.set_xticks(range(2, epochs + 1, 2))
    ax2.set_xticks(range(2, epochs + 1, 2))    

    # Hide ax1 ticks (looks like sharex)
    ax1.tick_params(axis='x', colors='white')
    
    # Set labels
    ax1.set_ylabel("Training loss", fontsize=14)
    ax2.set_ylabel("Validation accuracy", fontsize=14)
    ax2.set_xlabel("Epochs", fontsize=14)


class FCN(nn.Module):
    """
    A dynamic neural network with connections between each neuron in a layer with each neuron in the next layer.
    Admits an adjustable number of layers and an adjustable number of neurons per layer.
    """

    def __init__(self, input_dim, output_dim, num_hidden, num_neurons, batch_norm = False, dropout_weights = []):
        """
        Parameters
        ----------
        input_dim : int
            Number of dimensions of the input data.

        output_dim : int
            Number of dimensions of the output of the network (number of classes).

        num_layers : int
            Number of hidden layers in the network.

        num_neurons : int
            Number of neurons in each hidden layer of the network.

        batch_norm : bool
            True if batch normilization is to be preformed, False if not. Default False.

        dropout_weights : list
            List of dropout probabilities for connections between each layer. Default probabilities
            are zero for each connection, i.e. no dropout preformed. If len(dropout_weights) is less than
            the number of layer connections, the remaining connections will be assigned a dropout probability
            of zero.
        """
        super(FCN, self).__init__()

        self.num_hidden = num_hidden
        self.batch_norm = batch_norm
        
        # Initialize dropout_weights
        if len(dropout_weights) == 0:
            dropout_weights = [0 for i in range(0, num_hidden + 1)]
        elif len(dropout_weights) < num_hidden + 1:
            for i in range(len((dropout_weights)), num_hidden + 1):
                dropout_weights[i] = 0
        
        self.linear_layers = []
        # First layer
        self.linear_layers.append(nn.Linear(input_dim, num_neurons))
        for i in range(num_hidden):
            self.linear_layers.append(nn.Linear(num_neurons, num_neurons))
        # Last layer
        self.linear_layers.append(nn.Linear(num_neurons, output_dim))

        self.batch_norm_layers = []
        if (batch_norm):
            for i in range(num_hidden + 1):
                self.batch_norm_layers.append(nn.BatchNorm1d(num_neurons))

        self.dropout_layers = []
        for i in range(num_hidden + 1):
            self.dropout_layers.append(nn.Dropout(dropout_weights[i]))

        self.layers = nn.ModuleList()
        for i in range(0, num_hidden + 1):
            self.layers.append(self.linear_layers[i])
            if (batch_norm):
                self.layers.append(self.batch_norm_layers[i])
            self.layers.append(self.dropout_layers[i])
        self.layers.append(self.linear_layers[len(self.linear_layers) - 1])


    def forward(self, input):
        """
        Propogate input through the model.

        Parameters
        ----------
        input : 
            Input data of same dimensions as input_dimensions of the model.
        """
        x = input
        # Pass through first linear layer, batch norm, and dropout
        x = nn.functional.relu(self.linear_layers[0](x))
        if (self.batch_norm):
            x = self.batch_norm_layers[0](x)
        x = self.dropout_layers[0](x)

        # Next layer -> Batch norm -> Dropout
        for i in range(1, self.num_hidden + 1):
            x = nn.functional.relu(self.linear_layers[i](x))
            if (self.batch_norm):
                x = self.batch_norm_layers[i](x)
            x = self.dropout_layers[i](x)

        # Pass through last linear layer to output
        # CrossEntropyLoss will apply softmax automatically so no need to pass last layer through relu.
        output = self.linear_layers[len(self.linear_layers) - 1](x)

        return output


def train_model(model, train_batches, val_batches, test_batches, epochs, loss_func, optimizer, device) -> tuple[np.array, np.array, float, float]:
    """
    Trains the model to a set of training data given hyperparameters. Returns the loss, validation accuracy, and testing accuracy.

    Parameters
    ----------
    model : PyTorch nn.Module
        Neural Network model to train.
    train_batches : DataLoader
        Training data grouped into batches
    val_batches : DataLoader
        Validation data grouped into batches.
    test_batches : DataLoader
        Test data grouped into batches.
    epochs : int
        Number of epochs over which to train the model.
    loss_func : from torch.nn
        Loss function to be applied to the training.
    optimizer : from torch.optim
        Optimizer method to be applied to the training.
    device : torch.Device
        Device to load the data on for computation.

    Returns
    -------
    train_loss : np.array
        Loss from the model's predictions at each epoch.
    val_accuracy : np.array
        Accuracy of the model when applied to the validation data at each epoch.
    test_accuracy : float
        Accuracy of the model applied to the test data after training is completed.
    test_time : float
        Time elapsed during testing of the model.
    """
    
    train_loss = np.zeros((epochs, ))
    val_accuracy = np.zeros((epochs, ))

    for epoch in tqdm(range(0, epochs), position=0, leave=False):
        cum_loss = 0
        # Train model
        for train_features, train_labels in train_batches:
            features = train_features.to(device)
            labels = train_labels.to(device)
            
            optimizer.zero_grad()

            model.train()
            train_outputs = model(features)
            loss = loss_func(train_outputs, labels)        

            loss.backward()
            optimizer.step()

            cum_loss += loss.item()

        train_loss[epoch] = cum_loss / len(train_batches)

        # Validate model
        val_acc, _ = _evaluate_model(model, val_batches, device)
        val_accuracy[epoch] = val_acc

    # Test model
    test_acc, test_time = _evaluate_model(model, test_batches, device)

    return train_loss, val_accuracy, test_acc, test_time


def init_random_normal(l):
    """
    Initializes the weights of a layer with values from a Random Normal distribution.
    """
    if isinstance(l, nn.Linear):
        torch.nn.init.normal_(l.weight)


def init_xavier_normal(l):
    """
    Initializes the weights of a layer with values from Xavier Normal Distribution.
    """
    if isinstance(l, nn.Linear):
        torch.nn.init.xavier_normal_(l.weight)


def init_he_uniform(l):
    """
    Initializes the weights of a layer with values from Kaiming (He) Normal Distribution.
    """
    if isinstance(l, nn.Linear):
        torch.nn.init.kaiming_uniform_(l.weight, mode='fan_out', nonlinearity='relu')


def _evaluate_model(model, batches, device) -> tuple[float, float]:
    """
    Evaluates a model's accuracy without computing loss or updating weights.

    Parameters
    ----------
    model : PyTorch nn.Module
        Model to evaluate.
    batches : DataLoader
        Batched data to pass through the model.
    device : torch.Device
        Device to load the data on for computation.

    Returns
    -------
    acc : float
        Accuracy of the model during evaluation averaged across batches.
    elapsed_time : float
        Time elapsed during evaluation.
    """
    sum_acc = 0
    with torch.no_grad():
        st = time.time()
        for batch_features, batch_labels in batches:
            # Move data to the device
            features = batch_features.to(device)
            labels = batch_labels.to(device)
            model.eval()
            pred = model(features)
            # Get accuracy
            correct = (torch.argmax(pred, dim=1) == labels).type(torch.FloatTensor)
            # Add to sum of accuracy
            sum_acc += correct.mean().numpy()
        et = time.time()

    elapsed_time = et - st
    # Get average accuracy over all batches
    acc = sum_acc / len(batches)

    return [acc, elapsed_time]


def _generate_example_data(length) -> tuple[np.array, np.array, float]:
    """
    Generates example training loss and validation and test accuracies for
    testing of plotting methods.

    Parameters
    ----------
    length : int
        Length of the return arrays

    Returns
    -------
    training_loss : np.array
        Array of random values of len 'length' representing training loss.
    validation_acc : np.array
        Array of random values of len 'length' representing validation accuracy.
    test_acc : float
        Random value representing testing accuracy.
    """
    training_loss = np.random.rand((length))
    validation_acc = np.random.rand((length))
    test_acc = np.random.rand(1)

    return training_loss, validation_acc, test_acc[0]


if __name__ == "__main__":
    main()