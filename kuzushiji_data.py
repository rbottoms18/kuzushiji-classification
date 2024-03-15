"""
Ronan Bottoms
AMATH 523 HW 5

This file contains methods for loading the training data, testing data, projecting the data onto fewer dimensions using PCA,
and ploting the first 100 training images from the Kuzushiji49 Data Set.

"KMNIST Dataset" (created by CODH), adapted from "Kuzushiji Dataset" (created by NIJL and others), doi:10.20676/00000341
https://www.kaggle.com/datasets/anokas/kuzushiji
Vesion 3
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, Subset
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA


def main():
    os.system("cls")

    # Get size of the test data
    test_data, _ = get_test_array()
    print("Test data size:" + str(len(test_data)))

    # Plot full dimension first 100
    plot_train_100(title="First 100 Kuzushiji", path="Results\\first_100.png")

    # Plot projected first 100
    plot_train_100(138, title="First 100 Kuzushiji,\n138 Dimensions", path="Results\\first_100_138.png")


class KuzushijiImageDataset(Dataset):
    """
    Dataset object of Kuzushiji data.
    """
    def __init__(self, imgs, labels, transform=None, target_transform=None) -> None:
        """
        Parameters
        ----------
        imgs : np.ndarray | np.array
            Image data as an array.
        labels : np.array
            Ground truth for the images as an array.
        """
        self.imgs = imgs
        self.labels = labels
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        image = torch.tensor(self.imgs[index])
        label = self.labels[index]
        if (self.transform):
            image = self.transform(image)
        if (self.target_transform):
            label = self.target_transform(label)
        return image.float(), label


def get_train_array(reshape=True, dim = 0) -> tuple[np.ndarray, np.array]:
    """
    Gets the training data.

    Parameters
    ----------
    reshape : bool
        If true, reshapes the train data from 28x28 images
        to arrays of length 784. If false, the data will be left as 28x28. 
        Default true.
    dim : int
        Dimensions to reduce the training data to. Defaut value 0 corresponding to no reduction.
        If 'reshape', each image will be a projected vector of length 'dim'.
        If not 'reshape', the data will be projected back to 784 dimensions and reshaped to 28x28.

    Returns
    -------
    train_data : np.ndarray | np.array
        The training data.
    train_labels : np.array
        The ground truth value for each training sample.

    Exceptions
    ----------
    Raises an exception if dim >= 784.
    """

    if (dim >= 784):
        raise Exception("The dimensions to project to must be less than 784.")
    
    train_data = np.load("Data\\k49-train-imgs.npz")["arr_0"]
    train_labels = np.load("Data\\k49-train-labels.npz")["arr_0"]

    # If projecting down dimensions
    if (dim != 0):
        train_data = np.reshape(train_data, (train_data.shape[0], 784))
        pca = PCA(dim)
        pca.fit(train_data)
        components = pca.components_
        train_data = np.matmul(train_data, components.transpose())

        if not (reshape):
            # Take the tranformed data and multiply by the inverse transform to get back reconstructed images.
            train_data = np.matmul(train_data, components)
            train_data = np.reshape(train_data, (train_data.shape[0], 28, 28))

    # Not projecting down dimensions
    else:
        if (reshape):
            train_data = np.reshape(train_data, (train_data.shape[0], 784))
    
    return train_data, train_labels


def get_train_dataset(reshape, validation_percent = 0.1, dim = 0) -> tuple[Subset, Subset]:
    """
    Gets the training and validation data.

    Parameters
    ----------
    reshape : bool
        If true, reshapes the train data from 28x28 image tensors
        to np.ndarrays of length 784.
    dim : int
        Dimension to project the data to using PCA. Default value 0 corresponding to no
        dimension reduction.

    Returns
    -------
    train : Subset
        Subset of KuzushijiImageDataset holding training data.
    val : Subset
        Subset of KuzushijiImageDataset holding validation data.
    """
    train_data, train_labels = get_train_array(reshape, dim)

    if not (reshape):
        dataset = KuzushijiImageDataset(train_data, train_labels, transform=transforms.Compose([transforms.ToPILImage(), transforms.Resize((28,28)),
                                                                        transforms.ToTensor() ]))
    else:
        dataset = KuzushijiImageDataset(train_data, train_labels)

    # Generate indices for a validation set
    train_indices, val_indices, _, _ = train_test_split(
        range(len(train_data)),
        train_labels,
        stratify=train_labels,
        test_size=validation_percent
    )

    # Get training and validation subsets
    train = Subset(dataset, train_indices)
    val = Subset(dataset, val_indices)

    return train, val


def get_test_array(reshape = True, dim = 0) -> tuple[np.ndarray, np.array]:
    """
    Gets the testing data as a np.array.

    Parameters
    ----------
    reshape : bool
        If true, reshapes the test data from 28x28 images
        to arrays of length 784. Default true.
    dim : int
        Dimensions to reduce the testing data to. Defaut value 0 corresponding to no reduction.
        If 'reshape', each image will be a projected vector of length 'dim'.
        If not 'reshape', the data will be projected back to 784 dimensions and reshaped to 28x28.

    Returns
    -------
    test_data : np.ndarray | np.array
        The testing data.
    test_labels : np.array
        The ground truth value for each testing sample.
        
    Exceptions
    ----------
    Raises an exception if dim >= 784.
    """
    
    if (dim >= 784):
        raise Exception("The dimensions to project to must be less than 784.")
    
    test_data = np.load("Data\\k49-test-imgs.npz")["arr_0"]
    test_labels = np.load("Data\\k49-test-labels.npz")["arr_0"]

    # If projecting down dimensions
    if (dim != 0):
        test_data = np.reshape(test_data, (test_data.shape[0], 784))
        train_data, _ = get_train_array(reshape=True)
        pca = PCA(dim)
        pca.fit(train_data)
        components = pca.components_
        test_data = np.matmul(test_data, components.transpose())

        if not (reshape):
            # Take the tranformed data and multiply by the inverse transform to get back reconstructed images.
            test_data = np.matmul(test_data, components)
            test_data = np.reshape(test_data, (test_data.shape[0], 28, 28))

    # Not projecting down dimensions
    else:
        if (reshape):
            test_data = np.reshape(test_data, (test_data.shape[0], 784))
    
    return test_data, test_labels


def get_test_dataset(reshape, dim = 0) -> KuzushijiImageDataset:
    """
    Gets the testing data as a KuzushijiImageDataset.

    Parameters
    ----------
    reshape : bool
        If true, reshapes the train data from 28x28 image tensors
        to np.ndarrays of length 784.
    dim : int
        Dimension to project the data to using PCA. Default value 0 corresponding to no
        dimension reduction.

    Returns
    -------
    KuzushijiImageDataset of the testing data. 
    """
    test_data, test_labels = get_test_array(reshape, dim)

    if not (reshape):
        return KuzushijiImageDataset(test_data, test_labels, transform=transforms.Compose([transforms.ToPILImage(), transforms.Resize((28,28)),
                                                                        transforms.ToTensor() ]))
    return KuzushijiImageDataset(test_data, test_labels)


def pca_project(pca_data, proj_data, k) -> np.ndarray:
    """
    Projects data onto the first k PCA components of another set of data.

    Parameters
    ----------
    pca_data : np.ndarray
        Data used to get the PCA components.
    proj_data : np.ndarray
        Data to project onto the first k PCA components of pca_data.
    k : int
        Number of PCA components used.

    Returns
    -------
    pca_proj_data : np.ndarray
        proj_data projected onto the first k PCA components of pca_data.
    """
    pca = PCA(k)
    pca.fit(pca_data)
    components = pca.components_
    pca_proj_data = np.matmul(proj_data, components.transpose())
    return pca_proj_data


def plot_train_100(k = 0, title = "", path=""):
    """
    Plots the first 100 training kuzushiji.

    Parameters
    ----------
    path : str
        File path to save the resulting image to. Default empty.
    """
    if (title == ""):
        title = "First 100 Kuzushiji"

    if (k == 0):
        train_data, _ = get_train_array(reshape=False)
    else:
        train_data, _ = get_train_array(reshape=False, dim=k)
    _, axs = plt.subplots(10, 10, figsize=(8, 8), sharex=True, sharey=True)
    plt.suptitle(title, fontsize=24)
    for i in range(100):
        axs[i // 10, i % 10].set_xticks([])
        axs[i // 10, i % 10].set_yticks([])
        axs[i // 10, i % 10].imshow(train_data[i, :, :], cmap="gray_r")
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    if (path != ""):
        plt.savefig(path)
    plt.show()


if __name__ == "__main__":
    main()