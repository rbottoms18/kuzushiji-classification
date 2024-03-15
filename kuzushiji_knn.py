"""
Ronan Bottoms
AMATH 523

This file contains methods for computing and plotting the cumulative energy contained in PCA components
and classifying data using a KNN classifier, plotting and saving the results.

"KMNIST Dataset" (created by CODH), adapted from "Kuzushiji Dataset" (created by NIJL and others), doi:10.20676/00000341
https://www.kaggle.com/datasets/anokas/kuzushiji
Vesion 3
"""

import kuzushiji_data as data

import os
import time
from tqdm import tqdm
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

sns.set_style("whitegrid")


def main():
    os.system("cls")

    if not (os.path.isdir("Results")):
        os.mkdir("Results")
    if not (os.path.isdir("Results\\KNN")):
        os.mkdir("Results\\KNN")

    #train_data, train_labels = data.get_train_array(reshape=True)
    #test_data, test_labels = data.get_test_array(reshape=True)

    #plot_cum_energy(train_data, 150, 0.9, "Energy in k PCA Components of Training Data", path="Results\\KNN\\cum_energy.png")

    # Project down to 90% of the energy
    train_data, train_labels = data.get_train_array(reshape=True, dim=138)
    test_data, test_labels = data.get_test_array(reshape=True, dim=138)

    classify(train_data, train_labels, test_data, test_labels, num_neighbors = range(1, 11), file_name = "knn_output.csv")


def classify(train_data, train_labels, test_data, test_labels, num_neighbors, file_name) -> None:
    """
    Trains the KNN classifier over a range of number of neighbors, and prints and saves output.

    Parameters
    ----------
    train_data : array
        Training data to train the model.
    train_labels : array
        Ground truth of train_data.
    test_data : array
        Testing data to test the model.
    test_labels : array
        Ground truth of test_data.
    num_neighbors : array
        Array of number of neighbors values to test.
    """
    outputs = []
    for i in tqdm(range(len(num_neighbors)), position=0, leave=False):
        train_score, test_score, val_mean, val_std, test_time = train(train_data, train_labels, test_data, test_labels, num_neighbors[i])
        outputs.append([num_neighbors[i], train_score, test_score, val_mean, val_std, test_time])
    
    df = pd.DataFrame(outputs, columns=["Neighbors", "Train", "Test", "Val Mean", "Val STD", "Test Time"])
    df = df.set_index("Neighbors")
    print(df)
    df.to_csv("Results\\KNN\\{}".format(file_name), index=True, index_label="Neighbors")

    # Plot test accuracy vs number of neighbors
    f = plt.figure()
    df["Test"].plot()
    plt.xlabel("Number of Neighbors")
    plt.ylabel("Testing Accuracy")
    plt.title("Number of Neighbors vs Testing Accuracy")
    plt.savefig("Results\\KNN\\testing_accuracy.png")
    plt.show()

    # Plot test time vs number of neighbors
    f = plt.figure()
    df["Test Time"].plot()
    plt.xlabel("Number of Neighbors")
    plt.ylabel("Testing Time")
    plt.title("Number of Neighbors vs Testing Time")
    plt.savefig("Results\\KNN\\testing_time.png")
    plt.show()


def plot_cum_energy(data, k, target_percent, title, path = "") -> None:
    """
    Plots the cumulative energy of the first k PCA modes of the given data.

    Parameters
    ----------
    data : np.ndarray
        Data to plot the energy of
    k : int
        Number of PCA modes to calculate the energy for
    target_percent : float
        Target percentage of cumulative energy to reach. The target_percent and the minimum number
        of modes to attain the target_percent will be plotted once achieved.
    title : str
        Title of the plot.
    path : str
        File path to save the resulting plot to. Only saves if the path is non-empty.
    """

    pca = PCA(k)
    pca.fit_transform(data)

    # Get and center the covariance matrix
    C = pca.get_covariance()
    centered_C = C - np.mean(C, axis=1)[:, None]

    _, dS, _ = np.linalg.svd(centered_C)
    # Compute cumulative energy
    E = dS / np.sum(dS)
    energy = np.cumsum(E)[:k]

    modes_required = next(x[0] for x in enumerate(energy) if x[1] > target_percent)
    print("Modes required for " + str(target_percent * 100) + "% accuracy: " + str(modes_required))

    # Plot the energy curve
    dim = np.arange(1, k + 1, 1)
    plt.plot(dim, energy)
    plt.scatter(dim, energy, marker='o')

    # Plot target_percent lines
    plt.axhline(y=target_percent, color='black', linestyle='--', alpha=0.5)
    plt.axvline(x=modes_required, color='black', linestyle='--', alpha=0.5)

    # Add extra ticks if necessary
    if modes_required not in list(plt.xticks()[0]):
        new_ticks = list(plt.xticks()[0])
        # Remove any that are too close so they don't overlap
        for tick in new_ticks:
            if (np.abs(tick - modes_required) <= 2.0):
                new_ticks.remove(tick)
        new_ticks.append(modes_required)
        plt.xticks(new_ticks)
    plt.xticks(rotation=45)

    if target_percent not in list(np.round(plt.yticks()[0], 2)):
        plt.yticks(list(plt.yticks()[0]) + [target_percent])

    plt.xlim(left=1, right=k)
    plt.xlabel("k")
    plt.ylabel("Cumulative energy")
    plt.title(title)
    if (path != ""):
        plt.savefig(path)
    plt.show()


def train(train_data, train_labels, test_data, test_labels, num_neighbors, print_output = False) -> np.array:
    """
    Fits a KNN Classifier to the data and prints the resulting accuracies and times, rounded to 3 decimal places.

    Parameters
    ----------
    train_data : array
        Training data to train the model.
    train_labels : array
        Ground truth of train_data.
    test_data : array
        Testing data to test the model.
    test_labels : array
        Ground truth of test_data
    num_neighbors : int
        Number of neighbors to use in the KNN classifier.
    print_output : bool
        If true prints the Return values to the terminal.

    Returns
    -------
    train_score : float
        Training accuracy.
    test_score : float
        Testing accuracy.
    val_mean : float
        Cross-validation mean accuracy.
    val_std : float
        Cross-validation standard deviation
    test_time : float
        Elapsed time of testing the model in seconds.
    """

    knn = KNeighborsClassifier(n_neighbors=num_neighbors)
    knn.fit(train_data, train_labels)

    train_score = np.round(knn.score(train_data, train_labels), 3)

    st = time.time()
    test_score = knn.score(test_data, test_labels)
    et = time.time()
    test_score = np.round(test_score, 3)
    test_time = et - st
    test_time = np.round(test_time, 3)

    val_score = cross_val_score(knn, train_data, train_labels, cv=5)
    val_mean = np.round(val_score.mean(), 3)
    val_std = np.round(val_score.std(), 3)

    if (print_output):
        print("KNN Classifier")
        print("Number of Neighbors: {}".format(num_neighbors))
        print("Training Score: {}".format(train_score))
        print("Testing Score: {}".format(test_score))
        print("Testing time: {} seconds".format(test_time))
        print("{} accuracy with a standard deviation of {}".format(val_mean, val_std))

    return [train_score, test_score, val_mean, val_std, test_time]


if __name__ == "__main__":
    main()