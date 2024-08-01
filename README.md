# About

The Japanese language uses two alphabets in addition to Chinese characters. “Letters” in these alphabets
originally derived from Chinese characters themselves, often times with multiple different characters representing the same sound. 
For instance, before the alphabets were standardized, the “a” sound could be
represented by not only 安 , but also 阿 and 愛. Cursive or sloppy penmanship can lead to abridged
versions of the characters, like how 安 morphed into the modern day あ . These obsolete or antiquated short
form Chinese characters are referred to as kuzushiji ( 崩し字 [lit. crumbling/destroyed character]).

<p align="center">
  <img src="https://github.com/rbottoms18/kuzushiji-classification/blob/master/Results/first_100.png" width="400"/>
</p>

Nowadays, only one form of each letter exists. But if one wants to read a historical document written prior
to 1900, a knowledge of the many old forms of letters is necessary. For a layperson, or someone whose native
language is not Japanese, learning these different forms may be difficult and time-consuming, especially if
only one or two documents need to be read. The aim of this report is to create, test, and compare three
different machine learning algorithms tasked with classifying the different kuzushiji that may appear.

The analysis used three different kinds of classifier: a K-Nearest Neighbors classifier
(KNN), a Fully-Connected Neural Network (FCN), and a Convolutional Neural Network (CNN).
The Kuzushiji49 dataset, part of the Kuzushiji-MNIST [1] collection was used to test the models. This dataset contains 270,912 grayscale images of 49 different 
classes of kuzushiji (all 48 hiragana characters and one iteration mark).

The analysis aimed to answer the following questions:
1. Which of the three models is the most accurate?
2. Which of the three models is the most accurate in the least amount of time?
3. How does dimension reduction affect accuracy and runtime in all three models?

Additionally, this analysis posed an opportunity to test hyperparameter tuning of the FCN. The following question was added as a secondary goal:

4. How does the number of neurons and layers affect testing accuracy and runtime?

# Methods

The analysis proceeded in two phases: testing the models using the “full” unreduced data, and testing them
using the dimension reduced data. Testing of the models was performed using the dedicated testing data provided by Kuzushiji49 consisting of 38,547 images. 
Testing times were computed as the wall-clock time required to classify each sample in the testing set. The code to process the data for training and testing
is in `kuzushiji_data.py`.

The KNN was implemented using the sklearn.neighbors package. The model was tested for 1 through
10 neighbors and the testing accuracy and testing time were plotted as a function of the number of neighbors.
The code for the KNN can be found in `kuzushiji_cnn.py`.

The FCN was implemented using the PyTorch package. The model has an adjustable number of layers and neurons per
layer, which allows for easy hyperparameter tuning to determine the effects of the number of neurons and
layers on accuracy and speed. It used a learning rate of 0.001, ADAM as its optimizer, Cross Entropy Loss
as the loss function, and was trained over 40 epochs. Batch normalization was applied after each layer, but
dropout regularization was not used. The code for the FCN can be found in `kuzushiji_fcn.py`.

The CNN was also implemented using PyTorch. It was built with two convolutional layers, each max-
pooled, and two linear layers. Batch normalization was applied at each step. For continuity with the FCN,
it also used a learning rate of 0.001, ADAM as its optimizer, Cross Entropy Loss as the loss function, and
was trained over 40 epochs. The code for the CNN can be found in `kuzushiji_cnn.py`.

# Results

In response to the target questions, the following conclusions were reached.
1. The CNN was the most accurate at about 5% more accurate than the FCN and KNN.
2. The FCN was the fastest algorithm. The CNN was twice
as fast as the KNN, and the FCN eight times as fast as
the CNN.
3. * Dimensionality reduction made the KNN algorithm feasible as it failed to train in a reasonable amount of time using full data.
   * The FCN was marginally faster and more accurate using reduced data.
   * The CNN failed to train using reduced data.

4. The addition of more neurons and more layers to the FCN
generally increased both runtime and accuracy.


# Acknowledgements

This assignment was completed as a cumulative final assignment for AMATH 582 with Prof. Shlizerman
at the University of Washington, Winter 2024. The full writeup of the assignment can be read in `report.pdf`.

The scripts in this project require the Kuzushiji-MNIST dataset (Version 3) https://www.kaggle.com/datasets/anokas/kuzushiji.
To run the scripts, download the data and extract all of it to a folder named Data
in the same directory.


## Attestation
"KMNIST Dataset" (created by CODH), adapted from "Kuzushiji Dataset" (created by NIJL and others), 
doi:10.20676/00000341
