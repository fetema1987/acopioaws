# acopioaws


## Repository Structure

The repository is structured as follows:
 - /figures - folder containing project images.
 - /src/acopio-train.py - python file containing the dataset class.
 - /src/Train.ipynb- python file containing the train class.


## About
 - Date: 01/11/2020
 - Author: Fernando Tébar Martínez

## Motivation

Currently, when we receive a sales order, we connect to a web application and manually select the gathering store from where we will pick the items in the order. With this algorithm we try to automate the picking by classifying in real time the items in one of the stores based on historical manual assignments.

## Reach
 - [x] Analyze the dataset and perform feature selection in order to feed it optimally into the network
 - [x] Mitigate the heavy class imbalance
 - [x] Obtain great accuracy
 - [x] Train on a SageMaker notebook instance
 - [x] Deploy the model on a hosted endpoint in AWS SageMaker

## Dataset

The dataset was downloaded from the server and consists in 60477 sales order items captured from 2018 until mid-september 2020. Each observation contains 171 features, including several dates, univocal and categorical.

### Distribution

We plot the class distribution histogram and see a big class imbalance. Few classes have huge ocurrences while some have very few.

![Initial distribution](https://github.com/fetema1987/acopioaws/blob/main/figures/dist_inicial.png)

This IS a thing. We must tackle this imbalance by performing certain operations on our dataset. Such imbalance would end up for our classifier giving brilliant accuracy but really poor F1 score on a majority of the classes. The optimizer will try to minimize error and will do this by focusing in classes with more ocurrences and ignoring the ones with low appearances.
We perform feature resampling by undersampling the top appearing classes and oversampling the low ones. After this we have a dataset of 69857 lines and the distribution is as follows.

![Distribution after resampling](https://github.com/fetema1987/acopioaws/blob/main/figures/dist_oversampled.png)

### Feature selection

The features are analyzed to capture the insights of the data. Moreover, feature cleaning and transformation have to be performed so we can extract the maximum profit from our model.

We see that from the original feature space of 171 elements we have 84 empty columns we can drop. Features with a value of 0.0 in all observations are 26 and can be dropped too. Also, we drop unique and univocal value features.
After cleaning the feature space we have just 35 features. Still, we have many date and categorical features that need some transformation.
Date columns are dismissed the year and date and we create new features for the month of each original date feature. 
Finally, as this is SAP data, we have lots of categorical features. We have to perform one-hot encoding on them resulting in a new feature space of 267 features.

Finally, we execute label encoding. The classes are categorical ('Txyz') and need to be parsed into a numerical feature.

### Split validation

To evaluate our model we split the dataset into three parts. Training, validation and test splits will contain 70, 15 and 15% of the data. 

![Class stats](https://github.com/fetema1987/acopioaws/blob/main/figures/dist_splits.png)

## Architecture
### Network

The network consists in a deep neural network with an entry layer of 267 neurons (number of features), one hidden layer of 64 neurons and an output layer of 42 neurons (number of classes). Both entry and hidden have a ReLU activation layer afterwards.
 

### Optimizer

Optimizer: RMSprop
Loss function: Categorical cross entropy

### Metrics

Evaluating and comparing the experiments is a nuclear part of the scientific work and opens the path to adjust parameters and propose changes. Using these metrics it is possible to check if the model is over/underfitting or if during the process something has failed and ultimately compare trainings

#### F1 score (per class)
This has been the metric of reference when checking how good the model was at predicting each one of the classes, specially thos with low appearances. This is a number between 0 and 1 calculated with the predicted results  from the confusion matrix.

#### mF1

The other metric that helped ensuring the quality of the model was the mean F1 score of all the classes. While accuracy tend to deceive us in imbalanced datasets, this result, along with the F1 class table reveals us how the training went.

#### Accuracy

The model accuracy is calculated dividing the number of correct predictions by the number of total predictions. 

#### AWS environment
The process of training this model has been executed in a ml.m3.medium notebook instance in AWS Sagemaker. Other alternatives tried were using an XGBoost estimator from the Sagemaker SDK and similar results were obtained, almost 100% accuracy and mF1. Some of the built-in algorithms in SageMaker such as k-nn or Linear Learner with multiclass hyperparameter would also do the job.

## Results

In the first stages of development we got pretty good results with an accuracy of 99.59% and a mF1 of 0.9938. 
