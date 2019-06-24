# Artificial Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# Install Tensorflow from the website: https://www.tensorflow.org/versions/r0.12/get_started/os_setup.html

# Installing Keras
# pip install --upgrade keras

# Part 1 - Data Preprocessing
#artifical Neural Network

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')

X = dataset.iloc[:, 3:13].values #upper bound is discluded 
y = dataset.iloc[:, 13].values 

#must encode the categorical variables in input set (Geography and gender)

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
labelencoder_a = LabelEncoder()
X[:, 1] = labelencoder_a.fit_transform(X[:,1])
labelencoder_b = LabelEncoder()
X[:, 2] = labelencoder_b.fit_transform(X[:,2])
onehot_a = OneHotEncoder(categorical_features = [1]) #categorical_features chooses which column to transform
onehot_b = OneHotEncoder(categorical_features = [2])
X = onehot_a.fit_transform(X).toarray()
X = onehot_b.fit_transform(X).toarray()
X = X[:, 1:]

#forming training and test set

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Feature Scaling

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#Lets makwe the Artifical Neural Network!

import keras
from keras.models import Sequential
from keras.layers import Dense

#Initializing ANN

classifier = Sequential() #we will define the neural network as sequence of layers

#Adding first input layer and hidden layer using Dense 
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 12))
#adding the number of input nodes, initializng weights and activation functions, and how many nodes in hiddnen layer

#adding another hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
#we don't need to specify input dum as network is already initialized


#adding final output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
#for outputs with more than 2 categories, increase units and change activation to softmax


#Applying back progopagtion and  stochastic gradient descent to ANN

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
#loss is the type of cost function, for multiple categories it would be categorical_crossentropy and metrics is performacnce criterion we are measuring it against

#Fit to training set
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)
#batch size is how many training examples to go through before updating set




y_pred = classifier.predict(X_test) #y_pred is probabilities they will or will not leave bank
y_pred = (y_pred > 0.5) #return 1 if probability is over 50%

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)




























