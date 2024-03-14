# -*- coding: utf-8 -*-
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt



data = pd.read_csv("newdataset.csv")

data.head(50)

pd.DataFrame(data.dtypes, columns=['DataType'])

print(data.keys())

X = data.drop(columns=['class_bot'])
y = data["class_bot"]

X

y

from sklearn.model_selection import train_test_split



#test_size set to 20% testing 75% training
#to get the same data random state=0
#To split both training and test datasets with the same class ratio we can also add stratify=y
#if we don't add stratify there might not be balanced class distribution

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25, random_state=0,stratify=y)

print(X_train.shape)
print(y_train.shape)

"""# Data Scaling

We can have values in the dataset with different metrics

some variables might be between 1 and 10 and others might be between 1 to 100 so we need to normalize those variables between 0 and 1

MinMax formula:

x_norm={x-x_min}/{x_max-x_min}
"""

from sklearn.preprocessing import MinMaxScaler

"""
Normalization
"""

#an instance of the MinMaxScaler class
samplenorm = MinMaxScaler()

#Train the parameters and transform the data
X_train_mm = samplenorm.fit_transform(X_train)

#Do the transform for X_test
X_test_mm = samplenorm.transform(X_test)

"""
## Standardization

Mean is 0 and sd=1

All the columns are will have standard distribution

x_std={x-mean}/standard error
"""

from sklearn.preprocessing import StandardScaler

samplestd = StandardScaler()

X_train_std = samplestd.fit_transform(X_train)

X_test_std = samplestd.transform(X_test)

"""## Linear Regression

If the performance of the model is good in the training but bad in test then there is overfitting

If there is overfitting we can't generalize the model to predict the new data

The model will predict the training data good and will have difficulty to predict an unseen data.

To solve overfitting
- Collect more data
- Regularization
- Simple model
- Reduce the data size

"""

from sklearn.linear_model import LogisticRegression

"""Regularization Approach

l1, l2 regularization methods reduce model complexity

Those regularization methods penalize the large coefficients learned by the model penalized to reduce complexity of the model

L1 regularization promotes sparsity by setting some coefficients to exactly zero, while L2 regularization penalizes large coefficient values without necessarily eliminating them entirely. L1 regularization make feature coefficients closer to zero than L2.

When dealing with a dataset with a large number of features, some of those attributes may have little influence on the target variable. In such cases, using L1 regularization can help in feature selection by effectively setting the coefficients of less important features to zero. This results in a simpler and more interpretable model, as it eliminates irrelevant features from consideration.



"""

#By default C=1 smaller the C parameter the more the coefficient approach 0.
#smaller values specify stronger regularization. C=1 indicates moderate regularization.
#Increasing C reduces the strength of regularization, potentially leading to overfitting
#if the data is noisy or if there are many irrelevant features.
#liblinear is good for small to medium dataset
#ovr is one versus rest: logistic regression model is fit for each class
#separately, treating it as the positive class while the other classes are grouped together as the negative class.
#This strategy is appropriate when dealing with binary classification tasks or when the number of classes is large
lr = LogisticRegression(penalty = "l1", C = 1, solver = "liblinear",
                       multi_class="ovr")

lr.fit(X_train_std, y_train)

print(lr.score(X_train_std, y_train))
print(lr.score(X_test_std, y_test))

"""#K-fold Cross Validation

"""

from sklearn.model_selection import cross_val_score

scores=cross_val_score(lr, X_train_std, y_train, cv=5)

scores

import numpy as np

np.mean(scores)

"""SVM"""

from sklearn.svm import SVC

classifier = SVC(kernel='linear', random_state=0)
classifier.fit(X_train_std, y_train)

Y_Pred = classifier.predict(X_test_std)

Y_Pred

from sklearn import metrics

print("Accuracy with kernel= linear")

print(metrics.accuracy_score(y_test, Y_Pred))

"""SVM (kernel='rbf"""

classifier1=SVC(kernel='rbf')
classifier1.fit(X_train_std, y_train)

Y_Pred1 = classifier.predict(X_test_std)

Y_Pred1

print("Accuracy with kernel= rbf")

print(metrics.accuracy_score(y_test, Y_Pred1))

