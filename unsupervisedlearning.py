
"""
UNSUPERVISED LEARNING PART 1 - LOGISTIC REGRESSION
"""

#Importing the neccessary Libraries for the moment

import pandas as pd
import csv

"""I have my data from the Breast Cancer Wisconsin (Diagnostic) Data Set
link: https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data
"""

df=pd.read_csv('data.csv')
df

"""Below I have began my logistic regression where I have take all the results from the second column, which is either Malignant or Benign and put it as my label vector (y). Then I created my feature matrix (x) on the rest of the other neccessary information."""

# Extract the label vector (y) from the second column
y = df[df.columns[1]]

# Extract the feature matrix (X) from the rest of the columns
X = df.iloc[:, 2:32]

"""I used the 80/20 rule to train and split my data to ensure the most accurate results. The test_size 0.2 reffers to the testing size from the data set which is 20 percent. I chose a random_state of 10 because anything more or less would be either not volatile enough, or too volatile and disrupt the machine learning"""

# 80% training data, 20% training data

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=10)

"""In order to actually do the machine learning you have to initialize the machine learned for Logistic Regression."""

from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression(random_state = 10)

"""Now I will fit the training data into the previously called upon and assigned machine learning"""

logreg.fit(X_train, y_train)

"""y_pred is the prediction that the machine has done against 20% of the dataset and stores whether is was correct or incorrect"""

y_pred = logreg.predict(X_test)

"""For demonstration purposes, I have included to display the confusion matrix as this may be helpful for other to view how it is impossible to have an 100% accuracy rate. This woukd mean that the array below would not have any values outside the diagonal of the array"""

from sklearn import metrics

cnf_matrix = metrics.confusion_matrix(y_test,y_pred)

cnf_matrix #IF NUMBERS ONLY IN DIAGONAL ITS CORRECT

"""This will use a pre-existing function to get the accuracy of the machine learned once completed. in my case in was 93% accurate"""

print('Accuracy', metrics.accuracy_score(y_pred, y_test))

"""This following code will provide a 3D plot that the user can interact with to show all of the necessary information and help the user understand how the machine visualizes the data."""

import plotly.express as px

# Specify the columns for the 3D scatter plot
scatter_cols = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean']

# Create a 3D scatter plot
fig = px.scatter_3d(df, x='radius_mean', y='texture_mean', z='perimeter_mean', size='area_mean', color='diagnosis', opacity=0.7)

# Update layout for better visualization
fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))

# Show the plot
fig.show()

"""UNSUPERVISED LEARNING PART 2 - KMEANS CLUSTERING

I will now take the same dataset but use a different method of unsupervised learning, Kmeans Clustering. This is a different way to use unsupervise machine learning as this method, will split the data into clusters, and continously calculate the mean of the cluster until it believes it has reached the most accuract mean.
"""

#Importing the necessary libraries

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

"""The code below will seperate the data to be split into 2 clusters. Then using the X value that we initialized in the code earlier on it will use machine learning to find the most accuracte results."""

kmeans = KMeans(n_clusters=2)
kmeans.fit(X)

"""This code below will get the labels of 1's and 0's which represents the cluster assignments for each data point after the KMeans algorithm has been fit to the data. It is an array where each element corresponds to the cluster label assigned to the corresponding data point. Then I will use the cluster centers functions to get the coordinates of the centroids for each cluster in feature space


"""

labels = kmeans.labels_
centroids = kmeans.cluster_centers_

"""Then I will use a scatter plot to demonstrate how the machine used machine learning to seperate the data and where the centroids are located in regards to all the data"""

plt.scatter(X.iloc[:,0], X.iloc[:,1], c=labels, cmap = 'viridis')
plt.scatter(centroids[:,0], centroids[:,1], s=300, c='red', marker = 'x')