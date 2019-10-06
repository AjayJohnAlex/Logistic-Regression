# importing libraries
import numpy as np


# import data 
from sklearn.datasets import load_breast_cancer
# load dataset
data  = load_breast_cancer()

# print these commands to get an idea of the dataset
# print(type(data))
# print(data.DESCR)
# print(data.data.shape)
# print(data.target)

from sklearn.model_selection import train_test_split

X = data.data
y = data.target

# Spilting the data into train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42) 

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier()

rfc.fit(X_train,y_train)

print('Train Score ',rfc.score(X_train,y_train))

print('Test score ' ,rfc.score(X_test,y_test))

predictions = rfc.predict(X_test)
# Manual prediction of accuracy 
N = len(y_test)
print("Manual Prediction ",np.sum(predictions == y_test)/N)

from sklearn.metrics import accuracy_score

print("Accuracy Score" ,accuracy_score(y_test,predictions))

# using a Deep Learning model
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

# Need to give the NN classifier a parameter #max_iter=500 (It didnt converge as NN is need more time)
# ERROR : ConvergenceWarning: Stochastic Optimizer: Maximum iterations (200) reached and the
# optimization hasn't converged yet.

mlp = MLPClassifier(max_iter=500)
standardScale = StandardScaler()

# scaling data using Standard Scaler
X_train2 = standardScale.fit_transform(X_train)
X_test2 = standardScale.transform(X_test)

# training data for NN 
mlp.fit(X_train2,y_train)

print("NN train scsore",mlp.score(X_train2,y_train))
print("NN test scsore",mlp.score(X_test2,y_test))

# NN predictions
mlp_pred = mlp.predict(X_test2)

print("NN prediction score",accuracy_score(y_test,mlp_pred))