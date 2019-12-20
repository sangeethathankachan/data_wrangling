#First import your necessary packages
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Now read the dataset
Dataset = pd.read_csv('Data.csv')
#Dataset = Dataset.drop('Name', axis=1)

X = Dataset.iloc[:, :-1].values
Y = Dataset.iloc[:, 3].values

print(X)
print("\n")

# Import the necessary packages
from sklearn.impute import SimpleImputer

# Fitting missing values
imputer = SimpleImputer(strategy='mean', fill_value='NaN')
imputer = imputer.fit( X[ : , 1:3])
X[ : , 1:3] = imputer.transform(X[ : , 1:3 ])
print(X)
print("\n")


# Import the necessary packages
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer, make_column_transformer

# Dummy Encoding
onehotencoder = make_column_transformer(
    (OneHotEncoder(),[0]),remainder="passthrough")
X = onehotencoder.fit_transform(X)
X = np.array(X)
print(X)
print("\n")



# Import the necessary packages
from sklearn.model_selection import train_test_split

# Splitting the data into training data and test data
X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size = 0.2,
random_state=0)

print("Input for training: \n {}".format(X_train))
print("\n")

print("Output for training: \n {}".format(Y_train))
print("\n")

print("Input for testing: \n {}".format(X_test))
print("\n")

print("Output for testing: \n {}".format(Y_test))
print("\n")


# Import the necessary packages
from sklearn.preprocessing import StandardScaler

# Feature Scaling
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)

print("X_Train after Feature Scaling:\n {}".format(X_train))
print("\n")

print("X_Test after Feature Scaling:\n {}".format(X_test))
