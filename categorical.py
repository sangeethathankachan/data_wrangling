# Importing the Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import dataset
dataset = pd.read_csv("Data.csv")
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,3].values

#Taking care of Missing data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer = imputer.fit(X[:,1:3])
X[:,1:3] = imputer.transform(X[:,1:3])
print(X)

#Encoding Categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder,StandardScaler
from sklearn.compose import ColumnTransformer
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
transformer = ColumnTransformer([('one_hot_encoder', OneHotEncoder(), [0])],remainder='passthrough')
#X = np.array(transformer.fit_transform(X), dtype=np.int)
X = transformer.fit_transform(X.tolist())
#X=X[:,2:]
labelencoder_Y = LabelEncoder()

Y = labelencoder_Y.fit_transform(Y)
print(X) 
print(Y)

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size = 0.2,
random_state=0)


sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)

print(X_train)


