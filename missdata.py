import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

ds=pd.read_csv('Data.csv')
X=ds.iloc[:,:-1].values
y=ds.iloc[:,3].values

from sklearn.impute import SimpleImputer
imp=SimpleImputer(missing_values=np.nan,strategy='mean')
imp.fit(X[:,1:3])
X[:,1:3]=imp.transform(X[:,1:3])
print(X)
