from sklearn.linear_model import LinearRegression,Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
iris=pd.read_csv(r"C:\Users\kanis\Downloads\student data 5.csv")
x=iris.iloc[:,0:-1]
y=iris.iloc[:,-1:]
print(x)
print(y)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
ridge_model=Ridge(alpha=1.0)
ridge_model.fit(x_train,y_train)
y_pred=ridge_model.predict(x_test)
print(y_pred)
errordata=mean_squared_error(y_test,y_pred)
print(errordata)
