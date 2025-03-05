from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
iris=pd.read_csv(r"C:\Users\kanis\Downloads\student data 5.csv")
x = iris.iloc[:,0:-1]
y = iris.iloc[:,-1:]
print(x)
print(y)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
model=LinearRegression()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
print(y_pred)

errordata=mean_squared_error(y_test,y_pred)
print(errordata)
