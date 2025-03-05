from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
iris=pd.read_csv(r"C:\Users\kanis\Downloads\student data 5.csv")
x=iris.iloc[:,0:-1]
y=iris.iloc[:,-1:]
print(x)
print(y)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
poly=PolynomialFeatures(degree=2)
model=LinearRegression()
train_poly=poly.fit_transform(x_train)
test_poly=poly.transform(x_test)
model.fit(train_poly,y_train)
y_pred=model.predict(test_poly)
print(y_pred)
errordata=mean_squared_error(y_test,y_pred)
print(errordata)
