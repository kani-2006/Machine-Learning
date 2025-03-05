from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
iris=pd.read_csv(r"C:\Users\kanis\Downloads\student data.csv")
x=iris.iloc[:,0:14]
y=iris.iloc[:,14:]
print(x)
print(y)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
model = KNeighborsClassifier()
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
print(y_pred)
accuracy = accuracy_score(y_test,y_pred)
print(accuracy)
cm = confusion_matrix(y_test,y_pred)
print(cm)
