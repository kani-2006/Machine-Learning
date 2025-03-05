from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
import gradio as gr
from sklearn.preprocessing import StandardScaler
iris=pd.read_csv(r"C:\Users\kanis\Downloads\student data 5.csv")
x = iris.iloc[:,0:15]
y = iris.iloc[:,15]
scaler=StandardScaler()
x_scaled=scaler.fit_transform(x)
model=LinearRegression()
model.fit(x_scaled,y)
import matplotlib.pyplot as plt
plt.bar(x,y,label='bar chart',color='black')
plt.xlabel='x-axis'
plt.ylabel='y-axis'
plt.legend()
plt.show()
       


def predict_student(studentid,school,sex,age,traveltime,studytime,fails,nursery,higher,S1,S2,S3,S4,S5,MP):
    input_data=np.array([studentid,school,sex,age,traveltime,studytime,fails,nursery,higher,S1,S2,S3,S4,S5,MP])
    input_data_reshape=input_data.reshape(1,-1)
    input_scaled=scaler.transform(input_data_reshape)
    model_pred=model.predict(input_scaled)
    print(model_pred)
    return model_pred
    

iface=gr.Interface(
    fn=predict_student,
    inputs=[
       gr.Number(label="studentid"),
       gr.Number(label="school"),
       gr.Number(label="sex"),
       gr.Number(label="age"),
       gr.Number(label="traveltime"),
       gr.Number(label="studytime"),
       gr.Number(label="fails"),
       gr.Number(label="nursery"),
       gr.Number(label="higher"),
       gr.Number(label="S1"),
       gr.Number(label="S2"),
       gr.Number(label="S3"),
       gr.Number(label="S4"),
       gr.Number(label="S5"),
       gr.Number(label="MP")
       ],
    outputs="text",
    title="STUDENT PERFORMANCE USING MACHINE LEARNING AND HISTORICAL DATA",
    )
iface.launch(share=True)




       
