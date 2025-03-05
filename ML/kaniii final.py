from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import gradio as gr
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load data
file_path = r"C:\Users\kanis\Downloads\student data 5.csv"
df = pd.read_csv(file_path)

# Ensure dataset has enough columns
if df.shape[1] < 16:
    raise ValueError("Dataset does not have the expected 16 columns.")

X = df.iloc[:, 0:15]  # Features
y = df.iloc[:, 15]    # Target variable

# Standardize data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train linear regression model
model = LinearRegression()
model.fit(X_scaled, y)

# Prediction function
def predict_student(studentid, school, sex, age, traveltime, studytime, fails, nursery, higher, S1, S2, S3, S4, S5, MP):
    try:
        input_data = np.array([studentid, school, sex, age, traveltime, studytime, fails, nursery, higher, S1, S2, S3, S4, S5, MP])
        input_data = input_data.astype(float)  # Ensure numerical input
        input_data_reshape = input_data.reshape(1, -1)
        input_scaled = scaler.transform(input_data_reshape)
        model_pred = model.predict(input_scaled)
        return round(model_pred[0], 2)  # Returning rounded value for better readability
    except Exception as e:
        return str(e)

# Gradio interface
iface = gr.Interface(
    fn=predict_student,
    inputs=[
        gr.Number(label="Student ID"),
        gr.Number(label="School"),
        gr.Number(label="Sex"),
        gr.Number(label="Age"),
        gr.Number(label="Travel Time"),
        gr.Number(label="Study Time"),
        gr.Number(label="Fails"),
        gr.Number(label="Nursery"),
        gr.Number(label="Higher"),
        gr.Number(label="S1"),
        gr.Number(label="S2"),
        gr.Number(label="S3"),
        gr.Number(label="S4"),
        gr.Number(label="S5"),
        gr.Number(label="MP"),
    ],
    outputs="text",
    title="Student Performance Prediction Using Machine Learning",
)
iface.launch(share=True)

# Bar chart
plt.figure(figsize=(10, 5))
plt.bar(df.index, y, label="Student Performance", color="black")
plt.xlabel("Student Index")
plt.ylabel("Performance Score")
plt.legend()
plt.show()
