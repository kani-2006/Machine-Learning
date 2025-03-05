# **Student Performance Prediction using Machine Learning**  

This project aims to predict student performance using various machine learning techniques, including linear regression, polynomial regression, ridge regression, lasso regression, logistic regression, and K-Nearest Neighbors (KNN). The models are evaluated using metrics like Mean Squared Error (MSE) and Accuracy.

## **Project Structure**  

- **`main project.py`** – Implements a linear regression model with a Gradio interface for student performance prediction.  
- **`lasso2.py`** – Uses Lasso regression to predict student performance and calculates Mean Squared Error.  
- **`linear2.py`** – Implements a simple Linear Regression model for prediction.  
- **`linearpoly2.py`** – Uses Polynomial Regression to enhance prediction accuracy.  
- **`logistic2.py`** – Implements Logistic Regression and visualizes student data using Matplotlib.  
- **`ridge2.py`** – Implements Ridge Regression, a regularized version of Linear Regression.  
- **`KN2.py`** – Uses K-Nearest Neighbors (KNN) classification to predict student performance.  
- **`final2.py`** – Generates a bar chart comparing the error rates of different models.  
- **`KANISHKA.py`** – Similar to `main project.py`, with a focus on Gradio-based interactive prediction.  
- **`Figure_2 final.png`** – Visualization of model performance comparison.  

## **Technologies Used**  
- **Python**  
- **Scikit-learn** – Machine learning models  
- **Pandas & NumPy** – Data processing  
- **Matplotlib** – Data visualization  
- **Gradio** – Web interface for predictions  

## **How to Run the Project**  

1. Install dependencies:  
   ```bash
   pip install numpy pandas scikit-learn matplotlib gradio
   ```
2. Run the main prediction interface:  
   ```bash
   python "main project.py"
   ```
3. For individual model testing, run any of the scripts (e.g., `python lasso2.py`).  

## **Results**  
- Different models are compared based on their error rates.  
- `final2.py` generates a bar chart to visualize performance differences.  

## **Contributors**  
- Kanishka K V
