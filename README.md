# Fouling Prediction Using Machine Learning

## Introduction

This project focuses on predicting fouling in heat exchangers using machine learning techniques. Fouling in heat exchangers significantly reduces efficiency in petroleum processing. My goal was to develop an accurate and reliable predictive model that estimates the fouling factor (Rf) based on operational parameters such as flow rate, pressure, and temperature. This model can be used for predictive maintenance, reducing downtime and operational costs.

## Project Overview

In this project, I developed multiple machine learning models to predict the fouling factor in a network of heat exchangers. The dataset consists of process parameters including flow rate, pressure, temperature, and heat transfer coefficients, which serve as input variables. The output variable is the fouling factor (Rf), which indicates the level of deposition in the heat exchanger.

## Data Preprocessing

### 3.1 Data Collection and Preparation

The dataset was generated synthetically to simulate real-world refinery data. It includes:

- **Flow rate:** The volume of oil passing through the exchanger.
- **Pressure:** The pressure at the inlet and outlet.
- **Temperature:** The temperature at different points in the process.
- **Heat transfer coefficients:** Indicators of the efficiency of heat exchange on the shell and tube side.

### 3.2 Data Cleaning and Normalization

Before training the models, I cleaned the data to remove inconsistencies. To ensure a consistent scale across variables, I applied `MinMaxScaler`, normalizing values between 0 and 1. The dataset was then split into 80% training and 20% testing sets.

## Model Development

I developed and evaluated three machine learning models:

### 4.1 Backpropagation Neural Network (BPNN)

I implemented a fully connected neural network using TensorFlow. It consists of multiple dense layers with ReLU activation, optimized using Adam optimizer and mean squared error (MSE) loss function. The model was trained for 100 epochs with a batch size of 32.

### 4.2 Random Forest (RF)

To compare against neural networks, I trained a Random Forest model, which is known for handling nonlinear relationships between variables and preventing overfitting.

### 4.3 Long Short-Term Memory (LSTM)

Since fouling is a time-dependent process, I developed an LSTM model to capture sequential dependencies. The LSTM network was trained using reshaped time-series data, allowing it to learn patterns over time.

## Hybrid Model Implementation

After evaluating individual models, I applied a hybrid approach. I combined the predictions from BPNN, RF, and LSTM using a linear regression-based meta-model, also known as a stacked ensemble. This hybridization improved prediction accuracy by leveraging the strengths of multiple models.

## Model Evaluation

### 6.1 Performance Metrics

I measured each model’s performance using:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score (Coefficient of Determination)

The hybrid model achieved the highest R² score, indicating its superior predictive performance.

### 6.2 Visualization and Analysis

I generated graphs and charts to compare model predictions against actual values, including:

- Real vs. Predicted Graphs for each model.
- Metric Comparison Charts illustrating performance differences.

## How to Run the Project

### 7.1 Install Dependencies

pip install numpy pandas scikit-learn tensorflow matplotlib

###7.2 Execute the Pipeline

python main.py

The script will preprocess the data, train models, evaluate their performance, and save results in the /results/ directory.

##Conclusion

This machine learning-based approach provides an efficient way to predict fouling in heat exchangers, demonstrating the effectiveness of hybrid models in regression tasks. Future improvements may include hyperparameter tuning, feature selection techniques, and testing on real industrial datasets to further validate the model.

##References

Machine Learning for Predictive Maintenance.

Deep Learning for Regression Applications in Industry.

Optimization of Heat Exchanger Performance Using AI.
