from data_preprocessing import load_and_preprocess_data
from train import train_models
from evaluate import evaluate_models

# Load dataset and preprocess it
X_train, X_test, y_train, y_test = load_and_preprocess_data()

# Train all models
models, predictions = train_models(X_train, X_test, y_train, y_test)

# Evaluate and visualize results
evaluate_models(models, predictions, y_test)
