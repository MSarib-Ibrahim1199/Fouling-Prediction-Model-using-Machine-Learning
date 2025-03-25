from data_preprocessing import load_and_preprocess_data
from train import train_models
from evaluate import evaluate_models

X_train, X_test, y_train, y_test = load_and_preprocess_data()

models, predictions = train_models(X_train, X_test, y_train, y_test)

evaluate_models(models, predictions, y_test)
