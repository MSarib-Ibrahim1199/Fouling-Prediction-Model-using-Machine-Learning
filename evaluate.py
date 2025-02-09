import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_models(models, predictions, y_test):
    metrics = {}

    for name, y_pred in predictions.items():
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)

        metrics[name] = {'MAE': mae, 'RMSE': rmse, 'R²': r2}

        print(f'{name} - MAE: {mae}, RMSE: {rmse}, R²: {r2}')

    # Plot Real vs Predicted
    plt.figure(figsize=(15, 5))

    for i, (name, y_pred) in enumerate(predictions.items()):
        plt.subplot(1, 3, i + 1)
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')
        plt.title(f'{name} - Real vs Predicted')
        plt.xlabel('True Fouling Factor')
        plt.ylabel('Predicted Fouling Factor')

    plt.tight_layout()
    plt.show()

    return metrics
