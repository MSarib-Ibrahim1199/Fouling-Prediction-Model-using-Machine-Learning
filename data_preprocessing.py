import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def load_and_preprocess_data():
    np.random.seed(42)
    n_samples = 1000

    flow_rate = np.random.uniform(50, 200, n_samples)
    pressure = np.random.uniform(1, 10, n_samples)
    temperature = np.random.uniform(50, 200, n_samples)
    heat_transfer_shell = np.random.uniform(100, 500, n_samples)
    heat_transfer_tube = np.random.uniform(100, 500, n_samples)

    fouling_factor = (
        0.5 * flow_rate +
        0.3 * pressure +
        0.2 * temperature +
        0.1 * heat_transfer_shell +
        0.1 * heat_transfer_tube +
        np.random.normal(0, 10, n_samples)
    )

    data = pd.DataFrame({
        'flow_rate': flow_rate,
        'pressure': pressure,
        'temperature': temperature,
        'heat_transfer_shell': heat_transfer_shell,
        'heat_transfer_tube': heat_transfer_tube,
        'fouling_factor': fouling_factor
    })

    X = data.drop(columns=['fouling_factor'])
    y = data['fouling_factor']

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test
