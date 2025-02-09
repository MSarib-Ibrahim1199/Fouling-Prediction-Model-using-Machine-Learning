from models import build_bpnn, build_rf, build_lstm
import numpy as np

def train_models(X_train, X_test, y_train, y_test):
    models = {}

    # Train BPNN
    bpnn = build_bpnn(X_train.shape[1])
    bpnn.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), verbose=0)
    models['BPNN'] = bpnn
    y_pred_bpnn = bpnn.predict(X_test)

    # Train Random Forest
    rf = build_rf()
    rf.fit(X_train, y_train)
    models['RandomForest'] = rf
    y_pred_rf = rf.predict(X_test)

    # Train LSTM (Requires reshaping)
    X_train_lstm = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test_lstm = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

    lstm = build_lstm((X_train_lstm.shape[1], X_train_lstm.shape[2]))
    lstm.fit(X_train_lstm, y_train, epochs=100, batch_size=32, validation_data=(X_test_lstm, y_test), verbose=0)
    models['LSTM'] = lstm
    y_pred_lstm = lstm.predict(X_test_lstm)

    return models, {'BPNN': y_pred_bpnn, 'RandomForest': y_pred_rf, 'LSTM': y_pred_lstm}
