import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns
from scipy.stats import skew, kurtosis, linregress

def prepare_sequences(data, sequence_length):
    X, y = [], []
    for i in range(len(data) - sequence_length - 1):
        X.append(data[i:(i + sequence_length)])
        y.append(data[i + sequence_length])
    return np.array(X), np.array(y)

def create_lstm_model(input_shape, num_classes):
    model = Sequential([
        LSTM(128, input_shape=input_shape, return_sequences=True),
        Dropout(0.2),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    return model

def extract_features(window_data):
    window_mean = np.mean(window_data)
    window_median = np.median(window_data)
    window_std = np.std(window_data)
    window_var = np.var(window_data)
    window_min = np.min(window_data)
    window_max = np.max(window_data)
    window_range = window_max - window_min
    
    t_index = np.arange(len(window_data))
    slope, intercept, r_value, p_value, std_err = linregress(t_index, window_data)
    
    window_skew = skew(window_data)
    window_kurtosis = kurtosis(window_data)
    
    fft_vals = np.fft.fft(window_data)
    fft_freq = np.fft.fftfreq(len(window_data))
    idx = np.argmax(np.abs(fft_vals[1:])) + 1
    dominant_freq = fft_freq[idx]
    
    power_spectrum = np.abs(fft_vals) ** 2
    power_spectrum_norm = power_spectrum / np.sum(power_spectrum)
    spectral_entropy = -np.sum(power_spectrum_norm * np.log(power_spectrum_norm + 1e-12))
    
    momentum = window_data[-1] - window_data[0]
    
    return np.array([
        window_mean, window_median, window_std, window_var,
        window_min, window_max, window_range, slope,
        r_value**2, window_skew, window_kurtosis,
        dominant_freq, spectral_entropy, momentum
    ])

def main():
    df = pd.read_csv('mdm3g11trading/commodity_futures.csv')
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    commodity = 'GOLD'
    
    df = df[['Date', commodity]].dropna().sort_values('Date').reset_index(drop=True)
    df = df[df['Date'] <= pd.to_datetime('2023-08-04')].reset_index(drop=True)
    
    window_size = 20
    features = []
    for i in range(len(df) - window_size + 1):
        window_data = df[commodity].values[i:i+window_size]
        features.append(extract_features(window_data))
    
    features = np.array(features)
    
    returns = np.diff(np.log(df[commodity].values))[window_size-1:]
    quantiles = np.percentile(returns, [20, 40, 60, 80])
    y = np.digitize(returns, quantiles)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)
    
    sequence_length = 20
    X_seq, y_seq = prepare_sequences(X_scaled, sequence_length)
    y_seq = y[sequence_length:len(X_seq) + sequence_length]
    
    print(f"X_seq shape: {X_seq.shape}")
    print(f"y_seq shape: {y_seq.shape}")
    
    split_idx = int(len(X_seq) * 0.8)
    X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
    y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]
    
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = create_lstm_model(input_shape, 5)
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("Model summary:")
    model.summary()
    
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=50,
        batch_size=32
    )
    
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    cm = confusion_matrix(y_test, y_pred_classes)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.show()
    
    accuracy = accuracy_score(y_test, y_pred_classes)
    print("\nTest Accuracy: {:.2f}%".format(accuracy * 100))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_classes,
          target_names=['Strong Down', 'Weak Down', 'Neutral', 'Weak Up', 'Strong Up']))
    
    model.save('lstm_model_conventional.h5')

if __name__ == "__main__":
    main() 
    

#The code is a simple LSTM model for commodity csv.
#  It uses a sliding window to extract features from the data and a LSTM model
#  to predict the future direction of the price.

#the accuracy of the model is low, around 21-24%
