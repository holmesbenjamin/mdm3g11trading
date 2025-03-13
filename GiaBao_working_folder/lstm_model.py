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
from ripser import ripser
from persim import wasserstein
from scipy.stats import skew, kurtosis

def compute_RSI(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    return 100 - (100 / (1 + rs))

def compute_MACD(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line - signal_line

def compute_BB_width(series, window=20, num_std=2):
    rolling_mean = series.rolling(window=window).mean()
    rolling_std = series.rolling(window=window).std()
    upper_band = rolling_mean + num_std * rolling_std
    lower_band = rolling_mean - num_std * rolling_std
    return (upper_band - lower_band) / (rolling_mean + 1e-10)

def compute_stochastic_oscillator(series, window=14):
    min_val = series.rolling(window=window).min()
    max_val = series.rolling(window=window).max()
    return 100 * (series - min_val) / (max_val - min_val + 1e-10)

def prepare_sequences(data, sequence_length):
    X, y = [], []
    for i in range(len(data) - sequence_length):
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

def plot_metrics(history, y_test, y_pred_classes):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    ax1.plot(history.history['accuracy'], label='Training')
    ax1.plot(history.history['val_accuracy'], label='Validation')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    
    cm = confusion_matrix(y_test, y_pred_classes)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2)
    ax2.set_title('Confusion Matrix')
    ax2.set_ylabel('True')
    ax2.set_xlabel('Predicted')
    
    plt.tight_layout()
    plt.show()

def main():
    df = pd.read_csv('mdm3g11trading/commodity_futures.csv')
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    commodity = 'GOLD'
    
    df = df[['Date', commodity]].dropna().sort_values('Date').reset_index(drop=True)
    df = df[df['Date'] <= pd.to_datetime('2023-08-04')].reset_index(drop=True)
    
    df['log_price'] = np.log(df[commodity])
    df['return'] = df['log_price'].diff()
    df['lag1_return'] = df['return'].shift(1)
    df['lag2_return'] = df['return'].shift(2)
    
    df['rsi'] = compute_RSI(df[commodity])
    df['macd'] = compute_MACD(df[commodity])
    df['bb_width'] = compute_BB_width(df[commodity])
    df['stochastic'] = compute_stochastic_oscillator(df[commodity])
    
    rolling_window = 20
    df['sma_20'] = df['log_price'].rolling(window=rolling_window).mean()
    df['sma_50'] = df['log_price'].rolling(window=50).mean()
    df['volatility'] = df['return'].rolling(window=rolling_window).std()
    df['momentum'] = df['log_price'] - df['log_price'].shift(10)
    
    df['price_acceleration'] = df['return'].diff()
    df['ma_cross'] = df['sma_20'] - df['sma_50']
    
    df['skewness'] = df['return'].rolling(window=rolling_window).apply(skew)
    df['kurtosis'] = df['return'].rolling(window=rolling_window).apply(kurtosis)
    
    df['day_of_week'] = df['Date'].dt.dayofweek
    df['month'] = df['Date'].dt.month
    
    feature_columns = [
        'return', 'lag1_return', 'lag2_return',
        'rsi', 'macd', 'bb_width', 'stochastic',
        'sma_20', 'sma_50', 'volatility', 'momentum',
        'price_acceleration', 'ma_cross',
        'skewness', 'kurtosis',
        'day_of_week', 'month'
    ]
    
    df = df.dropna()
    X = df[feature_columns].values
    
    quantiles = np.percentile(df['return'], [20, 40, 60, 80])
    y = np.digitize(df['return'], quantiles)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    sequence_length = 20
    X_seq, y_seq = prepare_sequences(X_scaled, sequence_length)
    y_seq = y[sequence_length:]
    
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
    plt.plot(history.history['loss'], label='Training')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    accuracy = accuracy_score(y_test, y_pred_classes)
    print("Test Accuracy (Multi-class): {:.2f}%".format(accuracy * 100))
    
    cm = confusion_matrix(y_test, y_pred_classes)
    print("Confusion Matrix:")
    print(cm)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_classes, 
          target_names=['Strong Down', 'Weak Down', 'Neutral', 'Weak Up', 'Strong Up']))
    
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, cmap='Blues')
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(5)
    plt.xticks(tick_marks, ['Strong Down', 'Weak Down', 'Neutral', 'Weak Up', 'Strong Up'], rotation=45)
    plt.yticks(tick_marks, ['Strong Down', 'Weak Down', 'Neutral', 'Weak Up', 'Strong Up'])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()
    
    model.save('lstm_model.h5')

if __name__ == "__main__":
    main() 


#The code uses features like RSI, MACD, Bollinger Bands, 
# Stochastic Oscillator, SMA, Volatility, Momentum,
#  Price Acceleration, MA Cross, Skewness, Kurtosis, Day of Week, Month.
#The model is a simple LSTM model with 3 layers of LSTM and 2 layers of Dense.
 