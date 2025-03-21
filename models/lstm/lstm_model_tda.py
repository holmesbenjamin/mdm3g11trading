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
    df = pd.read_csv('datasets/WTICRUDEcombined_metrics_lists.csv')
    
    feature_columns = [
        'mean', 'median', 'std', 'var', 'min', 'max', 'range',
        'trend_slope', 'trend_intercept', 'r_squared',
        'skew', 'kurtosis', 'dominant_freq', 'spectral_entropy',
        'momentum', 'persistence_entropy_H1', 'weighted_entropy_H1'
    ]
    
    X = df[feature_columns].values
    print("X shape:", X.shape)
    
    returns = np.diff(df['mean'].values)
    print("returns shape:", returns.shape)
    
    quantiles = np.percentile(returns, [20, 40, 60, 80])
    y = np.digitize(returns, quantiles)
    print("y shape:", y.shape)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print("X_scaled shape:", X_scaled.shape)
    
    sequence_length = 20
    X_seq, y_seq = prepare_sequences(X_scaled, sequence_length)
    print("Before adjustment - X_seq shape:", X_seq.shape)
    print("Before adjustment - y_seq shape:", y_seq.shape)
    
    # Adjust X and y to have matching lengths
    X_seq = X_seq[:-1]  # Remove the last sequence
    y_seq = y[sequence_length:len(X_seq) + sequence_length]  # Adjust y_seq to match X_seq length
    
    print("After adjustment - X_seq shape:", X_seq.shape)
    print("After adjustment - y_seq shape:", y_seq.shape)
    
    split_idx = int(len(X_seq) * 0.8)
    X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
    y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]
    
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_train shape:", y_train.shape)
    print("y_test shape:", y_test.shape)
    
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
    
    print("y_test shape:", y_test.shape)
    print("y_pred_classes shape:", y_pred_classes.shape)
    
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
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_classes, 
          target_names=['Strong Down', 'Weak Down', 'Neutral', 'Weak Up', 'Strong Up']))
    
    plt.figure(figsize=(6, 5))
    cm = confusion_matrix(y_test, y_pred_classes)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()
    
    model.save('lstm_model_tda.h5')

if __name__ == "__main__":
    main()

# This model uses Topological Data Analysis (TDA) features including:
# - Persistence entropy and weighted entropy from H1 homology
# - Statistical features: mean, median, std, var, min, max, range
# - Time series features: trend, spectral properties, momentum
# The model architecture consists of 3 LSTM layers and 2 Dense layers 

#The accuracy of the model is around 60%

