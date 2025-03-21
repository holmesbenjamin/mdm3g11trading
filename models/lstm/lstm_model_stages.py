import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler, RobustScaler
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns
import sys
import os

# Add the parent directory to sys.path to import from sibling directories
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Ben_working_folder.featuresFromCSV import extract_stage_1, extract_stage_2, extract_stage_3, extract_stage_4

def prepare_sequences(data, sequence_length):
    X, y = [], []
    for i in range(len(data) - sequence_length - 1):
        X.append(data[i:(i + sequence_length)])
        y.append(data[i + sequence_length])
    return np.array(X), np.array(y)

def create_lstm_model(input_shape, num_classes):
    model = Sequential([
        # Original architecture
        LSTM(256, input_shape=input_shape, return_sequences=True),
        Dropout(0.2),
        
        LSTM(128, return_sequences=True),
        Dropout(0.2),
        
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        
        Dense(128, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    return model

def plot_metrics(history, y_test, y_pred_classes, stage):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    ax1.plot(history.history['accuracy'], label='Training')
    ax1.plot(history.history['val_accuracy'], label='Validation')
    ax1.set_title(f'Stage {stage} Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    
    cm = confusion_matrix(y_test, y_pred_classes)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2)
    ax2.set_title(f'Stage {stage} Confusion Matrix')
    ax2.set_ylabel('True')
    ax2.set_xlabel('Predicted')
    
    plt.tight_layout()
    plt.show()

def train_and_evaluate(X, y, stage_num):
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Prepare sequences
    sequence_length = 20
    X_seq, y_seq = prepare_sequences(X_scaled, sequence_length)
    y_seq = y[sequence_length:len(X_seq) + sequence_length]
    
    # Split data
    split_idx = int(len(X_seq) * 0.8)
    X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
    y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]
    
    # Create and compile model
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = create_lstm_model(input_shape, 5)
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"\nStage {stage_num} Model Summary:")
    model.summary()
    
    # Train model with normal batch size
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=50,
        batch_size=32  # Back to original batch size
    )
    
    # Evaluate model
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Plot results
    plot_metrics(history, y_test, y_pred_classes, stage_num)
    
    # Print metrics
    accuracy = accuracy_score(y_test, y_pred_classes)
    print(f"\nStage {stage_num} Test Accuracy: {accuracy * 100:.2f}%")
    print(f"\nStage {stage_num} Classification Report:")
    print(classification_report(y_test, y_pred_classes,
          target_names=['Strong Down', 'Weak Down', 'Neutral', 'Weak Up', 'Strong Up']))
    
    # Save model
    model.save(f'lstm_model_stage_{stage_num}.h5')
    
    return accuracy

def process_dataframe(df):
    # Function to safely convert string lists to float
    def convert_string_list(x):
        if isinstance(x, str) and x.startswith('[') and x.endswith(']'):
            # Convert string representation of list to actual list and get the first value
            try:
                return float(eval(x)[0])  # Take first value from the list
            except:
                return 0.0
        return x

    # Process each column
    for col in df.columns:
        if col != 'window_start':
            # Check if the column contains string lists
            if isinstance(df[col].iloc[0], str) and df[col].iloc[0].startswith('['):
                df[col] = df[col].apply(convert_string_list)
            else:
                df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df

def main():
    # Load commodity data for returns calculation
    df = pd.read_csv('commodity_futures.csv')
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    commodity = 'GOLD'
    df = df[['Date', commodity]].dropna().sort_values('Date').reset_index(drop=True)
    
    # Calculate returns for labels
    returns = np.diff(np.log(df[commodity].values))
    quantiles = np.percentile(returns, [20, 40, 60, 80])
    y = np.digitize(returns, quantiles)
    
    # Load features from all stages
    stage_accuracies = {}
    
    # Stage 1: LSTM Only (Price data)
    print("\nProcessing Stage 1: LSTM Only (Price data)")
    stage1_df = extract_stage_1('combined_metrics_lists.csv')
    stage1_df = process_dataframe(stage1_df)
    min_length = min(len(y), len(stage1_df))
    y_stage1 = y[:min_length]
    stage1_df = stage1_df.iloc[:min_length]
    X_stage1 = stage1_df.drop('window_start', axis=1).values
    X_stage1 = np.nan_to_num(X_stage1)
    stage_accuracies[1] = train_and_evaluate(X_stage1, y_stage1, 1)
    
    # Stage 2: LSTM + Technical Indicators
    print("\nProcessing Stage 2: LSTM + Technical Indicators")
    stage2_df = extract_stage_2('combined_metrics_lists.csv')
    stage2_df = process_dataframe(stage2_df)
    min_length = min(len(y), len(stage2_df))
    y_stage2 = y[:min_length]
    stage2_df = stage2_df.iloc[:min_length]
    X_stage2 = stage2_df.drop('window_start', axis=1).values
    X_stage2 = np.nan_to_num(X_stage2)
    stage_accuracies[2] = train_and_evaluate(X_stage2, y_stage2, 2)
    
    # Stage 3: LSTM + Technical Indicators + TDA Betti Curves
    print("\nProcessing Stage 3: LSTM + Technical Indicators + TDA Betti Curves")
    stage3_df = extract_stage_3('combined_metrics_lists.csv')
    stage3_df = process_dataframe(stage3_df)
    min_length = min(len(y), len(stage3_df))
    y_stage3 = y[:min_length]
    stage3_df = stage3_df.iloc[:min_length]
    X_stage3 = stage3_df.drop('window_start', axis=1).values
    X_stage3 = np.nan_to_num(X_stage3)
    stage_accuracies[3] = train_and_evaluate(X_stage3, y_stage3, 3)
    
    # Stage 4: LSTM + Technical Indicators + TDA Betti Curves + TDA Entropy
    print("\nProcessing Stage 4: All Features (including TDA entropy)")
    stage4_df = extract_stage_4('combined_metrics_lists.csv')
    stage4_df = process_dataframe(stage4_df)
    min_length = min(len(y), len(stage4_df))
    y_stage4 = y[:min_length]
    stage4_df = stage4_df.iloc[:min_length]
    X_stage4 = stage4_df.drop('window_start', axis=1).values
    X_stage4 = np.nan_to_num(X_stage4)
    stage_accuracies[4] = train_and_evaluate(X_stage4, y_stage4, 4)
    
    # Compare results
    print("\nAccuracy comparison across all stages:")
    for stage, acc in stage_accuracies.items():
        print(f"Stage {stage}: {acc * 100:.2f}%")
    
    # Plot comparison
    plt.figure(figsize=(12, 6))
    stages = list(stage_accuracies.keys())
    accuracies = [stage_accuracies[s] * 100 for s in stages]
    
    bars = plt.bar(stages, accuracies)
    plt.title('LSTM Model Accuracy Across Feature Stages')
    plt.xlabel('Stage')
    plt.ylabel('Accuracy (%)')
    plt.ylim(0, 100)
    
    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom')
    
    # Add stage descriptions
    plt.figtext(0.02, -0.1, "Stage 1: Price data only", wrap=True, horizontalalignment='left', fontsize=8)
    plt.figtext(0.02, -0.15, "Stage 2: Price data + Technical Indicators", wrap=True, horizontalalignment='left', fontsize=8)
    plt.figtext(0.02, -0.2, "Stage 3: Price data + Technical Indicators + TDA Betti Curves", wrap=True, horizontalalignment='left', fontsize=8)
    plt.figtext(0.02, -0.25, "Stage 4: All features (including TDA entropy)", wrap=True, horizontalalignment='left', fontsize=8)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.3)  # Adjust to make room for stage descriptions
    plt.show()

if __name__ == "__main__":
    main()

#very low accuracy for all stages

