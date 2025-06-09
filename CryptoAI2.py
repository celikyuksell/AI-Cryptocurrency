import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Layer

# --- Load and clean the new dataset ---
df = pd.read_csv("merged_crypto_data.csv")  # Replace with your actual path

# Convert Date to datetime (optional for plotting or time-based splitting)
df['Date'] = pd.to_datetime(df['Date'])

# Filter required columns
df = df[['Open', 'High', 'Low', 'Close', 'Volume']]  # Rename matches already

# Handle missing values or zero volume rows if needed
df = df[df['Volume'] > 0].dropna()

# --- Normalize ---
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)

# --- Create sequences ---
SEQ_LEN = 60
def create_sequences(data, seq_len):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len, 3])  # Predict 'Close'
    return np.array(X), np.array(y)

X, y = create_sequences(scaled_data, SEQ_LEN)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# --- Attention Layer ---
class Attention(Layer):
    def build(self, input_shape):
        self.W = self.add_weight(name='att_weight', shape=(input_shape[-1], 1),
                                 initializer='random_normal', trainable=True)
        self.b = self.add_weight(name='att_bias', shape=(input_shape[1], 1),
                                 initializer='zeros', trainable=True)
        super().build(input_shape)

    def call(self, x):
        e = tf.keras.backend.tanh(tf.keras.backend.dot(x, self.W) + self.b)
        a = tf.keras.backend.softmax(e, axis=1)
        output = x * a
        return tf.keras.backend.sum(output, axis=1)

# --- Build Model ---
def build_lstm_attention_model(input_shape):
    inputs = Input(shape=input_shape)
    x = LSTM(64, return_sequences=True)(inputs)
    x = Dropout(0.2)(x)
    x = LSTM(64, return_sequences=True)(x)
    x = Dropout(0.2)(x)
    attention = Attention()(x)
    output = Dense(1)(attention)
    model = Model(inputs, output)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

model = build_lstm_attention_model((SEQ_LEN, df.shape[1]))
model.summary()

# --- Train ---
history = model.fit(X_train, y_train, epochs=20, batch_size=64,
                    validation_data=(X_test, y_test), verbose=1)

# --- Predict and Evaluate ---
y_pred = model.predict(X_test)

# Reverse scale
def reverse_scale(values):
    filler = np.zeros((len(values), df.shape[1]))
    filler[:, 3] = values.reshape(-1)
    return scaler.inverse_transform(filler)[:, 3]

y_test_rescaled = reverse_scale(y_test)
y_pred_rescaled = reverse_scale(y_pred)

# --- Metrics ---
mse = mean_squared_error(y_test_rescaled, y_pred_rescaled)
mae = mean_absolute_error(y_test_rescaled, y_pred_rescaled)
r2 = r2_score(y_test_rescaled, y_pred_rescaled)

print(f"✅ MSE: {mse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")

# --- Plot Results ---
plt.figure(figsize=(14, 6))
plt.plot(y_test_rescaled, label='Actual Close Price')
plt.plot(y_pred_rescaled, label='Predicted Close Price')
plt.title("LSTM + Attention - AAVE Price Prediction")
plt.xlabel("Time")
plt.ylabel("Close Price")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
