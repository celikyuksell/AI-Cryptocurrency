# CryptoAI: Deep Learning for Cryptocurrency Price Prediction

CryptoAI is a deep learning project that predicts future closing prices of cryptocurrencies using a hybrid LSTM-Attention model. By leveraging historical data such as Open, High, Low, Close prices, and Volume, this model aims to provide accurate short-term forecasts — a valuable tool for traders and researchers alike.

## 🧠 Model Overview

The model combines:

- **LSTM (Long Short-Term Memory)**: Captures temporal dependencies in time series data.
- **Attention Mechanism**: Assigns dynamic importance to different time steps, improving interpretability and accuracy.

### Architecture

- Input: 60-day sequence of normalized OHLCV values.
- LSTM Layers: Two stacked layers with 64 units each.
- Dropout: 0.2 after each LSTM layer to mitigate overfitting.
- Attention Layer: Custom Keras implementation for dynamic focus on relevant time steps.
- Output: A single dense layer predicting the next day's closing price.

## 📊 Dataset

The model uses merged historical data of top cryptocurrencies such as BTC, ETH, ADA, DOT, BNB, and AAVE, derived from [CoinMarketCap](https://www.coinmarketcap.com).

**Features Used**:
- Open
- High
- Low
- Close
- Volume

### Preprocessing
- Dropped zero-volume and missing values.
- MinMax normalization applied to all features.
- Created overlapping sequences of 60 time steps for supervised learning.

## 🏋️‍♂️ Training

- **Loss**: Mean Squared Error (MSE)
- **Optimizer**: Adam
- **Batch Size**: 64
- **Epochs**: 20
- **Train/Test Split**: 80/20 (no shuffle)

## 📈 Evaluation Metrics

| Metric | Value       | Interpretation                                   |
|--------|-------------|--------------------------------------------------|
| MSE    | 2,219,864   | Low for assets like BTC or AAVE                 |
| MAE    | 585.79      | ~\$586 average prediction error                 |
| R²     | 0.964       | Explains 96.4% of price variance — excellent!  |

## 📉 Visualization

The model plots actual vs. predicted closing prices for test data, helping visualize prediction accuracy.



## 📂 File Structure

- `CryptoAI2.py`: Model code and training pipeline.
- `merged_crypto_data.csv`: Cleaned and combined dataset.
- `CryptoAI Report.docx`: Detailed technical report of the model.
- `README.md`: This project description.

## 🚀 Future Work

- Multi-currency predictions
- Incorporating external signals (news sentiment, macro indicators)
- Deployment via API for real-time predictions

## 📜 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

🔍 _Developed by Yuksel Celik — Powered by TensorFlow & Keras_
