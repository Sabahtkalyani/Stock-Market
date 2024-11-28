import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import streamlit as st
from datetime import datetime
from functions import prepare_data
import train_model


start = '2014-01-01'
end = '2023-12-31'

st.title('Stock Trend Prediction')

user_input = st.text_input('Enter Stock Ticker', 'AAPL')
df = yf.download(user_input, start=start, end=end)

# Describing data
st.subheader('Data from 2014 - 2023')
st.write(df.describe())

# Visualizations
st.subheader('Closing Price vs Time Chart')
fig = plt.figure(figsize=(12, 6))
plt.plot(df['Close'])
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with 100MA')
ma100 = df['Close'].rolling(100).mean()
fig = plt.figure(figsize=(12, 6))
plt.plot(df['Close'], label='Closing Price')
plt.plot(ma100, label='100-Day MA')
plt.legend()
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with 100MA & 200MA')
ma200 = df['Close'].rolling(200).mean()
fig = plt.figure(figsize=(12, 6))
plt.plot(df['Close'], label='Closing Price')
plt.plot(ma100, label='100-Day MA')
plt.plot(ma200, label='200-Day MA')
plt.legend()
st.pyplot(fig)

# Prepare data for testing
_, _, x_test, y_test, scaler = prepare_data(df)

# Reshape data for LSTM model
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))



# Load model
model = load_model('keras_model.h5')


# Making predictions
y_predicted = model.predict(x_test)
scale_factor = 1 / scaler.scale_[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

# Final graph
st.subheader('Predictions vs Original')
fig2 = plt.figure(figsize=(12, 6))
plt.plot(y_test, 'b', label='Original Price')
plt.plot(y_predicted, 'r', label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)
