import streamlit as st ## import streamlit
import pandas as pd ## import pandas
import numpy as np ## import numpy
from keras.preprocessing.sequence import TimeseriesGenerator ## import TimeseriesGenerator dari keras.preprocessing.sequence untuk membuat data generator
from sklearn.preprocessing import MinMaxScaler ## berfungsi untuk melakukan normalisasi data
import plotly.graph_objects as go ## berfungsi untuk membuat grafik
from keras.models import Sequential ## berfungsi untuk membuat model sequential
from keras.layers import LSTM, Dense, Activation ## berfungsi untuk membuat layer LSTM, Dense, Activation
from statsmodels.tsa.seasonal import seasonal_decompose ## berfungsi untuk membuat seasonal decompose
import matplotlib.pyplot as plt ## berfungsi untuk membuat grafik
st.set_option('deprecation.showPyplotGlobalUse', False) ## supaya tidak muncul pesan deprecated


st.write("# Data Covid 19")
# Load data
dataframe = pd.read_excel('datasetbulanan.xlsx') ## load dataset covid 19 dari excel
dataframe.index = [i for i in range(1, len(dataframe.values)+1)] ## membuat index baru
st.table(dataframe.head()) ## menampilkan 5 data teratas

# Mengubah format tanggal
dataframe['Date'] = pd.to_datetime(dataframe['Date'], format='%d/%m/%y').dt.strftime('%Y-%m-%d') ## mengubah format tanggal dari dd/mm/yy menjadi yyyy-mm-dd
st.table(dataframe.head()) ## menampilkan 5 data teratas

# Membuat Varibel Date, Data dan Cases
date = dataframe['Date'].tolist() ## mengubah data date menjadi list
data = dataframe['Cases'] ## mengubah data cases menjadi list
cases = dataframe['Cases'] ## mengubah data cases menjadi list

# Menampilkan Jumlah Data
st.write('Jumlah Data :', len(dataframe)) ## menampilkan jumlah data

# Mengubah Dataset Menjadi Numpy Array
dataset = np.array(data) ## mengubah data menjadi numpy array
st.write('Dataset Shape :', dataset.shape) ## menampilkan shape dari dataset

st.write("# Grafik")
st.write("### Grafik Data Covid 19")
# Menampilkan Grafik Dalam Bentuk Go Line Chart
trace = go.Scatter(
    x = date,
    y = cases,
    mode = 'lines',
    name = 'Data'
) ## membuat grafik dengan mode lines
layout = go.Layout(
    title = "Covid 19",
    xaxis = {'title' : "Date"},
    yaxis = {'title' : "Cases"}
) ## membuat layout dengan title dan xaxis dan yaxis
fig = go.Figure(data=[trace], layout=layout) ## membuat figure dengan data dan layout
st.plotly_chart(fig) ## menampilkan figure

st.write("### Grafik Seasonal Decompose")
# Menampilkan Grafik Seasonal Decompose
df_sd_date = pd.to_datetime(dataframe['Date'], format='%Y-%m-%d').dt.strftime('%Y-%m-%d') ## mengubah format tanggal dari yyyy-mm-dd menjadi yyyy-mm-dd
df_sd_case = dataframe['Cases'] ## mengubah data cases menjadi list
df_sd = pd.DataFrame(df_sd_case) ## mengubah data menjadi dataframe
df_sd.index = pd.DatetimeIndex(df_sd_date) ## mengubah index menjadi datetimeindex
fig = plt.Figure(figsize=(12,7)) ## membuat figure dengan ukuran 12x7
ax1 = plt.subplot(311) ## membuat subplot dengan 3 baris 1 kolom
ax2 = plt.subplot(312) ## membuat subplot dengan 3 baris 1 kolom
ax3 = plt.subplot(313) ## membuat subplot dengan 3 baris 1 kolom
sd = seasonal_decompose(df_sd) ## membuat seasonal decompose
sd.seasonal.plot(color='green', ax=ax1, title='Seasonality') ## membuat grafik seasonal
sd.trend.plot(color='green', ax=ax2, title='Trending') ## membuat grafik trending
sd.resid.plot(color='green', ax=ax3, title='Resid') ## membuat grafik resid
plt.subplots_adjust(hspace=1.5) ## membuat jarak antar subplot
st.pyplot() ## menampilkan figure

st.write("# Data Preprocessing")
st.write("### Min Max Scaler 5 Data Teratas")
scaler = MinMaxScaler() ## membuat objek MinMaxScaler
dataset = scaler.fit_transform(dataset.reshape(-1,1)) ## melakukan normalisasi data dengan fit_transform dan mengubah shape menjadi -1x1 atau 1 kolom
df_scaled = pd.DataFrame(dataset, columns=['Scaled Data Train'], index=dataframe.index) ## membuat dataframe dengan kolom Scaled Data Train dan index dataframe
st.table(df_scaled.head()) ## menampilkan 5 data teratas
st.write("### Min Max Scaler 5 Data Teratas")
st.table(df_scaled.tail()) ## menampilkan 5 data terbawah

st.write("# Modelling")
n_input = 6 ## jumlah input 6 bulan terakhir
n_features = 1 ## jumlah feature 1 yakni cases
train_generator = TimeseriesGenerator(dataset, dataset, length=n_input, batch_size=1) ## membuat generator dengan TimeseriesGenerator

model = Sequential() ## membuat model sequential
model.add(LSTM(128, activation='relu', return_sequences=True,input_shape=(n_input, n_features))) ## membuat layer LSTM dengan 128 neuron, activation relu, return_sequences true dan input shape 6x1
model.add(Dense(64)) ## membuat layer dense dengan 64 neuron
model.add(Dense(32)) ## membuat layer dense dengan 32 neuron
model.add(Dense(16)) ## membuat layer dense dengan 16 neuron
model.add(Dense(8)) ## membuat layer dense dengan 8 neuron
model.add(Dense(4)) ## membuat layer dense dengan 4 neuron
model.add(Dense(1)) ## membuat layer dense dengan 1 neuron
model.add(Activation('linear')) ## membuat layer activation linear
model.compile(optimizer='adam', loss='mse') ## compile model dengan optimizer adam dan loss function mse (mean squared error)

model.summary() ## menampilkan summary model

history = model.fit(train_generator, epochs=50, verbose=1) ## melatih model dengan model.fit_generator dan epochs 50 dan verbose 1

st.write("### Arsitektur Model")

for layer in model.layers:
    name = layer.get_config()['name'] ## mengambil nama layer
    output_shape = layer.output_shape ## mengambil output shape layer
    param = layer.count_params() ## mengambil jumlah parameter layer
    st.write(name, output_shape, param) ## menampilkan nama layer, output shape layer dan jumlah parameter layer
    
scores = model.evaluate(train_generator, verbose=0) ## menghitung loss model
st.write("Model Berhasil Dibuat...")
st.write("Loss Model: %.2f%%" % (scores))

st.write("# Forecasting (Future Prediction)")
num_prediction = st.number_input('Masukkan Jumlah Bulan Yang Ingin Di Prediksi', min_value=1, max_value=100, value=6, step=1) ## membuat input number dengan min value 1, max value 100, value 6 dan step 1
if num_prediction > 0: ## jika num_prediction lebih dari 0
    st.write("##### Prediksi Covid 19 Selama", num_prediction, "Bulan") ## menampilkan prediksi covid 19 selama num_prediction bulan
    
st.caption('Misalnya : 4 (Maka akan di prediksi 4 bulan kedepan dari bulan terakhir dari dataset) 2022-12-01, 2023-01-01, 2023-02-01, 2023-03-01, 2023-04-01')
dataset = scaler.inverse_transform(dataset) ## mengembalikan nilai dataset ke nilai semula dengan inverse_transform

def predict(num_prediction, model): ## membuat fungsi predict
    prediction_list = dataset[-n_input:] ## mengambil 6 data terakhir dari dataset
    for _ in range(num_prediction): ## melakukan perulangan sebanyak num_prediction kali
        x = prediction_list[-n_input:] ## mengambil 6 data terakhir dari prediction_list
        x = x.reshape((1, n_input, 1)) ## mengubah shape menjadi 1x6x1 atau 1 baris, 6 kolom dan 1 feature
        out = model.predict(x)[0][0] ## melakukan prediksi dengan model.predict dan mengambil nilai prediksi pertama
        out = scaler.inverse_transform(out.reshape(-1, 1)) ## mengembalikan nilai prediksi ke nilai semula dengan inverse_transform
        prediction_list = np.append(prediction_list, out) ## menambahkan nilai prediksi ke prediction_list
    prediction_list = prediction_list[n_input-1:] ## mengambil nilai prediksi dari index ke 5 sampai akhir
        
    return prediction_list ## mengembalikan nilai prediksi
    
def predict_bulan(num_prediction): ## membuat fungsi predict_bulan
    last_date = date[-1] ## mengambil tanggal terakhir dari date
    prediction_dates = pd.date_range(last_date, periods=num_prediction+1, freq='MS').tolist() ## membuat prediksi tanggal dengan pd.date_range
    return prediction_dates ## mengembalikan prediksi tanggal

forecast = predict(num_prediction, model).astype(int) ## memanggil fungsi predict dan mengubah nilai prediksi ke integer
forecast_bulan = predict_bulan(num_prediction) ## memanggil fungsi predict_bulan

forecast = [max(0, x) for x in forecast] ## mengubah nilai negatif menjadi 0

df_final = pd.DataFrame({'Date':np.array(forecast_bulan), 'Prediksi':np.array(forecast)}) ## membuat dataframe dengan date dan prediksi sebagai kolom
df_final['Date'] = pd.to_datetime(df_final['Date']).dt.strftime('%Y-%m-%d') ## mengubah format date menjadi yyyy-mm-dd

given_trace = go.Scatter(
    x = date,
    y = cases,
    mode = 'lines',
    name = 'Data Training' 
) ## membuat trace untuk data training
forcast_trace = go.Scatter(
    x = forecast_bulan,
    y = forecast,
    mode = 'lines',
    name = 'Forecast',
    line = dict(color='red')
) ## membuat trace untuk data prediksi
layout = go.Layout(
    title = "Covid 19",
    xaxis = {'title' : "Date"},
    yaxis = {'title' : "Case"}
) ## membuat layout untuk plotly
fig = go.Figure(data=[given_trace, forcast_trace], layout=layout) ## membuat figure untuk plotly
st.plotly_chart(fig) ## menampilkan plotly chart

st.table(df_final) ## menampilkan dataframe