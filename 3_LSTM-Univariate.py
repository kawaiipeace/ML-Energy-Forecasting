"""
Univariate Forecasting with LSTM model
Created and Modified by PEACE
Last modified: 17/04/2023
Further Reading: https://shorturl.at/1jibR
"""

import time
import numpy
import tensorflow as tf
from pandas import DataFrame
from pandas import Series
from pandas import concat
from pandas import read_csv
from pathlib import Path
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Bidirectional
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot

#### ตั้งค่าพารามิเตอร์ตรงนี้ INITIALIZATION START HERE!!! ####
filename = Path('dataset/Bangkok_solarpv_Trial.csv') # ชื่อไฟล์ที่จะนำมาพยากรณ์
horizon = 372 # ค่าสเตปการพยากรณ์ ดูเพิ่มเติมได้จาก dataset details.pdf
neurons = 10 # ค่าโหนดสำหรับแบบจำลอง LSTM ปรับแก้ไขได้ตามชอบ (ค่าปกติคือ 10)
epoch = 100 # ค่าการวนซ้ำสำหรับแบบจำลอง LSTM ปรับแก้ไขได้ตามชอบ (ค่าปกติคือ 100)
batch_size = 32 # จำนวนข้อมูลสำหรับการปรับปรุงในช่วงการเรียนรู้ในแต่ละ epoch (ค่าปกติคือ 32)
model_selection = "LSTM" # ให้เลือกโมเดลระหว่าง LSTM หรือ BiLSTM

# สร้างฟังก์ชั่น ทำข้อมูลเป็นเชิงลำดับให้กับแบบจำลอง Machine Learning
def timeseries_to_supervised(data, lag=1):
	df = DataFrame(data)
	columns = [df.shift(i) for i in range(1, lag+1)]
	columns.append(df)
	df = concat(columns, axis=1)
	df.fillna(0, inplace=True)
	return df

# สร้างฟังก์ชั่น ทำข้อมูลให้อยู่ในรูปแบบอนุกรมเวลาแบบความต่าง
def difference(dataset, interval=1):
	diff = []
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return Series(diff)

# สร้างฟังก์ชั่น แปลงข้อมูลจากอนุกรมความต่างให้เป็นรูปแบบปกติ
def inverse_difference(history, yhat, interval=1):
	return yhat + history[-interval]

# สร้างฟังก์ชั่น นอร์มอลไลซ์ข้อมูลด้วยวิธี Min-Max ให้อยู่ระหว่าง [-1,1]
def scale(train, test):
	# fit scaler
	scaler = MinMaxScaler(feature_range=(-1, 1))
	scaler.fit(train)
	# transform train
	# train = train.reshape(train.shape[0], train.shape[1])
	train_scaled = scaler.transform(train)
	# transform test
	#test = test.reshape(test.shape[0], test.shape[1])
	test_scaled = scaler.transform(test)
	return scaler, train_scaled, test_scaled

# สร้างฟังก์ชั่น แปลงข้อมูลนอร์มอลไลซ์ให้อยู่ในรูปแบบปกติ
def invert_scale(scaler, X, value):
	array = numpy.array([*X, value]).reshape(1, -1)
	inverted = scaler.inverse_transform(array)[0, -1]
	return inverted

# สร้างฟังก์ชั่น สร้างแบบจำลอง LSTM เพื่อใช้ในการฝึกฝน
def fit_lstm(train, batch_size, epochs, neurons):
    X, y = train[:, :-1], train[:, -1]
    X = X.reshape(X.shape[0], 1, X.shape[1])
    if model_selection == "BiLSTM":
        model = Sequential([
    		tf.keras.Input(shape=(X.shape[1], X.shape[2])),
			Bidirectional(LSTM(neurons, input_shape=(X.shape[1], X.shape[2]), return_sequences=False)),
			Dense(1)
		])
    else:
        model = Sequential([
    		tf.keras.Input(shape=(X.shape[1], X.shape[2])),
			LSTM(neurons, return_sequences=False),
			Dense(1)
		])
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=1, shuffle=False)
    return model

# สร้างฟังก์ชั่น LSTM เพื่อใช้ในการทดสอบหรือพยากรณ์
def forecast_lstm(model, X):
	X = X.reshape(1, 1, len(X))
	return model.predict(X)[0, 0]

#### เริ่มกระบวนการ Forecast ####

# โหลดข้อมูล
print('\n======= Univariate Forecasting is in progress, PLEASE WAIT... =======\n')
series = read_csv(filename, header=0, parse_dates=[0], index_col=0)

# แปลงข้อมูลให้อยู่ในรูปแบบอนุกรมเวลาที่คงที่
raw_values = series.values.flatten()
diff_values = difference(raw_values, 1)

# เปลี่ยนข้อมูลให้อยู่ในรูปแบบพร้อมใช้งานในแบบจำลอง
supervised = timeseries_to_supervised(diff_values, 1)
supervised_values = supervised.values

# แบ่งข้อมูลฝึกฝนและทดสอบ (พยากรณ์)
train, test = supervised_values[:-horizon], supervised_values[-horizon:]

# แปลงข้อมูลให้อยู่ในรูปแบบนอร์มัลไลซ์
scaler, train_scaled, test_scaled = scale(train, test)

# แบบจำลองทำการฝึกฝน
timer_begin  = time.time() # เริ่มจับเวลาการทำงานของแบบจำลอง
lstm_model = fit_lstm(train_scaled, batch_size, epoch, neurons)

# แบบจำลองเรียนรู้ข้อมูลการฝึกฝน
train_reshaped = train_scaled[:, :-1].reshape(len(train_scaled), 1, 1)
train_predictions = lstm_model.predict(train_reshaped)
train_predicted_values=[]
for i in range(len(train_predictions)):
	X_train, y_train = train_scaled[i, :-1], train_predictions[i,0]
	yhat_train = invert_scale(scaler, X_train, y_train)
	yhat_train = inverse_difference(raw_values, yhat_train, len(train_scaled)+1-i)
	train_predicted_values.append(yhat_train)
train_predictions = numpy.array(train_predicted_values)

# แบบจำลองทำการทดสอบ (พยากรณ์)
predictions = []
for i in range(len(test_scaled)):
	# สร้างการพยากรณ์ทีละหนึ่งสเตปข้อมูล
	X, y = test_scaled[i, :-1], test_scaled[i, -1]
	yhat = forecast_lstm(lstm_model, X)
	# แปลงข้อมูลจากรูปนอร์มัลไลซ์ให้อยู่ในรูปปกติ
	yhat = invert_scale(scaler, X, yhat)
	# แปลงข้อมูลจากอนุกรมความต่างให้เป็นรูปแบบปกติ
	yhat = inverse_difference(raw_values, yhat, len(test_scaled)+1-i)
	# เก็บข้อมูลพยากรณ์
	predictions.append(yhat)
	expected = raw_values[len(train) + i + 1]
	print(f"Point={i+1}, Predicted={yhat:.3f}, Expected={expected:.3f}")

# คำนวณเวลาการทำงานของแบบจำลอง
timer_end = time.time()
timer_result = timer_end - timer_begin
print(f"\nProcessing Time: {timer_result:.2f} seconds")

# รายงานผลลัพธ์
train_rmse = root_mean_squared_error(raw_values[:len(train)], train_predictions)
train_rsq = r2_score(raw_values[:len(train)], train_predictions)
test_rmse = root_mean_squared_error(raw_values[-horizon:], predictions)
test_rsq = r2_score(raw_values[-horizon:], predictions)
print(f"Train RMSE: {train_rmse:.3f}")
print(f"Train R-Squared: {train_rsq:.3f}")
print(f"Test RMSE: {test_rmse:.3f}")
print(f"Test R-Squared: {test_rsq:.3f}")

# สร้างกราฟผลลัพธ์
pyplot.plot(raw_values[-horizon:], label="Actual")
pyplot.plot(predictions, label="Forecast", linestyle="dashed")
pyplot.title("Bangkok Solar PV with LSTM Univariate Forecasting")
pyplot.xlabel("Data Point")
pyplot.ylabel("Solar Irradiance (W/$m^2$)")
pyplot.legend(loc="upper right")
pyplot.savefig(Path("./figure/3_Univariate_Forecasting.png"), dpi=600)
pyplot.show()

# เขียนผลลัพธ์ให้อยู่ในรูปแบบไฟล์ Excel
predtoarray = numpy.array(predictions)
outputdata = numpy.stack((raw_values[-horizon:], predtoarray), axis=1)
out = DataFrame(outputdata.reshape(horizon,2), columns=['Actual','Predict'])
out.to_excel(Path("./result/report_Bangkok_SolarPV.xlsx"), index=False, engine="openpyxl")

print('\n======= Univariate Forecasting is COMPLETED =======\n')