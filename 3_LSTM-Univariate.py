"""
Univariate Forecasting with LSTM model
Created and Modified by PEACE
Last modified: 17/04/2023
"""

from pandas import DataFrame
from pandas import Series
from pandas import concat
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from math import sqrt
from matplotlib import pyplot
import numpy
import time

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
	diff = list()
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
	scaler = scaler.fit(train)
	# transform train
	train = train.reshape(train.shape[0], train.shape[1])
	train_scaled = scaler.transform(train)
	# transform test
	test = test.reshape(test.shape[0], test.shape[1])
	test_scaled = scaler.transform(test)
	return scaler, train_scaled, test_scaled

# สร้างฟังก์ชั่น แปลงข้อมูลนอร์มอลไลซ์ให้อยู่ในรูปแบบปกติ
def invert_scale(scaler, X, value):
	new_row = [x for x in X] + [value]
	array = numpy.array(new_row)
	array = array.reshape(1, len(array))
	inverted = scaler.inverse_transform(array)
	return inverted[0, -1]

# สร้างฟังก์ชั่น สร้างแบบจำลอง LSTM เพื่อใช้ในการฝึกฝน
def fit_lstm(train, batch_size, nb_epoch, neurons):
	X, y = train[:, 0:-1], train[:, -1]
	X = X.reshape(X.shape[0], 1, X.shape[1])
	model = Sequential()
	model.add(LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
	model.add(Dense(1))
	model.compile(loss='mean_squared_error', optimizer='adam')
	for i in range(nb_epoch):
		model.fit(X, y, epochs=1, batch_size=batch_size, verbose=0, shuffle=False)
		model.reset_states()
	return model

# สร้างฟังก์ชั่น LSTM เพื่อใช้ในการทดสอบหรือพยากรณ์
def forecast_lstm(model, batch_size, X):
	X = X.reshape(1, 1, len(X))
	yhat = model.predict(X, batch_size=batch_size)
	return yhat[0,0]

#### ตั้งค่าพารามิเตอร์ตรงนี้ INITIALIZATION START HERE!!! ####
filename = 'dataset/Bangkok_solarpv_Trial.csv' # ชื่อไฟล์ที่จะนำมาพยากรณ์
horizon = 372 # ค่าสเตปการพยากรณ์ ดูเพิ่มเติมได้จาก dataset details.pdf
neurons = 10 # ค่าโหนดสำหรับแบบจำลอง LSTM ปรับแก้ไขได้ตามชอบ (ค่าปกติคือ 10)
epoch = 100 # ค่าการวนซ้ำสำหรับแบบจำลอง LSTM ปรับแก้ไขได้ตามชอบ (ค่าปกติคือ 100)

# โหลดข้อมูล
print('\n======= Univariate Forecasting is in progress, PLEASE WAIT... =======\n')
series = read_csv(filename, header=0, parse_dates=[0], index_col=0)

# แปลงข้อมูลให้อยู่ในรูปแบบอนุกรมเวลาที่คงที่
raw_values = series.values
diff_values = difference(raw_values, 1)

# เปลี่ยนข้อมูลให้อยู่ในรูปแบบพร้อมใช้งานในแบบจำลอง
supervised = timeseries_to_supervised(diff_values, 1)
supervised_values = supervised.values

# แบ่งข้อมูลฝึกฝนและทดสอบ (พยากรณ์)
train, test = supervised_values[0:-horizon], supervised_values[-horizon:]

# แปลงข้อมูลให้อยู่ในรูปแบบนอร์มัลไลซ์
scaler, train_scaled, test_scaled = scale(train, test)

# แบบจำลองทำการฝึกฝน
timer_begin  = time.time() # เริ่มจับเวลาการทำงานของแบบจำลอง
lstm_model = fit_lstm(train_scaled, 1, epoch, neurons)

# แบบจำลองเรียนรู้ข้อมูลการฝึกฝน
train_reshaped = train_scaled[:, 0].reshape(len(train_scaled), 1, 1)
lstm_model.predict(train_reshaped, batch_size=1)

# แบบจำลองทำการทดสอบ (พยากรณ์)
predictions = list()
for i in range(len(test_scaled)):
	# สร้างการพยากรณ์ทีละหนึ่งสเตปข้อมูล
	X, y = test_scaled[i, 0:-1], test_scaled[i, -1]
	yhat = forecast_lstm(lstm_model, 1, X)
	# แปลงข้อมูลจากรูปนอร์มัลไลซ์ให้อยู่ในรูปปกติ
	yhat = invert_scale(scaler, X, yhat)
	# แปลงข้อมูลจากอนุกรมความต่างให้เป็นรูปแบบปกติ
	yhat = inverse_difference(raw_values, yhat, len(test_scaled)+1-i)
	# เก็บข้อมูลพยากรณ์
	predictions.append(yhat)
	expected = raw_values[len(train) + i + 1]
	print('Point=%d, Predicted=%f, Expected=%f \n' % (i+1, yhat, expected))

# คำนวณเวลาการทำงานของแบบจำลอง
timer_end = time.time()       
timer_result = timer_end - timer_begin 
print('\nProcessing Time: %d second' %(timer_result))

# รายงานผลลัพธ์
rmse = sqrt(mean_squared_error(raw_values[-horizon:], predictions))
print('Test RMSE: %.3f' % rmse)

# สร้างกราฟผลลัพธ์
pyplot.plot(raw_values[-horizon:], label="Actual")
pyplot.plot(predictions, label="Forecast")
pyplot.title("Bangkok Solar PV with LSTM Univariate Forecasting")
pyplot.xlabel("Data Point")
pyplot.ylabel("Solar Irradiance (W/$m^2$)")
pyplot.legend(loc="upper right")
pyplot.savefig('./figure/3_Univariate_Forecasting' + '.png', format='png', dpi=600)
pyplot.show()

# เขียนผลลัพธ์ให้อยู่ในรูปแบบไฟล์ Excel
predtoarray = numpy.array(predictions)
outputdata = numpy.stack((raw_values[-horizon:], predtoarray), axis=1)
out = DataFrame(outputdata.reshape(horizon,2), columns=['Actual','Predict'])
out.to_excel(r'result/report_Bangkok_SolarPV.xlsx', index=True)

print('\n======= Univariate Forecasting is COMPLETED =======\n')