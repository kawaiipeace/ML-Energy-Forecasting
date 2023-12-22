"""
Multivariate Forecasting with LSTM model
Created and Modified by PEACE
Last modified: 17/04/2023
"""

from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import numpy
import time
 
# สร้างฟังก์ชั่น ทำข้อมูลเป็นเชิงลำดับให้กับแบบจำลอง Machine Learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	agg = concat(cols, axis=1)
	agg.columns = names
	if dropnan:
		agg.dropna(inplace=True)
	return agg

#### ตั้งค่าพารามิเตอร์ตรงนี้ INITIALIZATION START HERE!!! ####
filename = 'dataset/TetuanMAR_consumption_Trial.csv' # ชื่อไฟล์ที่จะนำมาพยากรณ์
horizon = 4320 # ค่าสเตปการพยากรณ์ ดูเพิ่มเติมได้จาก dataset details.pdf
neurons = 10 # ค่าโหนดสำหรับแบบจำลอง LSTM ปรับแก้ไขได้ตามชอบ (ค่าปกติคือ 10)
epoch = 100 # ค่าการวนซ้ำสำหรับแบบจำลอง LSTM ปรับแก้ไขได้ตามชอบ (ค่าปกติคือ 100)
 
# โหลดและจัดกระบวนการข้อมูล
print('\n======= Multivariate Forecasting is in progress, PLEASE WAIT... =======\n')
dataset = read_csv(filename, header=0, index_col=0)
values = dataset.values
# เช็คว่าข้อมูลทุกตัวเป็นค่าตัวเลขทั้งหมด
values = values.astype('float32')
# แปลงข้อมูลให้อยู่ในรูปแบบนอร์มัลไลซ์
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
# ทำข้อมูลเป็นเฟรม (รูปแบบ Time Series) ไว้เรียนรู้ในแบบจำลอง Machine Learning
reframed = series_to_supervised(scaled, 1, 1)
reframed.drop(reframed.columns[[5,6,7]], axis=1, inplace=True)
print(reframed.head())
 
# แบ่งข้อมูลฝึกฝนและทดสอบ (พยากรณ์)
values = reframed.values
train = values[:horizon, :]
test = values[horizon:, :]
# จัดการข้อมูลนำเข้าและข้อมูลเป้าหมาย
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]
# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
 
# design network
timer_begin  = time.time() # เริ่มจับเวลาการทำงานของแบบจำลอง
model = Sequential()
model.add(LSTM(neurons, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))
loss_measure='mae'
model.compile(loss=loss_measure, optimizer='adam')
# fit network
history = model.fit(train_X, train_y, epochs=epoch, batch_size=128, validation_data=(test_X, test_y), verbose=2, shuffle=False)
# plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.title("Multivariate Forecasting Validation (Train and Test Comparison)")
pyplot.xlabel("Epoch")
pyplot.ylabel("Evaluate Loss with: %s" %(loss_measure))
pyplot.legend(loc="upper right")
pyplot.legend()
pyplot.savefig('./figure/5_Multivariate_Validation' + '.png', format='png', dpi=600)
pyplot.show()
 
# make a prediction
yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
# invert scaling for forecast
inv_yhat = concatenate((yhat, test_X[:, 1:]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,0]
# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = concatenate((test_y, test_X[:, 1:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,0]

# คำนวณเวลาการทำงานของแบบจำลอง
timer_end = time.time()       
timer_result = timer_end - timer_begin 
print('\nProcessing Time: %d second' %(timer_result))

# calculate RMSE
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)

# สร้างกราฟผลลัพธ์
pyplot.plot(inv_y, label="Actual")
pyplot.plot(inv_yhat, label="Forecast")
pyplot.title("Tetuan Power Consumption with LSTM Multivariate Forecasting")
pyplot.xlabel("Data Point")
pyplot.ylabel("Power Consumption (kW)")
pyplot.legend(loc="upper right")
pyplot.savefig('./figure/5_Multivariate_Forecasting' + '.png', format='png', dpi=600)
pyplot.show()

# เขียนผลลัพธ์ให้อยู่ในรูปแบบไฟล์ Excel
outputdata = numpy.stack((inv_y, inv_yhat), axis=1)
out = DataFrame(outputdata)
out.to_excel(r'result/report_Tetuan_PowerConsumption.xlsx', index=True)

print('\n======= Multivariate Forecasting is COMPLETED =======\n')