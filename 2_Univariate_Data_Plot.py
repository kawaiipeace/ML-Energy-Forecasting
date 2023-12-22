"""
Univariate Forecasting Plot to Picture
Created and Modified by PEACE
Last modified: 17/04/2023
"""

from pandas import read_csv
from matplotlib import pyplot
import numpy

filename = 'dataset/Bangkok_solarpv_Trial.csv'

series = read_csv(filename, header=0, parse_dates=[0], index_col=0)
raw_values = series.values

x_axis_train = numpy.arange(1,731)
x_axis_test = numpy.arange(732,1104)

pyplot.figure()
pyplot.plot(x_axis_train,raw_values[1:731], label="Train")
pyplot.plot(x_axis_test,raw_values[732:1104], label="Test")
pyplot.title("Bangkok Solar PV")
pyplot.xlabel("Data Point")
pyplot.ylabel("Solar Irradiance (W/$m^2$)")
pyplot.legend(loc="upper right")
pyplot.savefig('./figure/2_Bangkok_solarpv_plot' + '.png', format='png', dpi=600)
pyplot.show()