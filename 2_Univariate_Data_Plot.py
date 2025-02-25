"""
Univariate Forecasting Plot to Picture
Created and Modified by PEACE
Last modified: 25/02/2025
"""

from pandas import read_csv
from matplotlib import pyplot
from datetime import datetime
from pathlib import Path
import numpy

filename = 'dataset/Bangkok_solarpv_Trial.csv'
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

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
pyplot.savefig(Path(f"./figure/2_plot_{timestamp}.png"), dpi=600)
pyplot.show()