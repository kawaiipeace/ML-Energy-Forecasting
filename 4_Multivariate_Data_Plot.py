"""
Multivariate Forecasting Plot to Picture
Created and Modified by PEACE
Last modified: 17/04/2023
"""

from pandas import read_csv
from matplotlib import pyplot

dataset = read_csv('dataset/TetuanMAR_consumption_Trial.csv', header=0, index_col=0)
values = dataset.values

groups = [0, 1, 2, 3]
i = 1

pyplot.figure()
for group in groups:
	pyplot.subplot(len(groups), 1, i)
	pyplot.plot(values[:, group])
	pyplot.title(dataset.columns[group], y=0.5, loc='right')
	i += 1
pyplot.tight_layout()
pyplot.savefig('./figure/4_TetuanMAR_consumption_plot' + '.png', format='png', dpi=600)
pyplot.show()