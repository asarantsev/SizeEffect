# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 12:43:45 2019

@author: UNR Math Stat
"""

#Importing Libraries
import pandas
import numpy
import math
import scipy
from scipy import stats
from matplotlib import pyplot
from statsmodels.graphics.gofplots import qqplot
from statsmodels import api

file = pandas.ExcelFile('Vanguard-Total-ETFs.xlsx')
df = file.parse('Size')
rate = file.parse('Treasury')
values = df.values
R = rate.values[0]
totalPrice = values[0, 5::3]
totalDiv = values[1, 5::3]
totalNum = len(totalPrice) - 1
totalRet = [math.log((totalPrice[k] + totalDiv[k])/totalPrice[k+1]) 
        for k in range(totalNum)]
print(totalNum)
smallPrice = values[6, 5::3]
smallNum = 60
smallPrice = smallPrice[:smallNum+1]
smallDiv = values[7, 5::3]
smallDiv = smallDiv[:smallNum]
smallRet = [math.log((smallPrice[k] + smallDiv[k])/smallPrice[k+1]) 
        for k in range(smallNum)]
T = smallNum
totalRet = totalRet[:T]
R = R[4::3]
R = R[:T]
bondRet = numpy.array([math.log(1 + item/4) for item in R])
totalP = totalRet - bondRet
smallP = smallRet - bondRet
rvalue = numpy.corrcoef(totalP, smallP)[0][1]
reg = stats.linregress(totalP, smallP)
s = reg.slope
i = reg.intercept
residuals = numpy.array([smallP[k] - s*totalP[k] - i for k in range(T)])
stderr = math.sqrt(numpy.dot(residuals, residuals)/(T - 2))
print('Results for small cap')
print('num of quarters = ', T)
print('slope = ', round(s, 4))
print('intercept = ', round(i, 4))
print('stderr = ', round(stderr, 4))
print('R Squared = ', round(rvalue*rvalue, 4))
qqplot(residuals, line = 's')
pyplot.show()
p = stats.shapiro(residuals)[1]
print('p-value for normality = ', round(p, 4))