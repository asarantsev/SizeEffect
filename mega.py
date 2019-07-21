# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 11:34:46 2019

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
stock = file.parse('Size')
rate = file.parse('Treasury')
values = stock.values
R = rate.values[0]
totalPrice = values[0, 5::3]
totalDiv = values[1, 5::3]
totalNum = len(totalPrice) - 1
totalRet = [math.log((totalPrice[k] + totalDiv[k])/totalPrice[k+1]) 
        for k in range(totalNum)]
megaPrice = values[2, 5::3]
megaPrice = megaPrice[:46]
megaDiv = values[3, 5::3]
megaDiv = megaDiv[:45]
T = 45
megaRet = [math.log((megaPrice[k] + megaDiv[k])/megaPrice[k+1]) 
        for k in range(T)]
megaRet = numpy.array(megaRet)
totalRet = numpy.array(totalRet[:T])
R = R[4::3]
R = R[:T]
bondRet = numpy.array([math.log(1 + item/4) for item in R])
totalP = totalRet - bondRet
megaP = megaRet - bondRet
rvalue = numpy.corrcoef(totalP, megaP)[0][1]
reg = stats.linregress(totalP, megaP)
s = reg.slope
i = reg.intercept
residuals = numpy.array([megaP[k] - s*totalP[k] - i for k in range(T)])
stderr = math.sqrt(numpy.dot(residuals, residuals)/(T - 2))
print('Mega Stock Analysis')
print('num of quarters = ', T)
print('slope = ', round(s, 4))
print('intercept = ', round(i, 4))
print('stderr = ', round(stderr, 4))
print('R Squared = ', round(rvalue*rvalue, 4))
qqplot(residuals, line = 's')
pyplot.show()
p = stats.shapiro(residuals)[1]
print('p-value for normality = ', round(p, 4))





