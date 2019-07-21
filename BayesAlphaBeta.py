# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 16:12:23 2019

@author: UNR Math Stat
"""



#Importing Libraries
import pandas
import math
import numpy
from numpy import random
import scipy
from scipy import stats
from matplotlib import pyplot
from statsmodels.graphics.gofplots import qqplot
from statsmodels import api

file1 = pandas.ExcelFile('Data500.xlsx')
df1 = file1.parse('Return')
df2 = file1.parse('Cap')

file2 = pandas.ExcelFile('VFINX.xlsx')
dfV = file2.parse('VFINX')
file3 = pandas.ExcelFile('TR3month-quarterly.xlsx')
dfR = file3.parse('Rates')

tempReturns = df1.values
tempCaps = df2.values
tempIndices = dfV.values
tempRates = dfR.values

NQTRS = 120
NSTOCKS = 500

#Create 2-dimensional lists that are better suited for our analysis
returns = [[math.log(1 + tempReturns[stock][NQTRS - t]) for t in range(NQTRS)] for stock in range(NSTOCKS)]
caps = [[tempCaps[stock][NQTRS + 1 - t] for t in range(NQTRS)] for stock in range(NSTOCKS)]

select1 = ~numpy.isnan(caps).any(axis = 1)
select2 = ~numpy.isnan(returns).any(axis = 1)
select = [select1[i] and select2[i] for i in range(NSTOCKS)]

Returns = [returns[k] for k in range(NSTOCKS) if select[k]]
Caps = [caps[k] for k in range(NSTOCKS) if select[k]]
Returns = numpy.array(Returns)
Caps = numpy.array(Caps)
NQTRS = numpy.size(Returns, axis = 1)
NSTOCKS = numpy.size(Returns, axis = 0)
Total = numpy.sum(Caps, axis = 0)
TreasuryRates = tempRates[:NQTRS, 1]
TreasuryRates = TreasuryRates[::-1]
TreasuryReturns = [math.log(1 + item/4) for item in TreasuryRates]
Benchmark = [sum([Returns[stock, time] * Caps[stock, time] for stock in range(NSTOCKS)])/Total[time] for time in range(NQTRS)]
Prems = [[Returns[stock][time] - TreasuryReturns[time] for time in range(NQTRS)] for stock in range(NSTOCKS)]
Prems = numpy.array(Prems)
BP = [Benchmark[time] - TreasuryReturns[time] for time in range(NQTRS)]
Alpha = []
Beta = []
for stock in range(NSTOCKS):
    s = stats.linregress(BP, Prems[stock])
    alpha = s.intercept
    Alpha.append(alpha)
    beta = s.slope
    Beta.append(beta)
LogCap = [math.log(Caps[stock, 0]) for stock in range(NSTOCKS)]
print(stats.linregress(LogCap, Alpha))
print(stats.linregress(LogCap, Beta))
Regression1(LogCap, Alpha)
Regression1(LogCap, Beta)