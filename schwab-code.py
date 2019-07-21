# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 19:32:00 2019

@author: UNR Math Stat
"""

import math
import numpy
import pandas
from scipy import stats
import matplotlib.pyplot as plt
import statsmodels.api as sm

inputFile = pandas.read_excel('inter-invesco.xlsx')


#open sheets
data = inputFile.values

Candidate = data[:, 1].astype(numpy.float)
Candidate = numpy.array([math.log(1 + item) for item in Candidate])
Benchmark = data[:, 2].astype(numpy.float)
Benchmark = numpy.array([math.log(1 + item) for item in Benchmark])
Treasury = data[:, 3].astype(numpy.float)
Treasury = numpy.array([math.log(1 + item) for item in Treasury])
T = len(Candidate)
print(T)

CPremium = Candidate - Treasury
BPremium = Benchmark - Treasury

df = pandas.DataFrame({'Slope': BPremium})
df = sm.add_constant(df)
Reg = sm.OLS(CPremium, df).fit()
print(Reg.summary())
residuals = CPremium - Reg.predict(df)
stderr = math.sqrt(1/(T-2)*numpy.dot(residuals, residuals))
print('sigma = ', stderr)
print('normality p = ', stats.shapiro(residuals)[1])

