
#%%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sympy import Max


f = open('../Missing_Scenario/Efficient_S', 'r')
lines = f.readlines()
array = []
print()
for line in lines:
    array.append(float(line))

max_value = max(array)
print('max_value = {}'.format(max_value))
    
x = np.sort(array)

#calculate CDF values
y = 1. * np.arange(len(array)) / (len(array) - 1)

#plot CDF
ax=plt.axes()
ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
# plt.xlim(32,140)
plt.rcParams["figure.figsize"] = (15,5)


plt.plot(x, y)
plt.xlabel('response time(ms)')
f.close()

#%%
