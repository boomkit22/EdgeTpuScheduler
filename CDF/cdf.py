
#%%
import numpy as np
import matplotlib.pyplot as plt

f = open('../Missing_Scenario/Efficient_S', 'r')
lines = f.readlines()
array = []
for line in lines:
    array.append(float(line))
    
x = np.sort(array)

#calculate CDF values
y = 1. * np.arange(len(array)) / (len(array) - 1)

#plot CDF
plt.plot(x, y)
plt.xlabel('response time(ms)')
f.close()

#%%
