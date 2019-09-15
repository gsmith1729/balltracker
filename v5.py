import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt

data = [[456, 277], [435, 266], [416, 256], [396, 246], [375, 238], [356, 231], [336, 225], [315, 220], [296, 216], [275, 213], [253, 211], [234, 210], [213, 210], [192, 211], [171, 214], [150, 217], [130, 222], [108, 228], [86, 234], [66, 242], [44, 251], [24, 261], [3, 273]]
x=[]
y=[]
for i in data:
    x.append(i[0])
    y.append(i[1])
print(np.array(x).shape)
print(np.polyfit(np.array(x),np.array(y),2))
