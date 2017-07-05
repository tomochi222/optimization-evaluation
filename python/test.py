import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np

def func(x,y):
    return x**2 + y**2

x = np.arange(-5,5,0.25)
y = np.arange(-5,5,0.25)

X, Y = np.meshgrid(x,y)

Z = func(X,Y)

print(Z)