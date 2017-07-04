
# coding: utf-8

import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

##### Optimization benchmark function group #####
##### Class Ackley function #####
class Ackley:
    def __init__(self, variable_num):
        ### Information each particle ###
        self.variable_num = variable_num
        self.max_search_range = 32.768
        self.min_search_range = -32.768
        self.optimal_solution = np.zeros((1,self.variable_num))

    def get_optimal_solution(self):
        return self.optimal_solution

    def get_search_range(self):
        return [self.max_search_range, self.min_search_range]

    def get_func_val(self, variables):
        return 20-20*np.exp(-0.2*np.sqrt(1/self.variable_num*np.sum(np.square(variables)))+np.e-np.exp(1/self.variable_num*np.sum(np.cos(variables*2*np.pi))))

    def plot_2dimension(self):
        if self.variable_num == 2:
            x = np.arange(self.min_search_range,self.max_search_range, 0.25)
            y = np.arange(self.min_search_range,self.max_search_range, 0.25)
            X, Y = np.meshgrid(x,y)
            Z = []
            for xy_list in zip(X,Y):
                z = []
                for xy_input in zip(xy_list[0],xy_list[1]):
                    z.append(self.get_func_val(np.array(xy_input)))
                Z.append(z)
            Z = np.array(Z)
            fig = plt.figure()
            ax = Axes3D(fig)
            ax.plot_wireframe(X,Y,Z)
            plt.show()
        else:
            print('This method only can use for 2 variables')

##### Class Sphere function #####
class Sphere:
    def __init__(self, variable_num):
        ### Information each particle ###
        self.variable_num = variable_num
        self.max_search_range = 1000 # nearly inf
        self.min_search_range = -1000 # nearly inf
        self.optimal_solution = np.zeros((1,self.variable_num))

    def get_optimal_solution(self):
        return self.optimal_solution

    def get_search_range(self):
        return [self.max_search_range, self.min_search_range]

    def get_func_val(self, variables):
        return np.sum(np.square(variables))

    def plot_2dimension(self):
        if self.variable_num == 2:
            x = np.arange(-100,100, 0.25)
            y = np.arange(-100,100, 0.25)
            X, Y = np.meshgrid(x,y)
            Z = []
            for xy_list in zip(X,Y):
                z = []
                for xy_input in zip(xy_list[0],xy_list[1]):
                    z.append(self.get_func_val(np.array(xy_input)))
                Z.append(z)
            Z = np.array(Z)
            fig = plt.figure()
            ax = Axes3D(fig)
            ax.plot_wireframe(X,Y,Z)
            plt.show()
        else:
            print('This method only can use for 2 variables')

def main():
    benchmark_func = Sphere(2)
    benchmark_func.plot_2dimension()

if __name__ == '__main__':
    main()
