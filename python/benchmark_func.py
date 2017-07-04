
# coding: utf-8

import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

##### Optimization benchmark function group #####
##### Class Ackley function #####
class Ackley:
    def __init__(self, variable_num):
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

##### Class Rosenbrock function #####
class Rosenbrock:
    def __init__(self, variable_num):
        self.variable_num = variable_num
        self.max_search_range = 5
        self.min_search_range = -5
        self.optimal_solution = np.ones((1,self.variable_num))

    def get_optimal_solution(self):
        return self.optimal_solution

    def get_search_range(self):
        return [self.max_search_range, self.min_search_range]

    def get_func_val(self, variables):
        f = 0
        for i in range(self.variable_num-1):
            f += 100*np.power(variables[i+1]-np.power(variables[i],2),2)+np.power(variables[i]-1,2)
        return f

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

##### Class Beale function #####
class Beale:
    def __init__(self):
        self.variable_num = 2
        self.max_search_range = 4.5
        self.min_search_range = -4.5
        self.optimal_solution = np.array([3,0.5])

    def get_optimal_solution(self):
        return self.optimal_solution

    def get_search_range(self):
        return [self.max_search_range, self.min_search_range]

    def get_func_val(self, variables):
        tmp1 = np.power(1.5 - variables[0] + variables[0] * variables[1],2)
        tmp2 = np.power(2.25 - variables[0] + variables[0] * np.power(variables[1],2),2)
        tmp3 = np.power(2.625 - variables[0] + variables[0] * np.power(variables[1],3),2)
        return tmp1+tmp2+tmp3

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

##### Class Goldstein-Price function #####
class GoldsteinPrice:
    def __init__(self):
        self.variable_num = 2
        self.max_search_range = 2
        self.min_search_range = -2
        self.optimal_solution = np.array([0,-1])

    def get_optimal_solution(self):
        return self.optimal_solution

    def get_search_range(self):
        return [self.max_search_range, self.min_search_range]

    def get_func_val(self, variables):
        tmp1 = (1+np.power(variables[0]+variables[1]+1,2)*(19-14*variables[0]+3*np.power(variables[0],2)-14*variables[1]+6*variables[0]*variables[1]+3*np.power(variables[1],2)))
        tmp2 = (30+(np.power(2*variables[0]-3*variables[1],2)*(18-32*variables[0]+12*np.power(variables[0],2)+48*variables[1]-36*variables[0]*variables[1]+27*np.power(variables[1],2))))
        return tmp1*tmp2

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


def main():
    benchmark_func = Beale()
    benchmark_func.plot_2dimension()

if __name__ == '__main__':
    main()
