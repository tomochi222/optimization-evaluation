
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

##### Class Booth function #####
class Booth:
    def __init__(self):
        self.variable_num = 2
        self.max_search_range = 10
        self.min_search_range = -10
        self.optimal_solution = np.array([1,-3])

    def get_optimal_solution(self):
        return self.optimal_solution

    def get_search_range(self):
        return [self.max_search_range, self.min_search_range]

    def get_func_val(self, variables):
        tmp1 = np.power(variables[0]+2*variables[1]-7,2)
        tmp2 = np.power(2*variables[0]+variables[1]-5,2)
        return tmp1+tmp2

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

##### Class Bukin function N.6 #####
class BukinN6:
    def __init__(self):
        self.variable_num = 2
        self.max_search_range = np.array([-5,3])
        self.min_search_range = np.array([-15,-3])
        self.optimal_solution = np.array([-10,1])

    def get_optimal_solution(self):
        return self.optimal_solution

    def get_search_range(self):
        return [self.max_search_range, self.min_search_range]

    def get_func_val(self, variables):
        tmp1 = 100*np.sqrt(np.absolute(variables[1]-0.01*np.power(variables[1],2)))
        tmp2 = 0.01*np.absolute(variables[0]+10)
        return tmp1+tmp2

    def plot_2dimension(self):
        if self.variable_num == 2:
            x = np.arange(self.min_search_range[0],self.max_search_range[0], 0.25)
            y = np.arange(self.min_search_range[1],self.max_search_range[1], 0.25)
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

##### Class Matyas function #####
class Matyas:
    def __init__(self):
        self.variable_num = 2
        self.max_search_range = np.array([10,10])
        self.min_search_range = np.array([-10,-10])
        self.optimal_solution = np.array([0,0])

    def get_optimal_solution(self):
        return self.optimal_solution

    def get_search_range(self):
        return [self.max_search_range, self.min_search_range]

    def get_func_val(self, variables):
        tmp1 = 0.26*(np.power(variables[0],2)+np.power(variables[1],2))
        tmp2 = 0.48*variables[0]*variables[1]
        return tmp1-tmp2

    def plot_2dimension(self):
        if self.variable_num == 2:
            x = np.arange(self.min_search_range[0],self.max_search_range[0], 0.25)
            y = np.arange(self.min_search_range[1],self.max_search_range[1], 0.25)
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
    benchmark_func = Matyas()
    benchmark_func.plot_2dimension()

##### Class Levi function N.13 #####
class LeviN13:
    def __init__(self):
        self.variable_num = 2
        self.max_search_range = np.array([10,10])
        self.min_search_range = np.array([-10,-10])
        self.optimal_solution = np.array([1,1])

    def get_optimal_solution(self):
        return self.optimal_solution

    def get_search_range(self):
        return [self.max_search_range, self.min_search_range]

    def get_func_val(self, variables):
        tmp1 = np.power(np.sin(3*np.pi*variables[0]),2)
        tmp2 = np.power(variables[0]-1,2)*(1+np.power(np.sin(3*np.pi*variables[1]),2))
        tmp3 = np.power(variables[1]-1,2)*(1+np.power(np.sin(2*np.pi*variables[1]),2))
        return tmp1+tmp2+tmp3

    def plot_2dimension(self):
        if self.variable_num == 2:
            x = np.arange(self.min_search_range[0],self.max_search_range[0], 0.25)
            y = np.arange(self.min_search_range[1],self.max_search_range[1], 0.25)
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

##### Class Three-hump camel function #####
class ThreeHumpCamel:
    def __init__(self):
        self.variable_num = 2
        self.max_search_range = np.array([5,5])
        self.min_search_range = np.array([-5,-5])
        self.optimal_solution = np.array([0,0])

    def get_optimal_solution(self):
        return self.optimal_solution

    def get_search_range(self):
        return [self.max_search_range, self.min_search_range]

    def get_func_val(self, variables):
        return 2*np.power(variables[0],2)-1.05*np.power(variables[0],4)+np.power(variables[0],6)/6+variables[0]*variables[1]+np.power(variables[1],2)

    def plot_2dimension(self):
        if self.variable_num == 2:
            x = np.arange(self.min_search_range[0],self.max_search_range[0], 0.25)
            y = np.arange(self.min_search_range[1],self.max_search_range[1], 0.25)
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

##### Class Easom function #####
class Easom:
    def __init__(self):
        self.variable_num = 2
        self.max_search_range = np.array([100,100])
        self.min_search_range = np.array([-100,-100])
        self.optimal_solution = np.array([np.pi,np.pi])

    def get_optimal_solution(self):
        return self.optimal_solution

    def get_search_range(self):
        return [self.max_search_range, self.min_search_range]

    def get_func_val(self, variables):
        return -1.0*np.cos(variables[0])*np.cos(variables[1])*np.exp(-(np.power(variables[0]-np.pi,2)+np.power(variables[1]-np.pi,2)))

    def plot_2dimension(self):
        if self.variable_num == 2:
            x = np.arange(self.min_search_range[0],self.max_search_range[0], 0.25)
            y = np.arange(self.min_search_range[1],self.max_search_range[1], 0.25)
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

##### Class Eggholder function #####
class Eggholder:
    def __init__(self):
        self.variable_num = 2
        self.max_search_range = np.array([512,512])
        self.min_search_range = np.array([-512,-512])
        self.optimal_solution = np.array([512,404.2319])

    def get_optimal_solution(self):
        return self.optimal_solution

    def get_search_range(self):
        return [self.max_search_range, self.min_search_range]

    def get_func_val(self, variables):
        tmp1 = -(variables[1]+47)*np.sin(np.sqrt(np.absolute(variables[1]+variables[0]/2+47)))
        tmp2 = -variables[0]*np.sin(np.sqrt(np.absolute(variables[0]-(variables[1]+47))))
        return tmp1+tmp2

    def plot_2dimension(self):
        if self.variable_num == 2:
            x = np.arange(self.min_search_range[0],self.max_search_range[0], 1)
            y = np.arange(self.min_search_range[1],self.max_search_range[1], 1)
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

##### Class McCormick function #####
class McCormick:
    def __init__(self):
        self.variable_num = 2
        self.max_search_range = np.array([4,4])
        self.min_search_range = np.array([-1.5,-3])
        self.optimal_solution = np.array([-0.54719,-1.54719])

    def get_optimal_solution(self):
        return self.optimal_solution

    def get_search_range(self):
        return [self.max_search_range, self.min_search_range]

    def get_func_val(self, variables):
        tmp1 = np.sin(variables[0]+variables[1])+np.power(variables[0]-variables[1],2)
        tmp2 = -1.5*variables[0]+2.5*variables[1]+1
        return tmp1+tmp2

    def plot_2dimension(self):
        if self.variable_num == 2:
            x = np.arange(self.min_search_range[0],self.max_search_range[0], 0.25)
            y = np.arange(self.min_search_range[1],self.max_search_range[1], 0.25)
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

##### Class Schaffer function N.2 #####
class SchafferN2:
    def __init__(self):
        self.variable_num = 2
        self.max_search_range = np.array([100,100])
        self.min_search_range = np.array([-100,-100])
        self.optimal_solution = np.array([0,0])

    def get_optimal_solution(self):
        return self.optimal_solution

    def get_search_range(self):
        return [self.max_search_range, self.min_search_range]

    def get_func_val(self, variables):
        tmp1 = np.power(np.sin(np.power(variables[0],2)-np.power(variables[1],2)),2)-0.5
        tmp2 = np.power(1+0.001*(np.power(variables[0],2)+np.power(variables[1],2)),2)
        return 0.5+tmp1/tmp2

    def plot_2dimension(self):
        if self.variable_num == 2:
            x = np.arange(self.min_search_range[0],self.max_search_range[0], 1)
            y = np.arange(self.min_search_range[1],self.max_search_range[1], 1)
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

##### Class Schaffer function N.4 #####
class SchafferN4:
    def __init__(self):
        self.variable_num = 2
        self.max_search_range = np.array([100,100])
        self.min_search_range = np.array([-100,-100])
        self.optimal_solution = np.array([0,1.25313])

    def get_optimal_solution(self):
        return self.optimal_solution

    def get_search_range(self):
        return [self.max_search_range, self.min_search_range]

    def get_func_val(self, variables):
        tmp1 = np.power(np.cos(np.sin(np.absolute(np.power(variables[0],2)-np.power(variables[1],2)))),2)-0.5
        tmp2 = np.power(1+0.001*(np.power(variables[0],2)+np.power(variables[1],2)),2)
        return 0.5+tmp1/tmp2

    def plot_2dimension(self):
        if self.variable_num == 2:
            x = np.arange(self.min_search_range[0],self.max_search_range[0], 1)
            y = np.arange(self.min_search_range[1],self.max_search_range[1], 1)
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

##### Class Styblinski-Tang function #####
class StyblinskiTang:
    def __init__(self,variable_num):
        self.variable_num = variable_num
        self.max_search_range = np.array([5] * variable_num)
        self.min_search_range = np.array([-5] * variable_num)
        self.optimal_solution = np.array([-2.903534] * variable_num)

    def get_optimal_solution(self):
        return self.optimal_solution

    def get_search_range(self):
        return [self.max_search_range, self.min_search_range]

    def get_func_val(self, variables):
        tmp1 = 0
        for i in range(self.variable_num):
        	tmp1 += np.power(variables[i],4)-16*np.power(variables[i],2)+5*variables[i]
        return tmp1/2

    def plot_2dimension(self):
        if self.variable_num == 2:
            x = np.arange(self.min_search_range[0],self.max_search_range[0], 0.25)
            y = np.arange(self.min_search_range[1],self.max_search_range[1], 0.25)
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

##### Class De Jong's function F1 #####
class DeJongsF1(Sphere):
    def __init__(self,variable_num):
        super().__init__(variable_num)

##### Class De Jong's function F2 #####
class DeJongsF2(Rosenbrock):
    def __init__(self,variable_num):
        super().__init__(variable_num)

##### Class De Jong's function F3 #####
class DeJongsF3:
    def __init__(self):
        self.variable_num = 5
        self.max_search_range = np.array([5.12] * self.variable_num)
        self.min_search_range = np.array([-5.12] * self.variable_num)
        self.optimal_solution = np.array([-5.12] * self.variable_num)

    def get_optimal_solution(self):
        return self.optimal_solution

    def get_search_range(self):
        return [self.max_search_range, self.min_search_range]

    def get_func_val(self, variables):
        tmp1 = 0
        for i in range(self.variable_num):
        	tmp1 += np.floor(variables[i])
        return tmp1

    def plot_2dimension(self):
        x = np.arange(self.min_search_range[0],self.max_search_range[0], 0.25)
        y = np.arange(self.min_search_range[1],self.max_search_range[1], 0.25)
        X, Y = np.meshgrid(x,y)
        Z = []
        for xy_list in zip(X,Y):
            z = []
            for xy_input in zip(xy_list[0],xy_list[1]):
                tmp = list(xy_input)
                tmp.extend(list(self.optimal_solution[0:self.variable_num-2]))
                z.append(self.get_func_val(tmp))
            Z.append(z)
        Z = np.array(Z)
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.plot_wireframe(X,Y,Z)
        plt.show()

##### Class De Jong's function F4 #####
class DeJongsF4:
    def __init__(self):
        self.variable_num = 30
        self.max_search_range = np.array([1.28] * self.variable_num)
        self.min_search_range = np.array([-1.28] * self.variable_num)
        self.optimal_solution = np.zeros([1,self.variable_num])

    def get_optimal_solution(self):
        return self.optimal_solution

    def get_search_range(self):
        return [self.max_search_range, self.min_search_range]

    def get_func_val(self, variables):
        tmp1 = 0
        for i in range(self.variable_num):
        	tmp1 += (i+1)*np.power(variables[i],4)
        return tmp1 + np.random.normal(0, 1)

    def plot_2dimension(self):
        x = np.arange(self.min_search_range[0],self.max_search_range[0], 0.1)
        y = np.arange(self.min_search_range[1],self.max_search_range[1], 0.1)
        X, Y = np.meshgrid(x,y)
        Z = []
        for xy_list in zip(X,Y):
            z = []
            for xy_input in zip(xy_list[0],xy_list[1]):
                tmp = list(xy_input)
                tmp.extend(list(self.optimal_solution[0,0:self.variable_num-2]))
                z.append(self.get_func_val(tmp))
            Z.append(z)
        Z = np.array(Z)
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.plot_wireframe(X,Y,Z)
        plt.show()

##### Class De Jong's function F5 #####
class DeJongsF5:
    def __init__(self):
        self.variable_num = 25
        self.max_search_range = np.array([65.536] * self.variable_num)
        self.min_search_range = np.array([-65.536] * self.variable_num)
        self.optimal_solution = np.array([-32.32]*self.variable_num)

    def get_optimal_solution(self):
        return self.optimal_solution

    def get_search_range(self):
        return [self.max_search_range, self.min_search_range]

    def get_func_val(self, variables):
        A = np.zeros([2,25])
        a = [-32,16,0,16,32]
        A[0,:] = np.tile(a,(1,5))
        tmp = []
        for x in a:
            tmp_list = [x]*5
            tmp.extend(tmp_list)
        A[1,:] = tmp

        sum = 0
        for i in range(self.variable_num):
            a1i = A[0,i]
            a2i = A[1,i]
            term1 = i
            term2 = np.power(variables[0]-a1i,6)
            term3 = np.power(variables[1]-a2i,6)
            new = 1/(term1+term2+term3)
            sum += new
        return 1/(0.002+sum)

    def plot_2dimension(self):
        x = np.arange(self.min_search_range[0],self.max_search_range[0], 1)
        y = np.arange(self.min_search_range[1],self.max_search_range[1], 1)
        X, Y = np.meshgrid(x,y)
        Z = []
        for xy_list in zip(X,Y):
            z = []
            for xy_input in zip(xy_list[0],xy_list[1]):
                tmp = list(xy_input)
                tmp.extend(list(self.optimal_solution[0:self.variable_num-2]))
                z.append(self.get_func_val(tmp))
            Z.append(z)
        Z = np.array(Z)
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.plot_wireframe(X,Y,Z)
        plt.show()

##### Class Ellipsoid function #####
class Ellipsoid:
    def __init__(self,variable_num):
        self.variable_num =variable_num
        self.max_search_range = np.array([5.12] * self.variable_num)
        self.min_search_range = np.array([-5.12] * self.variable_num)
        self.optimal_solution = np.array([0]*self.variable_num)

    def get_optimal_solution(self):
        return self.optimal_solution

    def get_search_range(self):
        return [self.max_search_range, self.min_search_range]

    def get_func_val(self, variables):
        tmp = 0
        for i in range(self.variable_num):
            tmp += np.power(np.power(1000,i/(self.variable_num-1))*variables[i],2)
        return tmp

    def plot_2dimension(self):
        x = np.arange(self.min_search_range[0],self.max_search_range[0], 1)
        y = np.arange(self.min_search_range[1],self.max_search_range[1], 1)
        X, Y = np.meshgrid(x,y)
        Z = []
        for xy_list in zip(X,Y):
            z = []
            for xy_input in zip(xy_list[0],xy_list[1]):
                tmp = list(xy_input)
                tmp.extend(list(self.optimal_solution[0:self.variable_num-2]))
                z.append(self.get_func_val(tmp))
            Z.append(z)
        Z = np.array(Z)
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.plot_wireframe(X,Y,Z)
        plt.show()

##### Class k-tablet function #####
class kTablet:
    def __init__(self,variable_num):
        self.variable_num =variable_num
        self.max_search_range = np.array([5.12] * self.variable_num)
        self.min_search_range = np.array([-5.12] * self.variable_num)
        self.optimal_solution = np.array([0]*self.variable_num)

    def get_optimal_solution(self):
        return self.optimal_solution

    def get_search_range(self):
        return [self.max_search_range, self.min_search_range]

    def get_func_val(self, variables):
        tmp = 0
        k = int(self.variable_num/4)
        for i in range(k):
            tmp += variables[i]

        for i in range(k,self.variable_num):
            tmp += np.power(100*variables[i],2)
        return tmp

    def plot_2dimension(self):
        x = np.arange(self.min_search_range[0],self.max_search_range[0], 1)
        y = np.arange(self.min_search_range[1],self.max_search_range[1], 1)
        X, Y = np.meshgrid(x,y)
        Z = []
        for xy_list in zip(X,Y):
            z = []
            for xy_input in zip(xy_list[0],xy_list[1]):
                tmp = list(xy_input)
                tmp.extend(list(self.optimal_solution[0:self.variable_num-2]))
                z.append(self.get_func_val(tmp))
            Z.append(z)
        Z = np.array(Z)
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.plot_wireframe(X,Y,Z)
        plt.show()

##### Class Five-well potential function #####
class fiveWellPotential:
    def __init__(self):
        self.variable_num = 2
        self.max_search_range = np.array([20] * self.variable_num)
        self.min_search_range = np.array([-20] * self.variable_num)
        self.optimal_solution = np.array([4.92,-9.89])

    def get_optimal_solution(self):
        return self.optimal_solution

    def get_search_range(self):
        return [self.max_search_range, self.min_search_range]

    def get_func_val(self, variables):
        tmp1 = []
        tmp1.append(1-1/(1+0.05*np.power(np.power(variables[0],2)+(variables[1]-10),2)))
        tmp1.append(-1/(1+0.05*(np.power(variables[0]-10,2)+np.power(variables[1],2))))
        tmp1.append(-1/(1+0.03*(np.power(variables[0]+10,2)+np.power(variables[1],2))))
        tmp1.append(-1/(1+0.05*(np.power(variables[0]-5,2)+np.power(variables[1]+10,2))))
        tmp1.append(-1/(1+0.1*(np.power(variables[0]+5,2)+np.power(variables[1]+10,2))))
        tmp1_sum = 0
        for x in tmp1:
            tmp1_sum += x
        tmp2 = 1+0.0001*np.power((np.power(variables[0],2)+np.power(variables[1],2)),1.2)
        return tmp1_sum*tmp2

    def plot_2dimension(self):
        x = np.arange(self.min_search_range[0],self.max_search_range[0], 0.25)
        y = np.arange(self.min_search_range[1],self.max_search_range[1], 0.25)
        X, Y = np.meshgrid(x,y)
        Z = []
        for xy_list in zip(X,Y):
            z = []
            for xy_input in zip(xy_list[0],xy_list[1]):
                tmp = list(xy_input)
                tmp.extend(list(self.optimal_solution[0:self.variable_num-2]))
                z.append(self.get_func_val(tmp))
            Z.append(z)
        Z = np.array(Z)
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.plot_wireframe(X,Y,Z)
        plt.show()

##### Class Weighted Sphere function or hyper ellipsodic function #####
class weightedSphere:
    def __init__(self,variable_num):
        self.variable_num = variable_num
        self.max_search_range = np.array([5.12] * self.variable_num)
        self.min_search_range = np.array([-5.12] * self.variable_num)
        self.optimal_solution = np.array([0] * self.variable_num)

    def get_optimal_solution(self):
        return self.optimal_solution

    def get_search_range(self):
        return [self.max_search_range, self.min_search_range]

    def get_func_val(self, variables):
        tmp = 0
        for i in range(self.variable_num):
            tmp += (i+1)*np.power(variables[i],2)
        return tmp

    def plot_2dimension(self):
        x = np.arange(self.min_search_range[0],self.max_search_range[0], 0.25)
        y = np.arange(self.min_search_range[1],self.max_search_range[1], 0.25)
        X, Y = np.meshgrid(x,y)
        Z = []
        for xy_list in zip(X,Y):
            z = []
            for xy_input in zip(xy_list[0],xy_list[1]):
                tmp = list(xy_input)
                tmp.extend(list(self.optimal_solution[0:self.variable_num-2]))
                z.append(self.get_func_val(tmp))
            Z.append(z)
        Z = np.array(Z)
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.plot_wireframe(X,Y,Z)
        plt.show()

class hyperEllipsodic(weightedSphere):
    def __init__(self,variable_num):
        super().__init__(variable_num)

##### Class Sum of different power function #####
class sumOfDifferentPower:
    def __init__(self,variable_num):
        self.variable_num = variable_num
        self.max_search_range = np.array([1] * self.variable_num)
        self.min_search_range = np.array([-1] * self.variable_num)
        self.optimal_solution = np.array([0] * self.variable_num)

    def get_optimal_solution(self):
        return self.optimal_solution

    def get_search_range(self):
        return [self.max_search_range, self.min_search_range]

    def get_func_val(self, variables):
        tmp = 0
        for i in range(self.variable_num):
            tmp += np.power(np.absolute(variables[i]),i+2)
        return tmp

    def plot_2dimension(self):
        x = np.arange(self.min_search_range[0],self.max_search_range[0], 0.25)
        y = np.arange(self.min_search_range[1],self.max_search_range[1], 0.25)
        X, Y = np.meshgrid(x,y)
        Z = []
        for xy_list in zip(X,Y):
            z = []
            for xy_input in zip(xy_list[0],xy_list[1]):
                tmp = list(xy_input)
                tmp.extend(list(self.optimal_solution[0:self.variable_num-2]))
                z.append(self.get_func_val(tmp))
            Z.append(z)
        Z = np.array(Z)
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.plot_wireframe(X,Y,Z)
        plt.show()

##### Class Griewank function #####
class griewank:
    def __init__(self,variable_num):
        self.variable_num = variable_num
        self.max_search_range = np.array([600] * self.variable_num)
        self.min_search_range = np.array([-600] * self.variable_num)
        self.optimal_solution = np.array([0] * self.variable_num)

    def get_optimal_solution(self):
        return self.optimal_solution

    def get_search_range(self):
        return [self.max_search_range, self.min_search_range]

    def get_func_val(self, variables):
        tmp1 = 0
        tmp2 = 1
        for i in range(self.variable_num):
            tmp1 += np.power(variables[i],2)
            tmp2 = tmp2*np.cos(variables[i]/np.sqrt(i+1))
        return tmp1/4000-tmp2

    def plot_2dimension(self):
        x = np.arange(self.min_search_range[0],self.max_search_range[0], 10)
        y = np.arange(self.min_search_range[1],self.max_search_range[1], 10)
        X, Y = np.meshgrid(x,y)
        Z = []
        for xy_list in zip(X,Y):
            z = []
            for xy_input in zip(xy_list[0],xy_list[1]):
                tmp = list(xy_input)
                tmp.extend(list(self.optimal_solution[0:self.variable_num-2]))
                z.append(self.get_func_val(tmp))
            Z.append(z)
        Z = np.array(Z)
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.plot_wireframe(X,Y,Z)
        plt.show()

##### Class Michalewicz function #####
class michalewicz:
    def __init__(self,variable_num):
        self.variable_num = variable_num
        self.max_search_range = np.array([np.pi] * self.variable_num)
        self.min_search_range = np.array([0] * self.variable_num)
        self.optimal_solution = np.array([0] * self.variable_num)

    # def get_optimal_solution(self):
    #     return self.optimal_solution

    def get_search_range(self):
        return [self.max_search_range, self.min_search_range]

    def get_func_val(self, variables):
        m = 10
        tmp1 = 0
        for i in range(self.variable_num):
            tmp1 += np.sin(variables[i])*np.power(np.sin((i+1)*np.power(variables[i],2)/np.pi),2*m)
        return -tmp1

    def plot_2dimension(self):
        x = np.arange(self.min_search_range[0],self.max_search_range[0], 0.1)
        y = np.arange(self.min_search_range[1],self.max_search_range[1], 0.1)
        X, Y = np.meshgrid(x,y)
        Z = []
        for xy_list in zip(X,Y):
            z = []
            for xy_input in zip(xy_list[0],xy_list[1]):
                tmp = list(xy_input)
                tmp.extend(list(self.optimal_solution[0:self.variable_num-2]))
                z.append(self.get_func_val(tmp))
            Z.append(z)
        Z = np.array(Z)
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.plot_wireframe(X,Y,Z)
        plt.show()

##### Class Perm function #####
class perm:
    def __init__(self,variable_num,beta):
        self.variable_num = variable_num
        self.beta = beta
        self.max_search_range = np.array([1] * self.variable_num)
        self.min_search_range = np.array([-1] * self.variable_num)
        tmp = []
        for i in range(self.variable_num):
            tmp.append(1/(i+1))
        self.optimal_solution = np.array(tmp)

    def get_optimal_solution(self):
        return self.optimal_solution

    def get_search_range(self):
        return [self.max_search_range, self.min_search_range]

    def get_func_val(self, variables):
        tmp1 = 0
        tmp2 = 0
        for j in range(self.variable_num):
            for i in range(self.variable_num):
                tmp1 += (i+1+self.beta)*(np.power(variables[i],j+1)-np.power(1/(i+1),j+1))
            tmp2 += np.power(tmp1,2)
            tmp1 = 0
        return tmp2

    def plot_2dimension(self):
        x = np.arange(self.min_search_range[0],self.max_search_range[0], 0.1)
        y = np.arange(self.min_search_range[1],self.max_search_range[1], 0.1)
        X, Y = np.meshgrid(x,y)
        Z = []
        for xy_list in zip(X,Y):
            z = []
            for xy_input in zip(xy_list[0],xy_list[1]):
                tmp = list(xy_input)
                tmp.extend(list(self.optimal_solution[0:self.variable_num-2]))
                z.append(self.get_func_val(tmp))
            Z.append(z)
        Z = np.array(Z)
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.plot_wireframe(X,Y,Z)
        plt.show()

##### Class Rastrigin function #####
class rastrigin:
    def __init__(self,variable_num):
        self.variable_num = variable_num
        self.max_search_range = np.array([5.12] * self.variable_num)
        self.min_search_range = np.array([-5.12] * self.variable_num)
        self.optimal_solution = np.array([0] * self.variable_num)

    def get_optimal_solution(self):
        return self.optimal_solution

    def get_search_range(self):
        return [self.max_search_range, self.min_search_range]

    def get_func_val(self, variables):
        tmp1 = 10 * self.variable_num
        tmp2 = 0
        for i in range(self.variable_num):
            tmp2 += np.power(variables[i],2)-10*np.cos(2*np.pi*variables[i])
        return tmp1+tmp2

    def plot_2dimension(self):
        x = np.arange(self.min_search_range[0],self.max_search_range[0], 0.1)
        y = np.arange(self.min_search_range[1],self.max_search_range[1], 0.1)
        X, Y = np.meshgrid(x,y)
        Z = []
        for xy_list in zip(X,Y):
            z = []
            for xy_input in zip(xy_list[0],xy_list[1]):
                tmp = list(xy_input)
                tmp.extend(list(self.optimal_solution[0:self.variable_num-2]))
                z.append(self.get_func_val(tmp))
            Z.append(z)
        Z = np.array(Z)
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.plot_wireframe(X,Y,Z)
        plt.show()

##### Class Schwefel function #####
class schwefel:
    def __init__(self,variable_num):
        self.variable_num = variable_num
        self.max_search_range = np.array([500] * self.variable_num)
        self.min_search_range = np.array([-500] * self.variable_num)
        self.optimal_solution = np.array([420.9687] * self.variable_num)

    def get_optimal_solution(self):
        return self.optimal_solution

    def get_search_range(self):
        return [self.max_search_range, self.min_search_range]

    def get_func_val(self, variables):
        tmp = 0
        for i in range(self.variable_num):
            tmp += variables[i]*np.sin(np.sqrt(np.absolute(variables[i])))
        return -tmp

    def plot_2dimension(self):
        x = np.arange(self.min_search_range[0],self.max_search_range[0], 1)
        y = np.arange(self.min_search_range[1],self.max_search_range[1], 1)
        X, Y = np.meshgrid(x,y)
        Z = []
        for xy_list in zip(X,Y):
            z = []
            for xy_input in zip(xy_list[0],xy_list[1]):
                tmp = list(xy_input)
                tmp.extend(list(self.optimal_solution[0:self.variable_num-2]))
                z.append(self.get_func_val(tmp))
            Z.append(z)
        Z = np.array(Z)
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.plot_wireframe(X,Y,Z)
        plt.show()

##### Class Six-hump camel function #####
class sixHumpCamel:
    def __init__(self):
        self.variable_num = 2
        self.max_search_range = np.array([3,2])
        self.min_search_range = np.array([-3,-2])
        self.optimal_solution = np.array([0.0898])

    def get_optimal_solution(self):
        return self.optimal_solution

    def get_search_range(self):
        return [self.max_search_range, self.min_search_range]

    def get_func_val(self, variables):
        return 4-2.1*np.power(variables[0],2)+1/3*np.power(variables[0],4)*np.power(variables[0],2)+variables[0]*variables[1]+4*(np.power(variables[1],2)-1)*np.power(variables[1],2)

    def plot_2dimension(self):
        x = np.arange(self.min_search_range[0],self.max_search_range[0], 0.25)
        y = np.arange(self.min_search_range[1],self.max_search_range[1], 0.25)
        X, Y = np.meshgrid(x,y)
        Z = []
        for xy_list in zip(X,Y):
            z = []
            for xy_input in zip(xy_list[0],xy_list[1]):
                tmp = list(xy_input)
                tmp.extend(list(self.optimal_solution[0:self.variable_num-2]))
                z.append(self.get_func_val(tmp))
            Z.append(z)
        Z = np.array(Z)
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.plot_wireframe(X,Y,Z)
        plt.show()

##### Class Shuberts function #####
class shuberts:
    def __init__(self):
        self.variable_num = 2
        self.max_search_range = np.array([1000,10]) # Set infinite as 1000 for x1
        self.min_search_range = np.array([-10,-1000]) # Set infinite as -1000 for x2
        self.optimal_solution = np.array([0,0]) # Not yet implemented

    def get_optimal_solution(self):
        return self.optimal_solution

    def get_search_range(self):
        return [self.max_search_range, self.min_search_range]

    def get_func_val(self, variables):
        n = 5
        tmp1 = 0
        tmp2 = 0
        for i in range(n):
            tmp1 += (i+1)*np.cos((i+1)+(i+2)*variables[0])
            tmp2 += (i+1)*np.cos((i+1)+(i+2)*variables[1])
        return tmp1*tmp2

    def plot_2dimension(self):
        x = np.arange(self.min_search_range[0],self.max_search_range[0], 10)
        y = np.arange(self.min_search_range[1],self.max_search_range[1], 10)
        X, Y = np.meshgrid(x,y)
        Z = []
        for xy_list in zip(X,Y):
            z = []
            for xy_input in zip(xy_list[0],xy_list[1]):
                tmp = list(xy_input)
                tmp.extend(list(self.optimal_solution[0:self.variable_num-2]))
                z.append(self.get_func_val(tmp))
            Z.append(z)
        Z = np.array(Z)
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.plot_wireframe(X,Y,Z)
        plt.show()

def main():
    benchmark_func = shuberts()
    benchmark_func.plot_2dimension()

if __name__ == '__main__':
    main()
