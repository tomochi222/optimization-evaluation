
# coding : utf-8

import os
from opteval.benchmark_func import *
import opteval.benchmark_func as bf

def main():
    print('Available function list is \n', bf.__all__)
    print('One arguments function list is \n', bf.__oneArgument__)
    print('Two arguments function list is \n', bf.__twoArgument__)
    print('Three arguments function list is \n', bf.__threeArgument__)    
    for func_name in bf.__oneArgument__:
        instance = eval(func_name)()
        print('Now saving figure is ',func_name,' ...')
        instance.save_fig()
    for func_name in bf.__twoArgument__:
        print('Now saving figure is ',func_name,' ...')
        instance = eval(func_name)(2)
        instance.save_fig()
    for func_name in bf.__threeArgument__:
        print('Now saving figure is ',func_name,' ...')
        instance = eval(func_name)(2,0.5)
        instance.save_fig()

if __name__ == '__main__':
    main()
