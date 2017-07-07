
# coding: utf-8

import benchmark_func as bf

def main():
    print(bf.__all__)
    test = bf.SchafferN2()
    test.plot()

if __name__ == '__main__':
    main()
