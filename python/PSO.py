
# coding: utf-8

import sys
import numpy as np

##### Class dealing with particles #####
class PSOclass:
    ##### Inner class on each particle #####
    class particles:
        def __init__(self, set_dim):
            ### Information each particle ###
            self.position = np.zeros(set_dim)
            self.velocity = np.zeros(set_dim)
            self.update_fitness()
            self.bestPosition = np.zeros(set_dim)
            self.bestFitness = sys.maxsize

        def update_fitness(self):
            self.fitness = 3.0 + pow(self.position[0],2) + pow(self.position[1],2)

        def disp_info(self):
            print('--\n Information of particle: ')
            print(' position = ' + str(self.position))
            print(' velocity = ' + str(self.velocity))
            print(' fitness = ' + str(self.fitness))
            print(' bestPosition = ' + str(self.bestPosition))
            print(' bestFitness = ' + str(self.bestFitness))

    def __init__(self, pp={}):
        if len(pp) == 0:
            ### PSO parameters ###
            self.__num_particles = 10                               # Number of perticle members
            self.__num_iterations = 10000                           # Number of iteration
            self.__dim = 2                                          # Number of solutions to calculate
            self.__minX = np.array([-100]*self.__dim)               # Minimum boundary of position solutions
            self.__maxX = np.array([100]*self.__dim)                # Maximum boundary of position solutions
            self.__minV = np.array([-1.0*i/5 for i in self.__maxX]) # Minimum boundary of velocity solutions
            self.__maxV = np.array([1.0*i/5 for i in self.__maxX])  # Maximum boundary of velocity solutions
            self.__best_global_posi = np.zeros(self.__dim)          # Optimal particle positions in all group
            self.__best_global_fit = sys.maxsize                    # Fitness of position
            self.__w = 0.7                                          # Inertial constant (recommended value)
            self.__c1 = 1.49445                                     # Local mass of particles (recommended value)
            self.__c2 = 1.49445                                     # Global mass of particles (recommended value)
            self.disp_param()
        else:
            self.set_param(pp)

        # Generate particle group
        self.ps = [PSOclass.particles(self.__dim) for i in range(self.__num_particles)]

    def set_param(self, pp):
        ### PSO parameters ###
        if "num_particles" in pp:
            self.__num_particles = pp["num_particles"]
        if "num_iterations" in pp:
            self.__num_iterations = pp["num_iterations"]
        if "dim" in pp:
            self.__dim = pp["dim"]
        if "minX" in pp:
            self.__minX = pp["minX"]
        if "maxX" in pp:
            self.__maxX = pp["maxX"]
        if "minV" in pp:
            self.__minV = pp["minV"]
        if "maxV" in pp:
            self.__maxV = pp["maxV"]
        if "best_global_posi" in pp:
            self.__best_global_posi = pp["best_global_posi"]
        if "best_global_fit" in pp:
            self.__best_global_fit = pp["best_global_fit"]
        if "w" in pp:
            self.__w = pp["w"]
        if "c1" in pp:
            self.__c1 = pp["c1"]
        if "c2" in pp:
            self.__c2 = pp["c2"]

        self.disp_param()

    def disp_param(self):
        print('--\n PSO parameters are setted as follows: ')
        print(' Number of particles = ' + str(self.__num_particles))
        print(' Number of iterations = ' + str(self.__num_iterations))
        print(' Dimension of parameters = ' + str(self.__dim))
        print(' Minimum limit of parameters = ' + str(self.__minX))
        print(' Maximum limit of solutions = ' + str(self.__maxX))
        print(' Minimum limit of velocities = ' + str(self.__minV))
        print(' Maximum limit of velocities = ' + str(self.__maxV))
        print(' Best global positions = ' + str(self.__best_global_posi))
        print(' Best global fitnesses = ' + str(self.__best_global_fit))
        print(' Inertial constant = ' + str(self.__w))
        print(' Local mass of particles = ' + str(self.__c1))
        print(' Global mass of particles = ' + str(self.__c2))

    ### Calculate fitness to optimize ###
    def get_fitness(self):
        return self.__fitness

    ### disp current best global fitness and best global position
    def disp_global_optimal(self):
        print(' Best global fitness = ' + str(self.__best_global_fit))
        print(' Best global position = ' + str(self.__best_global_posi))

    ##### Initilize particle group #####
    def initialize_particle(self):
        rap = np.zeros(self.__dim)   # random position
        rav = np.zeros(self.__dim)   # random velocity

        for i in range(self.__num_particles):
            # Initializing positions and velocities
            for j in range(self.__dim):
                rap[j] = abs(self.__maxX[j] - self.__minX[j]) * np.random.rand() + self.__minX[j]
                rav[j] = abs(self.__maxV[j] - self.__minV[j]) * np.random.rand() + self.__minV[j]

            self.ps[i].position = rap.copy()
            self.ps[i].velocity = rav.copy()

            # fitness in current random position
            self.ps[i].update_fitness()
            self.ps[i].bestPosition = rap.copy()
            self.ps[i].bestFitness = self.ps[i].fitness

            if self.ps[i].fitness<self.__best_global_fit:
                self.__best_global_fit = self.ps[i].fitness
                self.__best_global_posi = self.ps[i].position.copy()

        print('--\n >> Finished initializing particles')
        self.disp_global_optimal()

    def optimize_particle(self):
        newp = np.zeros(self.__dim)
        newv = np.zeros(self.__dim)

        print('--\n >> Start optimizing particles')

        update_cnt = 0
        for ite in range(self.__num_iterations):
            for i in range(self.__num_particles):

                ### Updating velocities ###
                for j in range(self.__dim):
                    # Velocity upating formula in PSO
                    newv[j] = ((self.__w * self.ps[i].velocity[j]) +\
                        (self.__c1 * np.random.rand() * (self.ps[i].bestPosition[j] - self.ps[i].position[j])) +\
                        (self.__c2 * np.random.rand() * (self.__best_global_posi[j] - self.ps[i].position[j])))
                    # Check if updated velocity is between minimum value and maximum value
                    if newv[j] < self.__minV[j]:
                        newv[j] = self.__minV[j]
                    elif newv[j] > self.__maxV[j]:
                        newv[j] = self.__maxV[j]
                    # Position upating formula in PSO
                    newp[j] = self.ps[i].position[j] + newv[j]
                    # Check if updated position is between minimum value and maximum value
                    if newp[j] < self.__minX[j]:
                        newp[j] = self.__minX[j]
                    elif newp[j] > self.__maxX[j]:
                        newp[j] = self.__maxX[j]

                # Update velocity and position of particle object
                self.ps[i].velocity = newv.copy()
                self.ps[i].position = newp.copy()

                # Update best fitness
                self.ps[i].update_fitness()
                if self.ps[i].fitness < self.ps[i].bestFitness:
                    self.ps[i].bestPosition = newp.copy()
                    self.ps[i].bestFitness = self.ps[i].fitness
                if self.ps[i].fitness < self.__best_global_fit:
                    update_cnt += 1
                    self.__best_global_posi = newp.copy()
                    self.__best_global_fit = self.ps[i].fitness
                    print('\n >> Update optimal solution')
                    self.disp_global_optimal()

        print('--\n >> Finished optimizing')
        self.disp_global_optimal()

def main():
    pso_param = {"num_particles":5,
                "num_iterations":50,
                "dim":2,
                "best_global_fit":sys.maxsize,
                "w":0.7,
                "c1":1.49445,
                "c2":1.49445}
    pso_param.update(
        {"best_global_posi":np.zeros(pso_param["dim"]),
         "minX":np.array([-100]*pso_param["dim"]),
         "maxX":np.array([100]*pso_param["dim"])})
    pso_param.update(
         {"maxV":np.array([1.0*i/5 for i in pso_param["maxX"]]),
          "minV":np.array([-1.0*i/5 for i in pso_param["maxX"]])})

    ##### PSO #####
    pps = PSOclass(pso_param)
    pps.initialize_particle()
    pps.optimize_particle()

if __name__ == '__main__':
    main()
