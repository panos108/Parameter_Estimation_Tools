

import matplotlib.pyplot as plt
import numpy as np
from PE_utilities import *
from PE_models import *
import pandas as pd
from ut import *
import operator
import random


import math
import functools

from deap import base
from deap import benchmarks
from deap import creator
from deap import tools

file_name = 'Sonogashira_data/E19-010426 Pareto results of sonogashira coupling reaction (graeme and adam c).xlsx'

dfs = pd.read_excel(file_name, sheet_name=None)
xl = np.array(dfs['Sheet1'])

X = np.array(xl[-80:99, [1,2,3]])
F = np.array(xl[-80:99, [9,11,13]])

N = 70
x_meas = np.zeros([N,14,2])+1e-8
u_meas = np.zeros([N,4,1])
x_meas[:,0,0]   = xl[-N:99, 7]
x_meas[:,3,0]   = xl[-N:99, 8]
x_meas[:,:3,1]  = xl[-N:99, [10, 12, 14]]
x_meas[:,4,0]   = xl[-N:99,17]
x_meas[:,9,0]   = xl[-N:99,17]
# x_meas[:,6,0]   = xl[-N:99,17]
# x_meas[:,7,0]   = xl[-N:99,17]
# x_meas[:,11,0]   = xl[-N:99,17]
# x_meas[:,12,0]   = xl[-N:99,17]

u_meas[:,:,0]   = xl[-N:99, 3:7]
dt = np.array(xl[-N:99, [1]])

#model0 = Reactor_pfr_model_Sonogashira_general_order(powers=[1,2,1,2])#Reactor_pfr_model_Sonogashira_hybrid()  # normalize=x_meas[:,:,1].max(0))
model0 = Reactor_pfr_model_Sonogashira_general()#Reactor_pfr_model_Sonogashira_general_order_polynomial(powers=[1,2,1,2])#Reactor_pfr_model_Sonogashira_hybrid()  # normalize=x_meas[:,:,1].max(0))

f, _, _, _ = model0.Generate_fun_integrator()
pe = ParameterEstimation(model0, print_level=5,generate_problem=False)

p_in, x0_exp_pe, x_exp_pe, u_exp_pe, dt_pe, shrink = pe.prepare_for_pe(x_meas, u_meas, dt)
objective = functools.partial(pe.Construct_total_integrator,(p_in))


def obj(objective, model0, x):
    p=0

    for i in range(model0.ntheta):
        if x[i] < model0.bounds_theta[0][i] or x[i] > model0.bounds_theta[1][i]:
            p += 100
    if p>0:
        f = p
    else:
        f = objective(x)
    return np.array([f])

obj_to_pso = functools.partial(obj,objective, model0)


creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Particle", list, fitness=creator.FitnessMin, speed=list,
    smin=None, smax=None, best=None)


def generate(size, pmin, pmax, smin, smax):
    part = creator.Particle(random.uniform(pmin, pmax) for _ in range(size))
    part.speed = [random.uniform(smin, smax) for _ in range(size)]
    part.smin = smin
    part.smax = smax
    return part


def updateParticle(part, best, phi1, phi2):
    u1 = (random.uniform(0, phi1) for _ in range(len(part)))
    u2 = (random.uniform(0, phi2) for _ in range(len(part)))
    v_u1 = map(operator.mul, u1, map(operator.sub, part.best, part))
    v_u2 = map(operator.mul, u2, map(operator.sub, best, part))
    part.speed = list(map(operator.add, part.speed, map(operator.add, v_u1, v_u2)))
    for i, speed in enumerate(part.speed):
        if abs(speed) < part.smin:
            part.speed[i] = math.copysign(part.smin, speed)
        elif abs(speed) > part.smax:
            part.speed[i] = math.copysign(part.smax, speed)
    part[:] = list(map(operator.add, part, part.speed))


toolbox = base.Toolbox()
toolbox.register("particle", generate, size=model0.ntheta, pmin=1, pmax=2, smin=-3, smax=3)
toolbox.register("population", tools.initRepeat, list, toolbox.particle)
toolbox.register("update", updateParticle, phi1=5.0, phi2=2.0)
toolbox.register("evaluate", obj_to_pso)


def main():
    pop = toolbox.population(n=10)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    logbook = tools.Logbook()
    logbook.header = ["gen", "evals"] + stats.fields

    GEN = 1000
    best = None

    for g in range(GEN):
        for part in pop:
            part.fitness.values = toolbox.evaluate(part)
            if not part.best or part.best.fitness < part.fitness:
                part.best = creator.Particle(part)
                part.best.fitness.values = part.fitness.values
            if not best or best.fitness < part.fitness:
                best = creator.Particle(part)
                best.fitness.values = part.fitness.values
        for part in pop:
            toolbox.update(part, best)

        # Gather all the fitnesses in one list and print the stats
        logbook.record(gen=g, evals=len(pop), **stats.compile(pop))
        print(logbook.stream)

    return pop, logbook, best


pop, logbook, best = main()




print(2)
