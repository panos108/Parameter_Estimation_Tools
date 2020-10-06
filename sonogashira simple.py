import matplotlib.pyplot as plt
import numpy as np
from PE_utilities import *
from PE_models import *
import pandas as pd
file_name = 'Sonogashira_data/E19-010426 Pareto results of sonogashira coupling reaction (graeme and adam c).xlsx'

dfs = pd.read_excel(file_name, sheet_name=None)
xl = np.array(dfs['Sheet1'])

X = np.array(xl[-80:99, [1,2,3]])
F = np.array(xl[-80:99, [9,11,13]])


x_meas = np.zeros([80,4,2])
u_meas = np.zeros([80,4,1])
x_meas[:,0,0]   = xl[-80:99, 7]
x_meas[:,3,0]   = xl[-80:99, 8]
x_meas[:,:-1,1] = xl[-80:99, [10, 12, 14]]
u_meas[:,:,0]   = xl[-80:99, 3:7]
dt = np.array(xl[-80:99, [1]])
model = Reactor_pfr_model_Sonogashira(normalize=[0.04421972, 0.23511237, 0.08936344, 0.])
pe = ParameterEstimation(model)




u_opt, x_opt, w_opt, x_pred, theta = pe.solve_pe(x_meas, u_meas, dt)

CI, t, t_ref, chi2, statistics = pe.Confidence_intervals()



print(2)
