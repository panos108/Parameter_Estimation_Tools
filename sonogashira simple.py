import matplotlib.pyplot as plt
import numpy as np
from PE_utilities import *
from PE_models import *
import pandas as pd
from ut import *
file_name = 'Sonogashira_data/E19-010426 Pareto results of sonogashira coupling reaction (graeme and adam c).xlsx'

dfs = pd.read_excel(file_name, sheet_name=None)
xl = np.array(dfs['Sheet1'])

X = np.array(xl[-80:99, [1,2,3]])
F = np.array(xl[-80:99, [9,11,13]])

N = 50
x_meas = np.zeros([N,4,2])
u_meas = np.zeros([N,4,1])
x_meas[:,0,0]   = xl[-N:99, 7]
x_meas[:,3,0]   = xl[-N:99, 8]
x_meas[:,:-1,1] = xl[-N:99, [10, 12, 14]]
u_meas[:,:,0]   = xl[-N:99, 3:7]
dt = np.array(xl[-N:99, [1]])

model0 = Reactor_pfr_model_Sonogashira_hybrid()  # normalize=x_meas[:,:,1].max(0))

f, _, _, _ = model0.Generate_fun_integrator()

pe = ParameterEstimation(model0)

u_opt, _, w_opt, x_pred, theta, chi2, mle = pe.solve_pe(x_meas, u_meas, dt)

CI, t, t_ref, _, statistics = pe.Confidence_intervals()
print('Chi2: ', chi2)
print('mle: ', mle)
print('t: ', t)

x_meas1 = np.zeros([80,4,2])
u_meas1 = np.zeros([80,4,1])
x_meas1[:,0,0]   = xl[-80:99, 7]
x_meas1[:,3,0]   = xl[-80:99, 8]
x_meas1[:,:-1,1] = xl[-80:99, [10, 12, 14]]
u_meas1[:,:,0]   = xl[-80:99, 3:7]
dt1 = np.array(xl[-80:99, [1]])
x_his = np.zeros([80,4])

for i in range(80):
    x = x_meas1[i, :, 0]
    u = u_meas1[i, :, 0]
    dts = dt1[i]
    x1 = model0.perform_evaluation(x, u, pe.theta, dts)
    x_his[i] = np.array(x1).reshape((-1,))
x2 = np.array(x_pred)
plt.plot(x_meas1[-N:, :-1, 1], x_his.T[:-1, -N:].T, 'o')
plt.plot(x_meas1[:N, :-1, 1], x_his.T[:-1, :N].T, 'r*')

plt.plot([0, np.max(x2[:-1, :].T)], [0, np.max(x2[:-1, :].T)], 'k-')
plt.xlabel('Measured Concentration(M)')
plt.ylabel('Predicted Concentration(M)')
plt.savefig('hybrid3.png', dpi=400)
plt.close()
powers = np.array([   [1,1,1,1],
                      [1,1,1,2],
                      [1,2,1,1],
                      [2,1,1,1],
                      [2,2,1,1],
                      [1,2,1,2],
                      [2,1,2,1],
                      [2,1,1,2],
                      [2,2,2,1],
                      [1,2,2,2],
                      [2,2,1,2]])
chi2s = []
ts    = []
xs    = []
for i in range(5,6):
    model1 = Reactor_pfr_model_Sonogashira_general_order(powers=powers[i])#normalize=x_meas[:,:,1].max(0))

    f,_,_,_ = model1.Generate_fun_integrator()

    pe = ParameterEstimation(model1)

    u_opt, _, w_opt, x_pred, theta, chi2, mle = pe.solve_pe(x_meas, u_meas, dt)

    CI, t, t_ref, _, statistics = pe.Confidence_intervals()
    chi2s +=[chi2]
    print('Iteration: ', i)
    print('Chi2: ', chi2)
    print('mle: ', mle)
    print('t: ', t)

    print('----------------')
    ts    +=[t]
    xs    +=[x_pred]

    for i in range(80):
        x = x_meas1[i, :, 0]
        u = u_meas1[i, :, 0]
        dts = dt1[i]
        x1 = model1.perform_evaluation(x, u, pe.theta, dts)
        x_his[i] = np.array(x1).reshape((-1,))
    x2 = np.array(x_pred)
    plt.plot(x_meas1[-N:, :-1, 1], x_his.T[:-1, -N:].T, 'o')
    plt.plot(x_meas1[:N, :-1, 1], x_his.T[:-1, :N].T, 'r*')

    plt.plot([0, np.max(x2[:-1, :].T)], [0, np.max(x2[:-1, :].T)], 'k-')
    plt.xlabel('Measured Concentration(M)')
    plt.ylabel('Predicted Concentration(M)')
    plt.savefig('poly2.png', dpi=400)


x_meas = np.zeros([80,4,2])
u_meas = np.zeros([80,4,1])
x_meas[:,0,0]   = xl[-80:99, 7]
x_meas[:,3,0]   = xl[-80:99, 8]
x_meas[:,:-1,1] = xl[-80:99, [10, 12, 14]]
u_meas[:,:,0]   = xl[-80:99, 3:7]
dt = np.array(xl[-80:99, [1]])
x0 = (x_meas[:,:,0].T)
x_meas = (x_meas[:,:,1].T)
u_meas = (u_meas[:,:,0].T)
mle = 0
s = 0

thetak = theta#pe.theta

for k_exp in range(pe.N_exp_max):
    if s > 0:
        s += 1
    # "Lift" initial conditions
    Xk = x0.T[k_exp, :].T  # MX.sym('X_' + str(s), nx)

    ms = pe.int_elem * 100
    for k in range((pe.t_span - 1)):
        if s > 0:
            s += 1
        # New NLP variable for the control
        Uk = u_meas.T[k_exp * (pe.t_span - 1) + k, :].T
        h = dt[k_exp * (pe.t_span - 1) + k] / ms
        s = 0
        for k_in in range(ms):
            # --------------------
            # State at collocation points
            # ---------------------------------

            k1, _ = f(Xk, Uk, thetak)
            k2, _ = f(Xk + h / 2 * k1, Uk, thetak)
            k3, _ = f(Xk + h / 2 * k2, Uk, thetak)
            k4, _ = f(Xk + h * k3, Uk, thetak)
            Xk = Xk + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            s += 1
        print(mle)
        mle += pe.maximum_likelihood_est(k_exp, Xk, x_meas, pe.Model_def.standard_deviation, k)

model = Reactor_pfr_model_Sonogashira()#normalize=x_meas[:,:,1].max(0))

f,_,_,_ = model.Generate_fun_integrator()

pe = ParameterEstimation(model)

u_opt, _, w_opt, x_pred, theta = pe.solve_pe(x_meas, u_meas, dt)

CI, t, t_ref, chi2, statistics = pe.Confidence_intervals()











x_0 = [0.2, 0.00, 0.00, 0.00]
lbx = [0] * 4  # [-0.25, -inf]
ubx = [inf] * 4

lbxp = [-inf] * 4 *4
ubxp = [inf] * 4 * 4

lbtheta = [-20.,0.] * 2#(ntheta[0])#[*[-20.] * (ntheta[0]-5), *[0] * 5]
ubtheta =  [40.] * (4)#[*[20.] * (ntheta[0]-5), *[2] * 5]# [60] * ntheta[0]
lbu = [u_meas[:,:,0]]  # [-1, -1]
ubu = [u_meas[:,:,0]]  # [1, 1]

problem, w0, lbw, ubw, lbg, ubg, trajectories, mle_fn = \
        construct_NLP_collocation(80, f, x_0, x_meas[:,:,0], lbx,ubx, lbu, ubu, lbtheta,
                              ubtheta, dt,
                                   1, x_meas[:, :-1, 1:],
                                  0.8+np.random.rand(4),#
                                  8, 25) #-----Change x_meas1 to x_meas ------#

solver = nlpsol('solver', 'ipopt', problem, {"ipopt.tol": 1e-12})#, {"ipopt.hessian_approximation":"limited-memory"})#, {"ipopt.tol": 1e-10, "ipopt.print_level": 0})#, {"ipopt.hessian_approximation":"limited-memory"})

# Function to get x and u trajectories

sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
w_opt = sol['x'].full().flatten()

x_opt, u_opt, xp_opt, chi2 = trajectories(sol['x'])















x2 = np.array(x_pred)
x1 = np.array(x_opt)
print(2)
