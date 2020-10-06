import numpy as np

from utilities_leeds import *#give_data_from_exp_recal, plant_model_GP, plant_model, integrator_model, give_data_from_exp, give_data_from_sim, maximum_likelihood_est, give_data_from_sim_update, compute_rf, objective_pe
from ut import construct_NLP_collocation, construct_NLP_MBDoE_collocation, bayopt_design, bayopt_pe, construct_NLP_RK4
import pandas as pd
import time
import scipy.stats as stats
from pyDOE import *
from GP_ut import *
from PE_utilities import *
from PE_models import *



np.random.seed(seed=0)

T = 200.


N_exp    = 8#5
PC       = 'Panos'
#PC       =  'ppets' #For laptop
labbot   = True
file     = ['NaN']*N_exp
date     = '28-Jan-2020/20200128-Panos'#'16-Jan-2020/20200116-Panos-2/20200116-Panos-2'#'trial/20190906_UCL_Panos_3'##'03-Sep-2019/20190903_UCL_Panos'#'07-Oct-2019/Run3/20191007_UCL_SS3'##
info     = '/Exp_Setup_Info_28-January-2020_11_26_27.csv'#'/Exp_Setup_Info_16-January-2020_14_29_54.csv'#
condition= '/Process_Conditions_Bounds_28-January-2020_11_26_27.csv'#'/Process_Conditions_Bounds_16-January-2020_14_29_54.csv'#


#date1      = '06-Sep-2019/20190906_UCL_Panos_3'#'trial/20190906_UCL_Panos_3'##'03-Sep-2019/20190903_UCL_Panos'#'07-Oct-2019/Run3/20191007_UCL_SS3'##
##info1      = '/Exp_Setup_Info_06-September-2019_11_34_19.csv'#
#condition1 = '/Process_Conditions_Bounds_06-September-2019_11_34_19.csv'#

bayesopt = True

time_to_wait = 60 * 3000
time_counter = 0
file0 = '/Users/' + PC + '/Dropbox/UCL/' + date + '/Peaks and Concentrations_' + str(N_exp) + '.csv'
while not os.path.exists(file0):
    time.sleep(1)
    time_counter += 1
    if time_counter > time_to_wait: break


if labbot==True:
    f, nu, nx, ntheta = plant_model([])
    #x_meas0, u_meas0, V0, c1o0, c2o0, dt0 = give_data_from_exp_recal(nu, nx, ntheta, N_exp, PC, date1, file, 1, info1)#Need to change dates
    x_meas, u_meas, V, c1o, c2o, dt = give_data_from_exp(nu, nx, ntheta, N_exp, PC, date, file, info)
#    compute_rf(nu, nx, ntheta, N_exp, PC, date, file)
    x_meas[0,:,:] = x_meas[1,:,:]
    u_meas[0,:] = u_meas[1,:]
    dt[0] = dt[1]
    x_meas0 = x_meas
    u_meas0 = u_meas
    dt0 = dt
else:
    true_theta = [np.log(57.9 * 60. * 10. ** (-2)), 33.3 / 10, np.log(2.7 * 60. * 10. ** (-2)), 35.3 / 10,
                  np.log(0.865 * 60. * 10. ** (-2)), 38.9 / 10, np.log(1.63 * 60. * 10. ** (-2)), 44.8 / 10]
    x_meas, u_meas, V, c1o, c2o, dt = give_data_from_sim(N_exp, PC, date, file, true_theta, info)
    f, nu, nx, ntheta = plant_model([])
#x_meas[6,:,:] = x_meas[0,:,:]
n_points = 1
s = 0
# Define Noises #

sigma = [1e-3]*(nx[0]-1)  # Assumed additive average noise
sigma0 = [1e-2]*(nx[0]-1)  # Percentage noise


xp_0 = [0] * (nx[0] * ntheta[0])



# -------------------------- Conduct First initial experiments -----------------
x_0 = [0.2, 0.00, 0.00, 0.00, 0.00]

lbx = [0] * nx[0]  # [-0.25, -inf]
ubx = [inf] * nx[0]

lbxp = [-inf] * ntheta[0] * nx[0]
ubxp = [inf] * ntheta[0] * nx[0]

lbtheta = [-20.,0.] * 4#(ntheta[0])#[*[-20.] * (ntheta[0]-5), *[0] * 5]
ubtheta =  [40.] * (ntheta[0])#[*[20.] * (ntheta[0]-5), *[2] * 5]# [60] * ntheta[0]
lbu = [u_meas]  # [-1, -1]
ubu = [u_meas]  # [1, 1]
start = time.time()

x_init = np.zeros([N_exp, nx[0]])
for i in range(nx[0]-1):
    x_init[:N_exp, i] = x_meas[:N_exp, i, 0]
x_init[:N_exp, -1] = c2o * u_meas[:N_exp,2]/sum(u_meas[:N_exp,i] for i in range(1,nu[0]))

# ---------------------------------------------
# ----------- Set values for the inputs -----------
model = Reactor_pfr_model()
pe = ParameterEstimation(Reactor_pfr_model)
x_meas_pe = np.zeros([N_exp, nx[0],n_points+1])
for i in range(nx[0]):
    x_meas_pe[:,i,0]  = x_init[:N_exp,i]
    if pe.measured[i]:
        x_meas_pe[:,i,1:] = x_meas[:N_exp,i,1:]



pe.solve_pe(x_meas_pe, u_meas, dt)

#-------------------------------------------------#
#-------------------test--------------------------#
#-------------------------------------------------#

problem, w0, lbw, ubw, lbg, ubg, trajectories, mle_fn = \
        construct_NLP_collocation(N_exp, f, x_0, x_init, lbx, ubx, lbu, ubu, lbtheta,
                              ubtheta, dt,
                                   n_points, x_meas[:, :, 1:],
                                  0.8+np.random.rand(ntheta[0]),#
                                  8, 25) #-----Change x_meas1 to x_meas ------#

solver = nlpsol('solver', 'ipopt', problem, {"ipopt.tol": 1e-12})#, {"ipopt.hessian_approximation":"limited-memory"})#, {"ipopt.tol": 1e-10, "ipopt.print_level": 0})#, {"ipopt.hessian_approximation":"limited-memory"})

# Function to get x and u trajectories from w

sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
w_opt = sol['x'].full().flatten()




#
# problem, w0, lbw, ubw, lbg, ubg, trajectories, mle_fn = \
#         construct_NLP_RK4(N_exp, f, x_0, x_init, lbx, ubx, lbu, ubu, lbtheta,
#                               ubtheta, dt,
#                                    n_points, x_meas[:, :, 1:],
#                                   0.8+np.random.rand(ntheta[0]),#
#                                   8, 20) #-----Change x_meas1 to x_meas ------#
#
# solver = nlpsol('solver', 'ipopt', problem)#, {"ipopt.hessian_approximation":"limited-memory"})#, {"ipopt.tol": 1e-10, "ipopt.print_level": 0})#, {"ipopt.hessian_approximation":"limited-memory"})
#
# # Function to get x and u trajectories from w
#
# sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
w_opt1 = w_opt#sol['x'].full().flatten()
#bayopt_pe(f, lbtheta, ubtheta, nu, nx, x_meas[:,:,1:], n_points, ntheta[0], u_meas,N_exp, V, c1o, c2o, w_opt[:8])


obj_f = objective_cov(f, u_meas,x_meas[:, :, 1:], N_exp, nx, n_points, nu,  V, c1o, c2o, w_opt1[:ntheta[0]])


obj_ff = functools.partial(objective_cov,f, u_meas,x_meas[:, :, 1:], N_exp, nx, n_points, nu,  V, c1o, c2o)


vv1 = compute_Hessian(obj_ff,  w_opt1[:ntheta[0]].reshape((1,-1)))

#obj = objective_pe(f, u_meas, x_meas[:,:,1:], N_exp, nx, n_points, nu, w_opt[:8], V, c1o, c2o)

elapsed_time_fl = (time.time() - start)
print(elapsed_time_fl)


x_opt, u_opt, xp_opt, chi2 = trajectories(sol['x'])
print(chi2)
x_opt = x_opt.full()# to numpy array
u_opt = u_opt.full()# to numpy array#


f, _, _, _ = plant_model('sensitivity')
x_meas1 = np.zeros([N_exp+10, nx[0],n_points+1])

xp_meas = np.zeros((ntheta[0]*nx[0], N_exp * n_points))
pp = 0
s = 0
mle = 0
x_meas1 = np.zeros([N_exp, nx[0], n_points + 1])
xmin = np.zeros(nx[0] - 1)
xmax = np.zeros(nx[0] - 1)  # -1)
x_meas_norm = x_meas.copy()
for i in range(nx[0] - 1):
    xmax[i] = np.max(x_meas[:, i, 1:])
    if xmax[i] > 1e-9:
        x_meas_norm[:, i, :] = x_meas[:, i, 1:] / xmax[i]
    else:
        x_meas_norm[:, i, :] = x_meas[:, i, 1:]
        xmax[i] = 1.
sd = np.array([0.005,0.005,0.002,0.0002])
chi2 = np.zeros(17)
for k0 in range(N_exp):
        x11 = x_init[k0, :]# change it
        x_meas1[s, :, 0] = np.array(x11.T[:nx[0]])
        xp1 = np.zeros([nx[0]*ntheta[0], 1])
        for i in range(n_points):
            F = integrator_model(f, nu, nx, ntheta, 'embedded', 'sensitivity', dt[k0, i])
            Fk = F(x0=vertcat(x11, xp1), p=vertcat(u_meas[k0, :], w_opt[:8]))

            x11 = Fk['xf'][0:nx[0]]
            xp1 = Fk['xf'][nx[0]:]
            # + np.random.multivariate_normal([0.] * nx[0], np.diag(np. square(sigma))).T
            x_meas1[s, :, i+1] = np.array(x11.T)
            xp_meas[:, pp] = np.array(xp1.T)
            pp += 1
        s += 1
        chi2[k0] = np.sum((x_meas[k0,:,1]-x_meas1[k0,:-1,-1])**2/sd[:]**2)
        chisquare_test(chi2, 0.95, 13*nx[0]-ntheta[0])
vv1 = 0
rif = np.zeros(N_exp*(n_points))
for k in range(0,N_exp*(n_points),2):
    xp_r = reshape(xp_meas[:, k], (nx[0], ntheta[0]))
#    vv = np.zeros([ntheta[0], ntheta[0], N])
#    for i in range(0, N):
#    for i in range(ntheta[0]):
#        xp_r[:, i] = w_opt[i] * xp_r[:, i]
    rif[k] = np.trace(xp_r[:-1, :].T@ np.linalg.inv(np.diag(np.square(sigma[:]))) @xp_r[:-1,:])
    vv1 += (xp_r[:-1, :].T@ np.linalg.inv(np.diag(np.square(sigma[:]))) @xp_r[:-1,:])
rif = rif/rif.sum()
vv = np.linalg.inv(vv1)


k_exp = N_exp+1

print('info')
print(det(vv1))

# ---------------------------------------------------------------
t = np.zeros(ntheta[0])
theta1 = w_opt[:ntheta[0]]
# ------------------------- Calculate the statistics ------------

for i in range(ntheta[0]):
    t[i] = theta1[i] / (np.sqrt(vv[i, i])*stats.t.ppf((1-0.01/2), 4*n_points*nx[0]-ntheta[0]))

t_ref = stats.t.ppf((1-0.05), 4*n_points*nx[0]-ntheta[0])

histheta = []
histheta.append(theta1)
histt = []
histinf = []

histt.append(t)
histinf.append(det(vv1))


CI1 = []
CI1.append(np.sqrt(vv[0, 0])*stats.t.ppf((1-0.05/2), 2))
CI2 = []
CI2.append(np.sqrt(vv[1, 1])*stats.t.ppf((1-0.05/2), 2))
CI3 = []
CI3.append(np.sqrt(vv[2, 2])*stats.t.ppf((1-0.05/2), 2))
CI4 = []
CI4.append(np.sqrt(vv[3, 3])*stats.t.ppf((1-0.05/2), 2))
CI5 = []
CI5.append(np.sqrt(vv[4, 4])*stats.t.ppf((1-0.05/2), 2))
CI6 = []
CI6.append(np.sqrt(vv[5, 5])*stats.t.ppf((1-0.05/2), 2))
CI7 = []
CI7.append(np.sqrt(vv[6, 6])*stats.t.ppf((1-0.05/2), 2))
CI8 = []
CI8.append(np.sqrt(vv[7, 7])*stats.t.ppf((1-0.05/2), 2))

#pickle.dump([[CI1, CI2, CI3, CI4, CI5, CI6, CI7, CI8], hist, inf, histheta, x_meas, x_meas1, u_meas], open( 'save_results_before_exp.p','wb'))

CI0 = [CI1, CI2, CI3, CI4, CI5, CI6, CI7, CI8]

t_val0 = t.copy()


# ----------------------------- MBDOE ----------------------------------------- #
for iter in range(9):

# ------ Compute a number of experiments --------------
    bounds = np.array(
        pd.read_csv('/Users/' + PC + '/Dropbox/UCL/' + date + condition))[
        0]
    lbu = [bounds[-3], *bounds[0:len(bounds) - 3:2]]  #[60, 0.3, 0.3, 0.3]#, 0.1]
    ubu = [bounds[-2], *bounds[1:len(bounds) - 2:2]]#[130, 3, 3, 3]#, 2.0]

    f, nu, nx, ntheta = plant_model('sensitivity')
    if bayesopt == False:
        #-----Perform multi-starts----#
        minf = 0.1
        soll = []
        start = time.time()
        for i in range(3):
            problem, w0, lbw, ubw, lbg, ubg, trajectories = \
                construct_NLP_MBDoE_collocation(1, f, x_0, xp_0, lbx, ubx, lbu, ubu, theta1, theta1, dt,
                                            n_points, u_meas, vv1, sigma, 8, 15, c1o,c2o, V)
            solver = nlpsol('solver', 'ipopt', problem,
                        {"ipopt.print_level": 5})#, "ipopt.hessian_approximation": "limited-memory"})#, "ipopt.linear_solver":"ma27"})
    # Function to get x and u trajectories from w

            sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)

            if minf>= sol['f'].full().flatten():
                soll += sol
                w_opt = sol['x'].full().flatten()
                minf  =  sol['f'].full().flatten()
                x_opt, u_opt, xp_opt, Ts = trajectories(sol['x'])

        elapsed_time_fl = (time.time() - start)
        print(elapsed_time_fl)
        print(u_opt)
        x_opt = x_opt.full()
        ul = (np.array(ubu) + np.array(lbu)) / 2
        ur = (np.array(ubu) - np.array(lbu)) / 2
        for i in range(nu[0]):
            u_opt[i] = u_opt[i] * ur[i] + ul[i]
        u_opt = u_opt.full()

        objective(f, u_opt[:,0].reshape(1, nu[0]), vv1, 1, nx, n_points, nu, theta1, sigma, V, c1o, c2o)
    else:


        u_t, J_b = bayopt_design(f, lbu, ubu, nu, nx, vv1, n_points, theta1, sigma,V, c1o, c2o)
        u_opt = u_t[:,np.argmin(J_b)]
    # ------------ Store data ---------------------------------
# to numpy array#
#    dt = t_plot.full()[0]
    #  Fk = F(x0=[w_opt[0], 0], p=[w_opt[9], w_opt[10]], DT=200, theta=true_theta, xp0=np.zeros([4, 1]))
#    x11 = [x_opt[0, 0], 0., 0., 0., u_opt[1]]



    xp1 = np.zeros(nx[0]*ntheta[0])
    ###########
    u_meas[k_exp, :] = np.array([u_opt.T])
    ############
    xp1_his = np.zeros([nx[0]*ntheta[0], n_points])
# --------------------------------------------------------------

# ----------------- Conduct in-silico experiment ---------------
    if labbot==True:
        if 1 + k_exp>=10:
            file1 = '/Users/'+PC+'/Dropbox/UCL/'+date+'/Requests_'+str(1 + k_exp)+'.csv'
        else:
            file1 = '/Users/'+PC+'/Dropbox/UCL/'+date+'/Requests_0'+str(1 + k_exp)+'.csv'
                                                                      #'/Users/Panos/OneDrive - University College London' \
                #'/Share-actions-Leeds/Development/'+date+'/Requests_0'+str(1+k_exp)+'.csv'





#    time.sleep(60*15)
        for iu in range(u_opt.shape[0]):
                if u_opt[iu] >= ubu[iu]:
                    u_opt[iu] = ubu[iu]
        for iu in range(u_opt.shape[0]):
                if u_opt[iu] <= lbu[iu]:
                    u_opt[iu] = lbu[iu]

        df = pd.DataFrame({'P1': [u_opt[2]],
                    'P2': [u_opt[1]],
                    'P3': [u_opt[3]],
                    'T': [u_opt[0]]})
        df.to_csv(file1,index=False)
#############################
#        if k_exp+1>=10:
 #           file = '/Users/'+PC+'/Dropbox/UCL/' + date + '/Requests_' + str(1 + k_exp) + '.csv'
 #       else:
 #           file = '/Users/'+PC+'/Dropbox/UCL/' + date + '/Requests_0' + str(1+k_exp) + '.csv'


        ul = np.array(pd.read_csv(file1))

        u_meas[k_exp, 1] = ul[0][1]
        u_meas[k_exp, 2] = ul[0][0]
        u_meas[k_exp, 3] = ul[0][2]

        u_meas[k_exp, 0] = ul[0][-1]

        if k_exp+1>10:
            file = '/Users/'+PC+'/Dropbox/UCL/' + date + '/Requests_' + str(k_exp) + '.csv'
        else:
            file = '/Users/'+PC+'/Dropbox/UCL/' + date + '/Requests_0' + str(k_exp) + '.csv'


        ul = np.array(pd.read_csv(file))

        u_meas[k_exp-1, 1] = ul[0][1]
        u_meas[k_exp-1, 2] = ul[0][0]
        u_meas[k_exp-1, 3] = ul[0][2]

        u_meas[k_exp-1, 0] = ul[0][-1]

######################################################
        file1 = '/Users/'+PC+'/Dropbox/UCL/'+date+'/Peaks and Concentrations_'+str(k_exp)+'.csv' #I changed this

        time_to_wait = 60*3000
        time_counter = 0
        while not os.path.exists(file1):
            time.sleep(1)
            time_counter += 1
            if time_counter > time_to_wait:break

        size = np.shape(np.array(pd.read_csv(file1)))
        xl = np.zeros([N_exp, size[0] - 1, 1])  # size[1]])

        for i in range(N_exp):
            xl[i, :, :] = np.array(pd.read_csv(file1)['Concentration (mol/L)'])[1:].reshape(4, 1)
            for j in range(size[0] - 1):
                for k in range(1):
                    if xl[i, j, k] < 0:
                        xl[i, j, k] = 0.

        x_meas[k_exp-1, 0, n_points] = xl[0, 0]
        x_meas[k_exp-1, 1, n_points] = xl[0, 3]
        x_meas[k_exp-1, 2, n_points] = xl[0, 2]
        x_meas[k_exp-1, 3, n_points] = xl[0, 1]

        x_meas[k_exp-1, 0, 0] = c1o * u_meas[k_exp-1, 1]/sum(u_meas[k_exp-1, 1:])#u_opt[1]/sum(u_opt[1:])
        x_meas[k_exp-1, 1, 0] = 0.
        x_meas[k_exp-1, 2, 0] = 0.
        x_meas[k_exp-1, 3, 0] = 0.
 #   x_meas[1 + k_exp, 0, :] = xl[0, 0:n * n_points + 1:n, 1].T
 #   x_meas[1 + k_exp, 1:nx[0], :] = xl[0, 0:n * n_points + 1:n, 3:nx[0] + 3].T
 #   x_meas[1 + k_exp, -1, :] = xl[0, 0:n * n_points + 1:n, 2].T
        dt[k_exp-1, :] = V/sum(u_meas[k_exp-1, 1:])#sum(u_opt[1:])#xl[0, n:n * n_points + 1:n, 0].T - xl[0, 0: n * n_points :n, 0].T
        dt[k_exp, :] = V/sum(u_meas[k_exp, 1:])#sum(u_opt[1:])#xl[0, n:n * n_points + 1:n, 0].T - xl[0, 0: n * n_points :n, 0].T

    #x_meas[k_exp, :, 0] = [c1o * u_opt[1]/sum(u_opt[1:]), 0., 0., 0.]#, 2.4 * u_opt[2]/sum(u_opt[1:])]
    else:
        u_meas[k_exp,:] = u_opt.reshape(4)
        x_meas, dt = give_data_from_sim_update(k_exp, x_meas, u_opt, dt, true_theta,c1o, c2o, V)

    # -------------------------- to be deleted down ------------------- #
# -------------------------- to be deleted up ------------------- #

# ----------------
# --- I changed here the residence time (up) ---
# ----------------

    lbxp = [-inf] * ntheta[0] * nx[0]
    ubxp = [inf] * ntheta[0] * nx[0]

    lbu = [u_meas[:k_exp,:]]  # [-1, -1]
    ubu = [u_meas[:k_exp,:]]  # [1, 1]
# ------------------------------------------------------------------


# ------------------ update the number of experiments --------------
    k_exp += 1
# ------------------ Conduct new PE --------------------------------

    f, _, _, _ = plant_model([])
    x_init = np.zeros([k_exp-1, nx[0]])
    for i in range(nx[0]-1):
        x_init[:k_exp-1, i] = x_meas[:k_exp-1, i,0]
    x_init[:k_exp-1, -1] = c2o * u_meas[:k_exp-1,2]/np.sum(u_meas[:k_exp-1,i] for i in range(1,nu[0]))
    #objective_pe(f, u_meas, x_meas[:,:,1:], k_exp, nx, n_points, nu, true_theta, V, c1o, c2o)
    start = time.time()
    problem, w0, lbw, ubw, lbg, ubg, trajectories = \
    construct_NLP_collocation(k_exp-1, f, x_0, x_init[:k_exp-1,:], lbx, ubx, lbu, ubu, lbtheta,
                              ubtheta, dt[:k_exp-1, :],
                              n_points, x_meas[:k_exp-1, :, 1:],
                              theta1, 8, 15)
    solver = nlpsol('solver', 'ipopt', problem, {"ipopt.print_level": 5})

# Function to get x and u trajectories from w

    sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
    elapsed_time_fl = (time.time() - start)
    print(elapsed_time_fl)
    w_opt = sol['x'].full().flatten()

    x_opt, u_opt, xp_opt, ch2 = trajectories(sol['x'])
    print(ch2)
    x_opt = x_opt.full()  # to numpy array
    u_opt = u_opt.full()  # to numpy array#
# ----------------------------------------------------------------------
# ----------------- Compute the new sensitivities ----------------------
    RIF = np.zeros([k_exp * (n_points), ntheta[0]])


    theta1 = w_opt[:ntheta[0]]

#    u_meas[k_exp, :] = np.array([u_opt[0,-1], u_opt[1,-1], x11[0]])
    xp1_his = np.zeros([nx[0] * ntheta[0], n_points*(k_exp-1)])
    f, _, _, _ = plant_model('sensitivity')
    pp = 0
    for k0 in range(k_exp-1):

        x11 = x_init[k0, :]
        xp1 = np.zeros([nx[0] * ntheta[0], 1])
        for i in range(n_points):
            F = integrator_model(f, nu, nx, ntheta, 'embedded', 'sensitivity', dt[k0, i])
            Fk = F(x0=vertcat(x11, xp1), p=vertcat(u_meas[k0, :], theta1))

            x11 = Fk['xf'][0:nx[0]]
            xp1 = Fk['xf'][nx[0]:]
    # + np.random.multivariate_normal([0.] * nx[0], np.diag(np. square(sigma))).T
            xp1_his[:, pp] = np.array(xp1.T)
            pp += 1



    xp_opt = xp1_his#np.array(xp_opt.full())  # to numpy array

    RIF = np.zeros([(k_exp-1) * (n_points), ntheta[0]])
    vv1 = 0.
    for k in range((k_exp-1) * (n_points)):
        xp_r = reshape(xp_opt[:, k], (nx[0], ntheta[0]))
        #    vv = np.zeros([ntheta[0], ntheta[0], N])
        #    for i in range(0, N):
        print('info')
        vv1 += (xp_r[:-1,:].T@ np.linalg.inv(np.diag(np.square(sigma[:]))) @xp_r[:-1,:])
        for i in range(ntheta[0]):
            RIF[k, i] = np.linalg.norm((xp_r[:-1, :].T @ np.linalg.inv(np.diag(np.square(sigma[:])))
                                        @ xp_r[:-1, :])[i, i])
    vv = np.linalg.inv(vv1)
    RIF = RIF / np.linalg.norm(vv1)
    #plt.plot(RIF)
    print(det(vv1))
    #   V += np.linalg.inv(sum(xp_r[:, :, i].T @ np.array([[0.03, 0], [0, 0.01]]) @ xp_r[:, :, i] for i in range(0, N)))

    for ii in range(ntheta[0]):
        theta1[ii] = w_opt[ii]# * thetar[ii] + thetal[ii]
    t = np.zeros(ntheta[0])
# ----------------------------------------------------------------------------------------

# -------------------- Compute the statistics --------------------------------------------

    for i in range(ntheta[0]):
        t[i] = theta1[i] / (np.sqrt(vv[i, i]) * stats.t.ppf((1 - 0.05 / 2), (k_exp-1) * n_points * nx[0] - ntheta[0]))

    t_ref = stats.t.ppf((1 - 0.05), (k_exp-1) * n_points * nx[0] - ntheta[0])

    histt.append(t)
    histinf.append(det(vv1))

    histheta.append(theta1)

    CI1.append(np.sqrt(vv[0, 0])*stats.t.ppf((1-0.05/2), k_exp*n_points*nx[0]-ntheta[0]))
    CI2.append(np.sqrt(vv[1, 1])*stats.t.ppf((1-0.05/2), k_exp*n_points*nx[0]-ntheta[0]))
    CI3.append(np.sqrt(vv[2, 2])*stats.t.ppf((1-0.05/2), k_exp*n_points*nx[0]-ntheta[0]))
    CI4.append(np.sqrt(vv[3, 3])*stats.t.ppf((1-0.05/2), k_exp*n_points*nx[0]-ntheta[0]))
    CI5.append(np.sqrt(vv[4, 4])*stats.t.ppf((1-0.05/2), k_exp*n_points*nx[0]-ntheta[0]))
    CI6.append(np.sqrt(vv[5, 5])*stats.t.ppf((1-0.05/2), k_exp*n_points*nx[0]-ntheta[0]))
    CI7.append(np.sqrt(vv[6, 6])*stats.t.ppf((1-0.05/2), k_exp*n_points*nx[0]-ntheta[0]))
    CI8.append(np.sqrt(vv[7, 7])*stats.t.ppf((1-0.05/2), k_exp*n_points*nx[0]-ntheta[0]))

# ---------------------------------------- Iterate -------------------------------------

print('2')

