import pylab
import numpy as np
import scipy.integrate as scp
from pylab import *
import matplotlib.pyplot as plt
#secondary utilities
import csv
#import itertools
import os
import sys
import copy
import numpy.random as rnd
from casadi import *
from scipy.spatial.distance import cdist

import sobol_seq
from scipy.optimize import minimize




#######################################################
# --- Simulating GP (No uncertainty propagation)  --- #
#######################################################

class GP_simulate:

    ###########################
    # --- initializing GP --- #
    ###########################
    def __init__(self, GP, x0, controls):

        # variable inputs
        self.GP, self.x0 = GP, x0
        self.controls = controls  # shape (nu, steps)
        self.steps = controls.shape[1]

        # internal variable definition
        self.udim, self.ydim = controls.shape[0], x0.shape[0]
        self.xdim = controls.shape[0] + x0.shape[0]

    #####################################
    # --- one-step ahead prediction --- #
    #####################################

    def one_step(self, Xt):
        '''
        --- decription ---
        '''
        # internal variable definition
        GP = self.GP

        yt_mean, yt_std = GP.GP_inference_np(Xt)

        return yt_mean, yt_std

    ####################################################################
    # --- multi-step ahead prediction (No uncertainty propagation) --- #
    ####################################################################

    def multi_step_noprop(self):
        '''
        --- decription ---
        '''
        # internal variable definition
        one_step, controls = self.one_step, self.controls
        steps, x0, ydim = self.steps, self.x0, self.ydim

        # creating lists to store data (includes initial control, hence +1)
        Y_mean = np.zeros((steps + 1, ydim))
        Y_std = np.zeros((steps + 1, ydim))

        # starting process
        xnew = x0
        Y_mean[0, :] = xnew
        Y_std[0, :] = np.zeros((xnew.shape))

        # GP simulation
        for si in range(steps):
            x = np.hstack((xnew, controls[:, si]))
            ymean, ysigma = one_step(x)
            xnew = ymean

            Y_mean[si + 1, :] = ymean
            Y_std[si + 1, :] = ysigma

        return Y_mean, Y_std

    ######################################################################
    # --- multi-step ahead prediction (WITH uncertainty propagation) --- #
    ######################################################################

    def multi_step_withprop(self):
        '''
        --- decription ---
        '''
        # internal variable definition
        one_step, controls = self.one_step, self.controls
        steps, x0, ydim, xdim = self.steps, self.x0, self.ydim, self.xdim
        step_Uncert_prop = self.step_Uncert_prop
        n_controls, X_mean = controls.shape[0], self.GP.X_mean

        # creating lists to store data (includes initial control, hence +1)
        Y_mean = np.zeros((steps + 1, ydim))
        Y_std = np.zeros((steps + 1, ydim))

        # starting process
        xnew = x0
        Y_mean[0, :] = xnew
        Y_std[0, :] = np.zeros((xnew.shape))

        # GP simulation
        for si in range(steps):
          #  print('============== step ============', si)

            if si == 0:
                x = np.hstack((xnew, controls[:, si]))  # add control
                ynew, ysigma = one_step(x)  # compute next step
                ycov = np.identity(ydim) * ysigma  # make matrix (ydim x ydim)
                xcov = np.hstack((ycov,
                                  np.zeros((ydim, n_controls))))  # make matrix (ydim x xdim)
            else:
                xnew = ynew
                x = np.concatenate((xnew, controls[:, si]))  # add control

                xcov = np.vstack(
                    (xcov, np.zeros((n_controls, xdim))))  # add control uncertaint make matrix (xdim x xdim)

                xcov = xcov + np.identity((xcov.shape[0])) * X_mean * 1e-6  # add positive definite term
                ynew, ycov = step_Uncert_prop(x, xcov)  # step with uncertainty propagation
                ysigma = np.diag(ycov)  # get diagonal elements for plotting
                xcov = np.hstack((ycov,
                                  np.zeros((ydim, n_controls))))  # make matrix (ydim x xdim)

            Y_mean[si + 1, :] = ynew
            Y_std[si + 1, :] = ysigma

        return Y_mean, Y_std

    ###################################
    # --- Uncertainty propagation --- #
    ###################################

    def step_Uncert_prop(self, m, s):
        '''
        --- decription ---
        m = mean of input distribution
        s = standard deviation of input distribution
        '''
        #print('Uncertainty propagation only available for Gaussian kernel')
        # internal variable definition
        GP, xdim, ydim = self.GP, self.xdim, self.ydim
        hypopt, invKopt, n_point = GP.hypopt, GP.invKopt, GP.n_point
        stdX, stdY, meanX, meanY = GP.X_std, GP.Y_std, GP.X_mean, GP.Y_mean
        X_norm_train, Y_norm_train = GP.X_norm, GP.Y_norm

        # X = self.X

        # variable definition
        m = (m - meanX) / stdX  # normalize mean
        s = s / (stdX.reshape(xdim, 1) * stdX.reshape(1, xdim))  # normalize std
        mean = np.zeros(ydim)
        var = np.zeros([ydim, ydim])
        beta = np.zeros([n_point, ydim])
        k = np.zeros([n_point, ydim])
        z_a = np.zeros([xdim, n_point, n_point])
        kk = np.zeros([n_point, n_point])
        Q = np.zeros([n_point, n_point])

        # samples - mean value
        nu = X_norm_train - m.T  # (n_point, ydim)
        for a in range(ydim):
            sf2opt = np.exp(2 * hypopt[xdim, a])
            ellopt = np.exp(2 * hypopt[:xdim, a])
            Lambda = np.diag(ellopt)
            k[:, a] = np.diag(sf2opt * np.exp(-0.5 * nu @ pinv(Lambda) @ nu.T))

        # --- Loop over each output (GP) --- #
        for a in range(ydim):
            # --- parameters of each output --- #
            invK = invKopt[a]
            hyper = hypopt[:, a]
            ellopt, sf2opt = np.exp(2 * hyper[:xdim]), np.exp(2 * hyper[xdim])
            Lambda = np.diag(ellopt)  # diagonalize the vector

            # --- determine covariance of each output --- #
            # Begin computation of mean (same as original GP)
            beta[:, a] = invK @ Y_norm_train[:, a]
            aa = sf2opt / (np.sqrt(np.linalg.det(s @ pinv(Lambda) + np.eye(xdim))))
            q = aa * (np.diag(np.exp(-0.5 * nu @ pinv(s + Lambda) @ nu.T)))
            # output mean value
            mean[a] = beta[:, a].T @ q  # k.T*Kinv*y

            # --- begin computation of covariance --- #
            k[:, a] = np.log(sf2opt) + np.diag(-0.5 * nu @ pinv(Lambda) @ nu.T)
            z_a1 = pinv(Lambda) @ nu.T
            for b in range(a + 1):
                hyperb = hypopt[:, b]  # repeated
                elloptb, sf2optb = np.exp(2 * hyperb[:xdim]), np.exp(2 * hyperb[xdim])
                Lambdab = np.diag(elloptb)
                k[:, b] = np.log(sf2optb) + np.diag(-0.5 * nu @ pinv(Lambdab) @ nu.T)
                R = s @ (pinv(Lambda) + pinv(Lambdab)) + np.eye(xdim)
                z_a2 = pinv(Lambdab) @ nu.T
                kk = k[:, a].reshape([1, n_point]).T + k[:, b].reshape([1, n_point])

                U1, s1, V1 = (np.linalg.svd((pinv(R) @ s), full_matrices=True))
#                L_eig = np.diagflat(np.linalg.eig(pinv(R) @ s)[0])  # Make it better
#                X_eig = np.linalg.eig(pinv(R) @ s)[1]
                qq1 = U1 @ np.sqrt(np.diagflat(s1))
                qq2 = V1.T @ np.sqrt(np.diagflat(s1))
#                for i in range(n_point):
#                    for j in range(n_point):
#                        zij = pinv(Lambda) @ nu[i,:].T + pinv(Lambdab) @ nu[j,:].T
#                        Q[i,j] = (np.exp(kk +0.5 * zij.T @ pinv(R) @ s @ zij))/np.sqrt(np.linalg.det(R))
                Q = (np.exp(kk + 0.5 * cdist(z_a1.T @ qq1, -z_a2.T @ qq2,
                                             'euclidean', V=np.ones(xdim)) ** 2) / np.sqrt(np.linalg.det(R)))

                var[a, b] = beta[:, a].T @ Q @ beta[:, b] - mean[a] * mean[b]
                var[b, a] = var[a, b]
                if a == b:
                    var[a, a] = var[a, a] + sf2opt - np.trace(np.matmul(invK, Q))

        # --- compute un-normalized mean --- #
        # print('z_a2 = ',z_a2)
        # print('X_eig = ',X_eig)
        # print('L_eig = ',L_eig)

        mean_sample = mean * stdY + meanY
        var_sample = var * (stdY.reshape(ydim, 1) * stdY.reshape(1, ydim))
        #print('var_sample = ', var_sample)

        return mean_sample, var_sample

class GP_model:

    ###########################
    # --- initializing GP --- #
    ###########################
    def __init__(self, X, Y, kernel, multi_hyper, hyper):

        # GP variable definitions
        self.X, self.Y, self.kernel = X, Y, kernel
        self.n_point, self.nx_dim = X.shape[0], X.shape[1]
        self.ny_dim = Y.shape[1]
        self.multi_hyper = multi_hyper

        # normalize data
        self.X_mean, self.X_std = np.mean(X, axis=0), np.std(X, axis=0)
        self.Y_mean, self.Y_std = np.mean(Y, axis=0), np.std(Y, axis=0)
        self.X_norm, self.Y_norm = (X - self.X_mean) / self.X_std, (Y - self.Y_mean) / self.Y_std

        # determine hyperparameters
        self.hypopt, self.invKopt, self.K_n, self.K_opt, self.K_opt2 = self.determine_hyperparameters(hyper)

        #############################

    # --- Covariance Matrix --- #
    #############################

    def Cov_mat(self, kernel, X_norm, W, sf2):
        '''
        Calculates the covariance matrix of a dataset Xnorm
        --- decription ---
        '''

        if kernel == 'RBF':
            dist = cdist(X_norm, X_norm, 'seuclidean', V=W) ** 2
            cov_matrix = sf2 * np.exp(-0.5 * dist)
            return cov_matrix
            # Note: cdist =>  sqrt(sum(u_i-v_i)^2/V[x_i])
        else:
            print('ERROR no kernel with name ', kernel)

    def covSEard(self):
        nx_dim = self.nx_dim
        ell    = SX.sym('ell', nx_dim)
        sf2    = SX.sym('sf2')
        x, z   = SX.sym('x', nx_dim), SX.sym('z', nx_dim)
        dist   = sum1((x - z)**2 / ell)
        covSEfcn = Function('covSEfcn',[x,z,ell,sf2],[sf2*exp(-.5*dist)])

        return covSEfcn


    def GP_predictor(self, x):#, X,Y):
        nd, invKopt, hypopt      = self.ny_dim, self.invKopt, self.hypopt
        Ynorm, Xnorm             = SX(DM(self.Y_norm)), SX(DM(self.X_norm))
        ndat                     = Xnorm.shape[0]
        nX, covSEfcn         = self.nx_dim, self.covSEard()
        stdX, stdY, meanX, meanY = SX(self.X_std),SX(self.Y_std),SX(self.X_mean),SX(self.Y_mean)
        Kopt                     = self.K_opt2
#        nk     = 12
        x      = SX.sym('x',nX)
        #nk     = X.shape[0]
        xnorm  = (x - meanX)/stdX
        #Xnorm2 = (X - meanX)/stdX
        #Ynorm2 = (Y - meanY)/stdY

        k      = SX.zeros(ndat)
        #k2     = SX.zeros(ndat+nk)
        mean   = SX.zeros(nd)
        mean2  = SX.zeros(nd)
        var    = SX.zeros(nd)
        #Xnorm2 = SX.sym('Xnorm2',ndat+nk,nX)
        #invKY2 = SX.sym('invKY2',ndat+nk,nd)

        for i in range(nd):
            invK           = SX(DM(invKopt[i]))
            hyper          = SX(DM(hypopt[:,i]))
            ellopt, sf2opt = exp(2*hyper[:nX]), exp(2*hyper[nX])
            for j in range(ndat):
                k[j]  = covSEfcn(xnorm,Xnorm[j,:],ellopt,sf2opt)
            #for j in range(ndat+nk):
            #    k2[j] = covSEfcn(xnorm,Xnorm2[j,:],ellopt,sf2opt)

            invKYnorm = mtimes(invK,Ynorm[:,i])
            mean[i]   = mtimes(k.T,invKYnorm)
            #mean2[i]  = mtimes(k2.T,invKY2[:,i])
            var[i]    = sf2opt - mtimes(mtimes(k.T,invK),k)

        meanfcn  = Function('meanfcn',[x],[mean*stdY + meanY])
        #meanfcn2 = Function('meanfcn2',[x,Xnorm2,invKY2],[mean2*stdY + meanY])
        varfcn   = Function('varfcn',[x] ,[var*stdY**2])
        #varfcnsd = Function('varfcnsd',[x],[var])

        return meanfcn, varfcn#, meanfcn2, varfcnsd

    def GP_predictor1(self, x):  # , X,Y):
            nd, invKopt, hypopt = self.ny_dim, self.invKopt, self.hypopt
            Ynorm, Xnorm = SX(DM(self.Y_norm)), SX(DM(self.X_norm))
            ndat = Xnorm.shape[0]
            nX, covSEfcn = self.nx_dim, self.covSEard()
            stdX, stdY, meanX, meanY = SX(self.X_std), SX(self.Y_std), SX(self.X_mean), SX(self.Y_mean)
            Kopt = self.K_opt2
            #        nk     = 12
            #x = SX.sym('x', nX)
            # nk     = X.shape[0]
            xnorm = (x - meanX) / stdX
            # Xnorm2 = (X - meanX)/stdX
            # Ynorm2 = (Y - meanY)/stdY

            k = SX.zeros(ndat)
            # k2     = SX.zeros(ndat+nk)
            mean = SX.zeros(nd)
            mean2 = SX.zeros(nd)
            var = SX.zeros(nd)
            # Xnorm2 = SX.sym('Xnorm2',ndat+nk,nX)
            # invKY2 = SX.sym('invKY2',ndat+nk,nd)

            for i in range(nd):
                invK = SX(DM(invKopt[i]))
                hyper = SX(DM(hypopt[:, i]))
                ellopt, sf2opt = exp(2 * hyper[:nX]), exp(2 * hyper[nX])
                for j in range(ndat):
                    k[j] = covSEfcn(xnorm, Xnorm[j, :], ellopt, sf2opt)
                # for j in range(ndat+nk):
                #    k2[j] = covSEfcn(xnorm,Xnorm2[j,:],ellopt,sf2opt)

                invKYnorm = mtimes(invK, Ynorm[:, i])
                mean[i] = mtimes(k.T, invKYnorm)
                # mean2[i]  = mtimes(k2.T,invKY2[:,i])
                var[i] = sf2opt - mtimes(mtimes(k.T, invK), k)

            #meanfcn = Function('meanfcn', [x], [mean * stdY + meanY])
            # meanfcn2 = Function('meanfcn2',[x,Xnorm2,invKY2],[mean2*stdY + meanY])
            #varfcn = Function('varfcn', [x], [var * stdY ** 2])
            # varfcnsd = Function('varfcnsd',[x],[var])

            return [mean * stdY + meanY], [var * stdY ** 2]  # , meanfcn2, varfcnsd

    ################################
    # --- Covariance of sample --- #
    ################################

    def calc_cov_sample(self, xnorm, Xnorm, ell, sf2):
        '''
        Calculates the covariance of a single sample xnorm against the dataset Xnorm
        --- decription ---
        '''
        # internal parameters
        nx_dim = self.nx_dim

        dist = cdist(Xnorm, xnorm.reshape(1, nx_dim), 'seuclidean', V=ell) ** 2
        cov_matrix = sf2 * np.exp(-.5 * dist)

        return cov_matrix

        ###################################

    # --- negative log likelihood --- #
    ###################################

    def negative_loglikelihood(self, hyper, X, Y):
        '''
        --- decription ---
        '''
        # internal parameters
        n_point, nx_dim = self.n_point, self.nx_dim
        kernel = self.kernel

        W = np.exp(2 * hyper[:nx_dim])  # W <=> 1/lambda
        sf2 = np.exp(2 * hyper[nx_dim])  # variance of the signal
        sn2 = np.exp(2 * hyper[nx_dim + 1])  # variance of noise

        K = self.Cov_mat(kernel, X, W, sf2)  # (nxn) covariance matrix (noise free)
        K = K + (sn2 + 1e-8) * np.eye(n_point)  # (nxn) covariance matrix
        K = (K + K.T) * 0.5  # ensure K is simetric
        L = np.linalg.cholesky(K)  # do a cholesky decomposition
        logdetK = 2 * np.sum(
            np.log(np.diag(L)))  # calculate the log of the determinant of K the 2* is due to the fact that L^2 = K
        invLY = np.linalg.solve(L, Y)  # obtain L^{-1}*Y
        alpha = np.linalg.solve(L.T, invLY)  # obtain (L.T L)^{-1}*Y = K^{-1}*Y
        NLL = np.dot(Y.T, alpha) + logdetK  # construct the NLL

        return NLL

    ############################################################
    # --- Minimizing the NLL (hyperparameter optimization) --- #
    ############################################################

    def determine_hyperparameters(self, hyper):
        '''
        --- decription ---
        Notice we construct one GP for each output
        '''
        # internal parameters0
        X_norm, Y_norm = self.X_norm, self.Y_norm
        nx_dim, n_point = self.nx_dim, self.n_point
        kernel, ny_dim = self.kernel, self.ny_dim
        Cov_mat = self.Cov_mat

        lb = np.array([-3.] * (nx_dim + 1) + [-8.])  # lb on parameters (this is inside the exponential)
        ub = np.array([3.] * (nx_dim + 1) + [2.])  # lb on parameters (this is inside the exponential)
        bounds = np.hstack((lb.reshape(nx_dim + 2, 1),
                            ub.reshape(nx_dim + 2, 1)))
        multi_start = self.multi_hyper  # multistart on hyperparameter optimization
        if len(hyper) == 0:
            multi_startvec = sobol_seq.i4_sobol_generate(nx_dim + 2, multi_start)
        else:
            multi_startvec = hyper
        options = {'disp': False, 'maxiter': 10000}  # solver options
        hypopt = np.zeros((nx_dim + 2, ny_dim))  # hyperparams w's + sf2+ sn2 (one for each GP i.e. output var)
        localsol = [0.] * multi_start  # values for multistart
        localval = np.zeros((multi_start))  # variables for multistart

        invKopt = []
        Kopt2   = []

        # --- loop over outputs (GPs) --- #
        for i in range(ny_dim):
            # --- multistart loop --- #
            for j in range(multi_start):
                print('multi_start hyper parameter optimization iteration = ', j, '  input = ', i)
                hyp_init = lb + (ub - lb) * multi_startvec[j, :]
                # --- hyper-parameter optimization --- #
                res = minimize(self.negative_loglikelihood, hyp_init, args=(X_norm, Y_norm[:, i]) \
                               , method='SLSQP', options=options, bounds=bounds, tol=1e-12)
                localsol[j] = res.x
                localval[j] = res.fun

            # --- choosing best solution --- #
                minindex = np.argmin(localval)
                hypopt[:, i] = localsol[minindex]


            ellopt = np.exp(2. * hypopt[:nx_dim, i])
            sf2opt = np.exp(2. * hypopt[nx_dim, i])
            sn2opt = np.exp(2. * hypopt[nx_dim + 1, i]) + 1e-8

            # --- constructing optimal K --- #
            Kopt = Cov_mat(kernel, X_norm, ellopt, sf2opt) + sn2opt * np.eye(n_point)
            K_noh = Cov_mat(kernel, X_norm, ellopt, 1)
            # --- inverting K --- #
            invKopt += [np.linalg.solve(Kopt, np.eye(n_point))]
            Kopt2   += [Kopt]
        return hypopt, invKopt, Kopt2, K_noh, Kopt

    ########################
    # --- GP inference --- #
    ########################

    def GP_inference_np(self, x, var_out=True):
        '''
        --- decription ---
        '''
        nx_dim = self.nx_dim
        kernel, ny_dim = self.kernel, self.ny_dim
        hypopt, Cov_mat = self.hypopt, self.Cov_mat
        stdX, stdY, meanX, meanY = self.X_std, self.Y_std, self.X_mean, self.Y_mean
        calc_cov_sample = self.calc_cov_sample
        invKsample = self.invKopt
        Xsample, Ysample = self.X_norm, self.Y_norm
        # Sigma_w                = self.Sigma_w (if input noise)

        xnorm = (x - meanX) / stdX
        mean = np.zeros(ny_dim)
        var = np.zeros(ny_dim)
        # --- Loop over each output (GP) --- #
        for i in range(ny_dim):
            invK = invKsample[i]
            hyper = hypopt[:, i]
            ellopt, sf2opt = np.exp(2 * hyper[:nx_dim]), np.exp(2 * hyper[nx_dim])

            # --- determine covariance of each output --- #
            k = calc_cov_sample(xnorm, np.vstack((Xsample)), ellopt, sf2opt)


            mean[i] = np.matmul(np.matmul(k.T, invK), Ysample[:, i])
            var[i] = sf2opt - np.matmul(np.matmul(k.T, invK), k)
            # var[i] = sf2opt + Sigma_w[i,i]/stdY[i]**2 - np.matmul(np.matmul(k.T,invK),k) (if input noise)

        # --- compute un-normalized mean --- #
        mean_sample = mean * stdY + meanY
        var_sample = var * stdY ** 2

        if var_out:
            return mean_sample, var_sample
        else:
            return mean_sample

    def step_Uncert_prop(self, m, s):
        '''
          --- decription ---
          m = mean of input distribution
          s = standard deviation of input distribution
          '''

        # internal variable definition
        xdim, ydim = self.nx_dim, self.ny_dim
        hypopt, invKopt, n_point = self.hypopt, self.invKopt, self.n_point
        stdX, stdY, meanX, meanY = self.X_std, self.Y_std, self.X_mean, self.Y_mean
        X_norm_train, Y_norm_train = self.X_norm, self.Y_norm

        # X = self.X

        # variable definition
        m = (m - meanX) / stdX  # normalize mean
        s = s / (stdX.reshape(xdim, 1) * stdX.reshape(1, xdim))  # normalize std
        mean = np.zeros(ydim)
        var = np.zeros([ydim, ydim])
        beta = np.zeros([n_point, ydim])
        k = np.zeros([n_point, ydim])
        z_a = np.zeros([xdim, n_point, n_point])
        kk = np.zeros([n_point, n_point])
        Q = np.zeros([n_point, n_point])

        # samples - mean value
        nu = X_norm_train - m.T  # (n_point, ydim)
        for a in range(ydim):
            sf2opt = np.exp(2 * hypopt[xdim, a])
            ellopt = np.exp(2 * hypopt[:xdim, a])
            Lambda = np.diag(ellopt)
            k[:, a] = np.diag(sf2opt * np.exp(-0.5 * nu @ pinv(Lambda) @ nu.T))

        # --- Loop over each output (GP) --- #
        for a in range(ydim):
            # --- parameters of each output --- #
            invK = invKopt[a]
            hyper = hypopt[:, a]
            ellopt, sf2opt = np.exp(2 * hyper[:xdim]), np.exp(2 * hyper[xdim])
            Lambda = np.diag(ellopt)  # diagonalize the vector

            # --- determine covariance of each output --- #
            # Begin computation of mean (same as original GP)
            beta[:, a] = invK @ Y_norm_train[:, a]
            aa = sf2opt / (np.sqrt(np.linalg.det(s @ pinv(Lambda) + np.eye(xdim))))
            q = aa * (np.diag(np.exp(-0.5 * nu @ pinv(s + Lambda) @ nu.T)))
            # output mean value
            mean[a] = beta[:, a].T @ q  # k.T*Kinv*y

            # --- begin computation of covariance --- #
            k[:, a] = np.log(sf2opt) + np.diag(-0.5 * nu @ pinv(Lambda) @ nu.T)
            z_a1 = pinv(Lambda) @ nu.T
            for b in range(a + 1):
                hyperb = hypopt[:, b]  # repeated
                elloptb, sf2optb = np.exp(2 * hyperb[:xdim]), np.exp(2 * hyperb[xdim])
                Lambdab = np.diag(elloptb)
                k[:, b] = np.log(sf2optb) + np.diag(-0.5 * nu @ pinv(Lambdab) @ nu.T)
                R = s @ (pinv(Lambda) + pinv(Lambdab)) + np.eye(xdim)
                z_a2 = pinv(Lambdab) @ nu.T
                kk = k[:, a].reshape([1, n_point]).T + k[:, b].reshape([1, n_point])

                U1, s1, V1 = (np.linalg.svd((pinv(R) @ s), full_matrices=True))
                #                L_eig = np.diagflat(np.linalg.eig(pinv(R) @ s)[0])  # Make it better
                #                X_eig = np.linalg.eig(pinv(R) @ s)[1]
                qq1 = U1 @ np.sqrt(np.diagflat(s1))
                qq2 = V1.T @ np.sqrt(np.diagflat(s1))
                #                for i in range(n_point):
                #                    for j in range(n_point):
                #                        zij = pinv(Lambda) @ nu[i,:].T + pinv(Lambdab) @ nu[j,:].T
                #                        Q[i,j] = (np.exp(kk +0.5 * zij.T @ pinv(R) @ s @ zij))/np.sqrt(np.linalg.det(R))
                Q = (np.exp(kk + 0.5 * cdist(z_a1.T @ qq1, -z_a2.T @ qq2,
                                             'euclidean', V=np.ones(xdim)) ** 2) / np.sqrt(np.linalg.det(R)))

                var[a, b] = beta[:, a].T @ Q @ beta[:, b] - mean[a] * mean[b]
                var[b, a] = var[a, b]
                if a == b:
                    var[a, a] = var[a, a] + sf2opt - np.trace(np.matmul(invK, Q))

        # --- compute un-normalized mean --- #
        # print('z_a2 = ',z_a2)
        # print('X_eig = ',X_eig)
        # print('L_eig = ',L_eig)

        mean_sample = mean * stdY + meanY
        var_sample = var * (stdY.reshape(ydim, 1) * stdY.reshape(1, ydim))
        #print('var_sample = ', var_sample)


        return mean_sample, var_sample

    def gp_exact_moment(self, invK, X, Y, hyper, inputmean, inputcov):

        maha = self.maha
        xdim, ydim = self.nx_dim, self.ny_dim
        hyper, invK, n_point = self.hypopt.T, self.invKopt, self.n_point
        stdX, stdY, meanX, meanY = self.X_std, self.Y_std, self.X_mean, self.Y_mean
        X, Y = self.X_norm, self.Y_norm
        inputmean = np.array(inputmean).reshape((-1, 1)).T
        Ny = ydim#len(invKopt)
        N, Nx = n_point, xdim
        mean = MX.zeros(Ny, 1)
        beta = MX.zeros(N, Ny)
        log_k = MX.zeros(N, Ny)

        inputmean = (inputmean - meanX) / stdX.reshape((1, xdim))  # normalize mean
        inputcov = inputcov / (stdX.reshape((xdim, 1)) * stdX.reshape((1, xdim))) # normalize std

        v = X - repmat(inputmean, N, 1)

        covariance = MX.zeros(Ny, Ny)

        # TODO: Fix that LinsolQr don't work with the extended graph?
        A = SX.sym('A', inputcov.shape)
        [Q, R2] = qr(A)
        determinant = Function('determinant', [A], [ exp( trace( log(R2)))])

        for a in range(Ny):
            beta[:, a] = invK[a]@ Y[:, a]
            iLambda = diag( exp(-2 * hyper[a, :Nx]))
            R = inputcov + diag(exp(2 * hyper[a, :Nx]))
            iR = mtimes(iLambda, (MX.eye(Nx) - solve(( MX.eye(Nx)
                                                       +  mtimes(inputcov, iLambda)),
                                                     ( mtimes(inputcov, iLambda)))))
#            T =  mtimes(v, iR)
#            c =  exp(2 * hyper[a, Nx]) /  sqrt(determinant(R+ np.eye(xdim))) \
#                *  exp( sum2(hyper[a, :Nx]))
#            q2 = c *  exp(- sum2(T * v) * 0.5)
#            qb = q2 * beta[:, a]
#            mean[a] =  sum1(qb)

            #beta[:, a] = invK[a] @ Y[:, a]
            aa = np.exp(2 * hyper[a, Nx]) / (np.sqrt(determinant(inputcov @ (iLambda) + np.eye(xdim))))
            q = aa * (np.diag(np.exp(-0.5 * v @ pinv(inputcov + pinv(iLambda)) @ v.T)))

            mean[a] = (invK[a] @ Y[:, a]).reshape((-1,1)).T@q#beta[:, a].T @ q

            #t =  repmat( exp(hyper[a, :Nx]), N, 1)
            #v1 = v / t
            #log_k[:, a] = 2 * hyper[a, Nx] -  sum2(v1 * v1) * 0.5
            log_k[:, a] = 2 * hyper[a, Nx] + np.diag(-0.5 * v @ iLambda @ v.T)
            z_a1 = iLambda @ v.T
        # covariance with noisy input



            for b in range(a + 1):
                hyperb = hyper.T[:, b]  # repeated
                elloptb, sf2optb = np.exp(2 * hyperb[:xdim]), np.exp(2 * hyperb[xdim])
                Lambdab = np.diag(elloptb)
                log_k[:, b] = np.log(sf2optb) + np.diag(-0.5 * v @ pinv(Lambdab) @ v.T)
                R = inputcov @ (iLambda + pinv(Lambdab)) + np.eye(xdim)
                z_a2 = pinv(Lambdab) @ v.T
                #kk = log_k[:, a].reshape((1, n_point)).T + log_k[:, b].reshape((1, N))

            #U1, s1, V1 = (np.linalg.svd((pinv(R) @ inputcov), full_matrices=True))
            #                L_eig = np.diagflat(np.linalg.eig(pinv(R) @ s)[0])  # Make it better
            #                X_eig = np.linalg.eig(pinv(R) @ s)[1]
            #qq1 = U1 @ np.sqrt(np.diagflat(s1))
            #qq2 = V1.T @ np.sqrt(np.diagflat(s1))
            #                for i in range(n_point):
            #                    for j in range(n_point):
            #                        zij = pinv(Lambda) @ nu[i,:].T + pinv(Lambdab) @ nu[j,:].T
            #                        Q[i,j] = (np.exp(kk +0.5 * zij.T @ pinv(R) @ s @ zij))/np.sqrt(np.linalg.det(R))
            #Q = (np.exp(kk + 0.5 * cdist(z_a1.T @ qq1, -z_a2.T @ qq2,
            #                             'euclidean', V=np.ones(xdim)) ** 2) / np.sqrt(np.linalg.det(R)))
                Q = exp(repmat(log_k[:, a], 1, N)
                     + repmat(transpose(log_k[:, b]), N, 1)
                     + maha(z_a1.T, -z_a2.T, solve(R, inputcov * 0.5), N))
                covariance[a, b] = beta[:, a].T @ Q @ beta[:, b] - mean[a] * mean[b]
                covariance[b, a] = covariance[a, b]
                if a == b:
                    covariance[a, a] = covariance[a, a] + exp(2 * hyperb[xdim]) - trace(invK[a]@ Q)

        mean_sample = mean * stdY + meanY
        var_sample = covariance * (stdY.reshape(ydim, 1) * stdY.reshape(1, ydim))
        return [mean_sample, var_sample]

    def maha(self, a1, b1, Q1, N):
        """Calculate the Mahalanobis distance
        Copyright (c) 2018, Eric Bradford
        """
        aQ =  mtimes(a1, Q1)
        bQ =  mtimes(b1, Q1)
        K1 =  repmat( sum2(aQ * a1), 1, N) \
             +  repmat( transpose( sum2(bQ * b1)), N, 1) \
             - 2 *  mtimes(aQ,  transpose(b1))
        return K1

    def gp_taylor_approx(self, inputmean, inputcovar,
                         meanFunc='zero', diag=False, log=False):
        maha = self.maha
        xdim, ydim = self.nx_dim, self.ny_dim
        hyper, invK, n_point = self.hypopt.T, self.invKopt, self.n_point
        stdX, stdY, meanX, meanY = self.X_std, self.Y_std, self.X_mean, self.Y_mean
        X, Y = self.X_norm, self.Y_norm
        covSE = self.covSEard()
        inputmean = np.array(inputmean).reshape((-1, 1)).T
        Ny = ydim#len(invKopt)
        N, Nx = n_point, xdim
        mean = MX.zeros(Ny, 1)
        beta = MX.zeros(N, Ny)
        log_k = MX.zeros(N, Ny)

        inputmean = (inputmean - meanX) / stdX.reshape((1, xdim))  # normalize mean
        inputcovar = inputcovar / (stdX.reshape((xdim, 1)) * stdX.reshape((1, xdim))) # normalize std

        if log:
            X = log(X)
            Y = log(Y)
            inputmean = log(inputmean)

        Ny = len(invK)
        N, Nx = MX.size(X)
        mean = MX.zeros(Ny, 1)
        var = MX.zeros(Nx, 1)
        v = X - repmat(inputmean, N, 1)
        covar_temp = MX.zeros(Ny, Ny)

        covariance = MX.zeros(Ny, Ny)
        d_mean = MX.zeros(Ny, 1)
        dd_var = MX.zeros(Ny, Ny)

        # Casadi symbols
        x_s = SX.sym('x', Nx)
        z_s = SX.sym('z', Nx)
        ell_s = SX.sym('ell', Nx)
        sf2_s = SX.sym('sf2')
        #covSE = Function('covSE', [x_s, z_s, ell_s, sf2_s],
        #                    [covSEard(x_s, z_s, ell_s, sf2_s)])

        for a in range(Ny):
            ell = hyper[a, :Nx]
            w = 1 / ell ** 2
            sf2 = MX(hyper[a, Nx] ** 2)
            m = get_mean_function(hyper[a, :], inputmean, func=meanFunc)
            iK = MX(invK[a])
            alpha = mtimes(iK, Y[:, a] - m(inputmean)) + m(inputmean)
            kss = sf2

            ks = MX.zeros(N, 1)
            for i in range(N):
                ks[i] = covSE(X[i, :], inputmean, ell, sf2)

            invKks = mtimes(iK, ks)
            mean[a] = mtimes(ks.T, alpha)
            var[a] = kss - mtimes(ks.T, invKks)
            d_mean[a] = mtimes(transpose(w[a] * v[:, a] * ks), alpha)

            # BUG: This don't take into account the covariance between states
            for d in range(Ny):
                for e in range(Ny):
                    dd_var1a = mtimes(transpose(v[:, d] * ks), iK)
                    dd_var1b = mtimes(dd_var1a, v[e] * ks)
                    dd_var2 = mtimes(transpose(v[d] * v[e] * ks), invKks)
                    dd_var[d, e] = -2 * w[d] * w[e] * (dd_var1b + dd_var2)
                    if d == e:
                        dd_var[d, e] = dd_var[d, e] + 2 * w[d] * (kss - var[d])

            mean_mat = mtimes(d_mean, d_mean.T)
            covar_temp[0, 0] = inputcovar[a, a]
            covariance[a, a] = var[a] + trace(mtimes(covar_temp, .5
                                                           * dd_var + mean_mat))

        return [mean, covariance]

    def derivatives_gp(self):
        xdim, ydim = self.nx_dim, self.ny_dim
        hyper, invK, n_point = self.hypopt.T, self.invKopt, self.n_point
        stdX, stdY, meanX, meanY = self.X_std, self.Y_std, self.X_mean, self.Y_mean
        X, Y = self.X_norm, self.Y_norm
        gp1   = self.GP_predictor1
        x_s = SX.sym('x', xdim)
        hmu = []
        hvf = []
        mu  = Function('mu', [x_s], (gp1(x_s)[0]), ['x'], ['mu'])
        vf  = Function('v', [x_s], (gp1(x_s)[1]), ['x'], ['v'])
        #Jmu = Function('Jmu', [x_s], [jacobian((gp1(x_s)[0][0]), x_s)], ['x'], ['Jmu'])
        #Jvf = Function('Jvf', [x_s], [jacobian((gp1(x_s)[1][0]), x_s)], ['x'], ['Jvf'])

        #for i in range(ydim):
        #    hmu += [Function('hmu', [x_s], [jacobian(jacobian(gp1(x_s)[0][0][i], x_s), x_s)],
        #                     ['x'], ['hmu'])]#[jacobian(jacobian((gp1(x_s)[0][0]), x_s).T, x_s)]
        #    hvf += [Function('hvf', [x_s], [jacobian(jacobian(gp1(x_s)[1][0][i], x_s), x_s)],
        #                     ['x'], ['hvf'])]#[jacobian(jacobian((gp1(x_s)[1][0]), x_s).T, x_s)]

        return mu, vf#, Jmu, Jvf, hmu, hvf