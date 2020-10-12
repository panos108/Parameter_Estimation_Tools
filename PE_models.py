from casadi import *
import numpy as np

class Reactor_pfr_model:
    def __init__(self, A=None, bounds_theta=None, normalize=None):
        assert A==None, "Houston we've got a problem, the A is not given."
        A = np.array([[-1,1,0,0,-1],
                      [-1,0,1,0,-1],
                      [0,-1,0,1,-1],
                      [0,0,-1,1,-1]]).T
        self.A = A
        self.nx = A.shape[0]
        self.ntheta = A.shape[1]*2
        self.nu = 4
        self.measured = [True,True,True,True,False]
        self.lower_bound        = np.zeros(self.nx)
        self.upper_bound        = np.zeros(self.nx)+inf
        self.standard_deviation = np.array([1. , 1. , 1., 1., 1.])
        if normalize==None:
            self.normalize_data     = self.standard_deviation/max(self.standard_deviation)
        if bounds_theta == None:
            self.lbtheta       = [-20., 0.] * 4
            self.ubtheta       = [40.] * (self.ntheta)
            self.bounds_theta  = [self.lbtheta, self.ubtheta]
        else:
            self.bounds_theta = bounds_theta

    def plant_model_real(self, transformation):
        """
        Define the model that is meant to describe the physical system
        :return: model f
        """
        A  = self.A
        x = MX.sym('x', self.nx)
        u = MX.sym('u', self.nu)
        theta = MX.sym('theta', self.ntheta)

        self.constraints_x  = []
        self.constraints_x += [self.lower_bound]
        self.constraints_x += [self.upper_bound]

        if transformation:
            k  = exp( theta[0::2]- theta[1::2] *1e4 / 8.314 * (1 / (u[0] + 273.15) - 1 / (90 + 273)))
        else:
            k = theta[0::2] * exp(- theta[1::2] / 8.314 * (1 / (u[0] + 273) - 1 / (90 + 273)))

        r = []  # MX.sym('r', (A.shape[1],1))
        for i in range(A.shape[1]):
            r1 = x[0] ** np.heaviside(-A[0, i], 0) * k[i]
            for j in range(1, A.shape[0]):
                if A[j, i] != 0:
                    r1 = x[j] ** np.heaviside(-A[j, i], 0) * r1
            r = vertcat(r, r1)
        xdot = A @ r  # +\

        # Quadrature
        L = []
        # Algebraic
        alg = []
        return xdot, L, alg, x, u, theta

    def Generate_fun_integrator(self, sensitivity=False, transformation=True):
        # Calculate on the fly dynamic sensitivities without the need of perturbations
        xdot, L, alg, x, u, theta = self.plant_model_real(transformation)
        # x = MX.sym('x', self.nx)
        # u = MX.sym('u', self.nu)
        # theta = MX.sym('theta', self.ntheta)

        if sensitivity:
            x_p = MX.sym('xp', np.shape(x)[0] * np.shape(theta)[0])
            xpdot = []
            for i in range(np.shape(theta)[0]):
                xpdot = vertcat(xpdot, jacobian(xdot, x) @ (x_p[self.nx * i: self.nx * i + self.nx])
                                + jacobian(xdot, theta)[self.nx * i: self.nx * i + self.nx])
                f = Function('f', [x, u, theta, x_p], [xdot, L, xpdot],
                             ['x', 'u', 'theta', 'xp'], ['xdot', 'L', 'xpdot'])
        else:
            f = Function('f', [x, u, theta], [xdot, L], ['x', 'u', 'theta'], ['xdot', 'L'])

        nu = np.shape(u)[0]
        nx =  np.shape(x)[0]
        ntheta =  np.shape(theta)[0]

        return f, nu, nx, ntheta



class Reactor_pfr_model_Sonogashira:
    def __init__(self, A=None, bounds_theta=None, normalize=[None]):
        assert A==None, "Houston we've got a problem, the A is not given."
        A = np.array([[-1,1,0,-1],
                      [0,-1,1,-1]]).T
        self.A = A
        self.nx = A.shape[0]
        self.ntheta = A.shape[1]*2
        self.nu = 4
        self.measured = [True,True,True,False]
        self.lower_bound        = np.zeros(self.nx)
        self.upper_bound        = np.zeros(self.nx)+inf
        self.standard_deviation = np.array([1. , 1. , 1., 1.])#*np.sqrt(2.9105444796447138e-05)
        if normalize[0]==None:
            self.normalize_data     = self.standard_deviation/max(self.standard_deviation)
        else:
            self.normalize_data     = normalize

        if bounds_theta == None:
            self.lbtheta       = [-20., 0.] * 2
            self.ubtheta       = [40.] * (self.ntheta)
            self.bounds_theta  = [self.lbtheta, self.ubtheta]
        else:
            self.bounds_theta = bounds_theta

    def plant_model_real(self, transformation):
        """
        Define the model that is meant to describe the physical system
        :return: model f
        """
        A  = self.A
        x = MX.sym('x', self.nx)
        u = MX.sym('u', self.nu)
        theta = MX.sym('theta', self.ntheta)

        self.constraints_x  = []
        self.constraints_x += [self.lower_bound]
        self.constraints_x += [self.upper_bound]

        if transformation:
            k  = exp( theta[0::2]- theta[1::2] *1e4 / 8.314 * (1 / (u[0] + 273.15) - 1 / (90 + 273)))
            self.tranformed=True
        else:
            k = theta[0::2] * exp(- theta[1::2] / 8.314 * (1 / (u[0] + 273) - 1 / (90 + 273)))


        r = []  # MX.sym('r', (A.shape[1],1))
        for i in range(A.shape[1]):
            r1 = x[0] ** np.heaviside(-A[0, i], 0) * k[i]
            for j in range(1, A.shape[0]):
                if A[j, i] != 0:
                    r1 = x[j] ** np.heaviside(-A[j, i], 0) * r1
            r = vertcat(r, r1)
        xdot = A @ r  # +\

        # Quadrature
        L = []
        # Algebraic
        alg = []
        return xdot, L, alg, x, u, theta

    def Generate_fun_integrator(self, sensitivity=False, transformation=True):
        # Calculate on the fly dynamic sensitivities without the need of perturbations
        xdot, L, alg, x, u, theta = self.plant_model_real(transformation)
        # x = MX.sym('x', self.nx)
        # u = MX.sym('u', self.nu)
        # theta = MX.sym('theta', self.ntheta)

        if sensitivity:
            x_p = MX.sym('xp', np.shape(x)[0] * np.shape(theta)[0])
            xpdot = []
            for i in range(np.shape(theta)[0]):
                xpdot = vertcat(xpdot, jacobian(xdot, x) @ (x_p[self.nx * i: self.nx * i + self.nx])
                                + jacobian(xdot, theta)[self.nx * i: self.nx * i + self.nx])
                f = Function('f', [x, u, theta, x_p], [xdot, L, xpdot],
                             ['x', 'u', 'theta', 'xp'], ['xdot', 'L', 'xpdot'])
        else:
            f = Function('f', [x, u, theta], [xdot, L], ['x', 'u', 'theta'], ['xdot', 'L'])

        nu = np.shape(u)[0]
        nx =  np.shape(x)[0]
        ntheta =  np.shape(theta)[0]

        return f, nu, nx, ntheta

    def transformations_parameters(self, theta):
        theta_nom = theta.copy()
        theta_nom[0::2] = np.exp(theta[0::2])
        theta_nom[1::2] = theta[1::2]*1e4
        self.theta_nominal = theta_nom
        return theta_nom


class Reactor_pfr_model_Sonogashira_general_order:
    def __init__(self, A=None, bounds_theta=None, normalize=[None], powers=[1,1,1,1]):
        assert A==None, "Houston we've got a problem, the A is not given."
        A = np.array([[-1,1,0,-1],
                      [0,-1,1,-1]]).T
        self.A = A
        self.nx = A.shape[0]
        self.ntheta = A.shape[1]*2
        self.nu = 4
        self.measured = [True,True,True,False]
        self.lower_bound        = np.zeros(self.nx)
        self.upper_bound        = np.zeros(self.nx)+inf
        self.standard_deviation = np.array([1. , 1. , 1., 1.])*np.sqrt(2.9105444796447138e-04)
        self.powers             = powers
        if normalize[0]==None:
            self.normalize_data     = self.standard_deviation/max(self.standard_deviation)
        else:
            self.normalize_data     = normalize

        if bounds_theta == None:
            self.lbtheta       = [-20., 0.] * 2# + [0]*4
            self.ubtheta       = [40.] * 4# + [5]*4#(self.ntheta)
            self.bounds_theta  = [self.lbtheta, self.ubtheta]
        else:
            self.bounds_theta = bounds_theta

    def plant_model_real(self, transformation):
        """
        Define the model that is meant to describe the physical system
        :return: model f
        """
        A  = self.A
        x = MX.sym('x', self.nx)
        u = MX.sym('u', self.nu)
        theta = MX.sym('theta', self.ntheta)

        self.constraints_x  = []
        self.constraints_x += [self.lower_bound]
        self.constraints_x += [self.upper_bound]

        if transformation:
            k  = exp( theta[0:4:2]- theta[1:4:2] *1e4 / 8.314 * (1 / (u[0] + 273.15) - 1 / (90 + 273)))
            self.tranformed=True
        else:
            k  = exp( theta[0:4:2]- theta[1:4:2] *1e4 / 8.314 * (1 / (u[0] + 273.15) - 1 / (90 + 273)))

        # power1 = theta[4:]
        r = []  # MX.sym('r', (A.shape[1],1))
        kk = 0
        for i in range(A.shape[1]):
            r1 = (x[0]) ** (np.heaviside(-A[0, i], 0)*self.powers[kk]) * k[i]
            for j in range(1, A.shape[0]):
                if A[j, i] != 0:
                    if np.heaviside(-A[j, i], 0) == 1:
                        kk += 1
                    r1 = (x[j]) ** (np.heaviside(-A[j, i], 0)*self.powers[kk]) * r1

            r = vertcat(r, r1)
            if np.heaviside(-A[0, i], 0)==1 and i>0:
                kk+=1
        xdot = A @ r  # +\

        # Quadrature
        L = []
        # Algebraic
        alg = []
        return xdot, L, alg, x, u, theta

    def Generate_fun_integrator(self, sensitivity=False, transformation=True):
        # Calculate on the fly dynamic sensitivities without the need of perturbations
        xdot, L, alg, x, u, theta = self.plant_model_real(transformation)
        # x = MX.sym('x', self.nx)
        # u = MX.sym('u', self.nu)
        # theta = MX.sym('theta', self.ntheta)

        if sensitivity:
            x_p = MX.sym('xp', np.shape(x)[0] * np.shape(theta)[0])
            xpdot = []
            for i in range(np.shape(theta)[0]):
                xpdot = vertcat(xpdot, jacobian(xdot, x) @ (x_p[self.nx * i: self.nx * i + self.nx])
                                + jacobian(xdot, theta)[self.nx * i: self.nx * i + self.nx])
                f = Function('f', [x, u, theta, x_p], [xdot, L, xpdot],
                             ['x', 'u', 'theta', 'xp'], ['xdot', 'L', 'xpdot'])
        else:
            f = Function('f', [x, u, theta], [xdot, L], ['x', 'u', 'theta'], ['xdot', 'L'])

        nu = np.shape(u)[0]
        nx =  np.shape(x)[0]
        ntheta =  np.shape(theta)[0]

        return f, nu, nx, ntheta

    # def transformations_parameters(self, theta):
    #     theta_nom = theta.copy()
    #     theta_nom[0:4:2] = np.exp(theta[0::2])
    #     theta_nom[1:4:2] = theta[1:4:2]*1e4
    #     self.theta_nominal = theta_nom
    #     return theta_nom
    #
    def integration(self,dt, sensitivity=False, transformation=True):
            X = SX.sym('X0', self.nx)
            U = SX.sym('U', self.nu)
            theta = SX.sym('theta', self.ntheta)
            xp0 = SX.sym('xp', np.shape(X)[0] * np.shape(theta)[0])
            f, nu, nx, ntheta = self.Generate_fun_integrator(sensitivity, transformation)
            if sensitivity:
                xdot, qj, xpdot = f(X, U, theta, xp0)
                dae = {'x': vertcat(X, xp0), 'p': vertcat(U, theta), 'ode': vertcat(xdot, xpdot)}
            else:
                xdot, qj = f(X, U, theta)
                dae = {'x': vertcat(X), 'p': vertcat(U, theta), 'ode': vertcat(xdot)}
            opts = {'tf': dt}  # interval length
            F_int = integrator('F', 'cvodes', dae, opts)
            return  F_int

    def perform_evaluation(self, x, u, theta, dt, sensitivity=False, transformation=True):
            xp1 = []
            F_int = self.integration(dt, sensitivity, transformation)
            if sensitivity:
                Fk            = F_int(x0=vertcat(x, xp1), p=vertcat(u, theta))
                x11 = Fk['xf'][0:self.nx]
                xp1 = Fk['xf'][self.nx:]
                return x11, xp1
            else:
                Fk            = F_int(x0=vertcat(x, xp1), p=vertcat(u, theta))
                x11 = Fk['xf'][0:self.nx]
                return x11



class Reactor_pfr_model_Sonogashira_hybrid:
        def __init__(self, A=None, bounds_theta=None, normalize=[None]):
            assert A == None, "Houston we've got a problem, the A is not given."
            A = np.array([[-1, 1, 0, -1],
                          [0, -1, 1, -1]]).T
            self.A = A
            self.nx = A.shape[0]
            self.ntheta = A.shape[1] * 2+4
            self.nu = 4
            self.measured = [True, True, True, False]
            self.lower_bound = np.zeros(self.nx)
            self.upper_bound = np.zeros(self.nx) + inf
            self.standard_deviation = np.array([1., 1., 1., 1.]) * np.sqrt(2.9105444796447138e-04)

            if normalize[0] == None:
                self.normalize_data = self.standard_deviation / max(self.standard_deviation)
            else:
                self.normalize_data = normalize

            if bounds_theta == None:
                self.lbtheta = [-20., 0.] * 2   + [0]*4
                self.ubtheta = [40.] * 4   + [5]*4#(self.ntheta)
                self.bounds_theta = [self.lbtheta, self.ubtheta]
            else:
                self.bounds_theta = bounds_theta

        def plant_model_real(self, transformation):
            """
            Define the model that is meant to describe the physical system
            :return: model f
            """
            A = self.A
            x = MX.sym('x', self.nx)
            u = MX.sym('u', self.nu)
            theta = MX.sym('theta', self.ntheta)

            self.constraints_x = []
            self.constraints_x += [self.lower_bound]
            self.constraints_x += [self.upper_bound]

            if transformation:
                k = exp(theta[0:4:2] - theta[1:4:2] * 1e4 / 8.314 * (1 / (u[0] + 273.15) - 1 / (90 + 273)))
                self.tranformed = True
            else:
                k = exp(theta[0:4:2] - theta[1:4:2] * 1e4 / 8.314 * (1 / (u[0] + 273.15) - 1 / (90 + 273)))

            power1 = theta[4:]
            r = []  # MX.sym('r', (A.shape[1],1))
            kk = 0
            for i in range(A.shape[1]):
                r1 = (x[0]) ** (np.heaviside(-A[0, i], 0) * power1[kk]) * k[i]
                for j in range(1, A.shape[0]):
                    if A[j, i] != 0:
                        if np.heaviside(-A[j, i], 0) == 1:
                            kk += 1
                        r1 = (x[j]) ** (np.heaviside(-A[j, i], 0) * power1[kk]) * r1

                r = vertcat(r, r1)
                if np.heaviside(-A[0, i], 0) == 1 and i > 0:
                    kk += 1
            xdot = A @ r  # +\

            # Quadrature
            L = []
            # Algebraic
            alg = []
            return xdot, L, alg, x, u, theta

        def Generate_fun_integrator(self, sensitivity=False, transformation=True):
            # Calculate on the fly dynamic sensitivities without the need of perturbations
            xdot, L, alg, x, u, theta = self.plant_model_real(transformation)
            # x = MX.sym('x', self.nx)
            # u = MX.sym('u', self.nu)
            # theta = MX.sym('theta', self.ntheta)

            if sensitivity:
                x_p = MX.sym('xp', np.shape(x)[0] * np.shape(theta)[0])
                xpdot = []
                for i in range(np.shape(theta)[0]):
                    xpdot = vertcat(xpdot, jacobian(xdot, x) @ (x_p[self.nx * i: self.nx * i + self.nx])
                                    + jacobian(xdot, theta)[self.nx * i: self.nx * i + self.nx])
                    f = Function('f', [x, u, theta, x_p], [xdot, L, xpdot],
                                 ['x', 'u', 'theta', 'xp'], ['xdot', 'L', 'xpdot'])
            else:
                f = Function('f', [x, u, theta], [xdot, L], ['x', 'u', 'theta'], ['xdot', 'L'])

            nu = np.shape(u)[0]
            nx = np.shape(x)[0]
            ntheta = np.shape(theta)[0]
            self.nx     = nx
            self.nu     = nu
            self.ntheta = ntheta
            return f, nu, nx, ntheta

        # def transformations_parameters(self, theta):
        #     theta_nom = theta.copy()
        #     theta_nom[0:4:2] = np.exp(theta[0::2])
        #     theta_nom[1:4:2] = theta[1:4:2]*1e4
        #     self.theta_nominal = theta_nom
        #     return theta_nom
        #
        def integration(self,dt, sensitivity=False, transformation=True):
            X = SX.sym('X0', self.nx)
            U = SX.sym('U', self.nu)
            theta = SX.sym('theta', self.ntheta)
            xp0 = SX.sym('xp', np.shape(X)[0] * np.shape(theta)[0])
            f, nu, nx, ntheta = self.Generate_fun_integrator(sensitivity, transformation)
            if sensitivity:
                xdot, qj, xpdot = f(X, U, theta, xp0)
                dae = {'x': vertcat(X, xp0), 'p': vertcat(U, theta), 'ode': vertcat(xdot, xpdot)}
            else:
                xdot, qj = f(X, U, theta)
                dae = {'x': vertcat(X), 'p': vertcat(U, theta), 'ode': vertcat(xdot)}
            opts = {'tf': dt}  # interval length
            F_int = integrator('F', 'cvodes', dae, opts)
            return  F_int

        def perform_evaluation(self, x, u, theta, dt, sensitivity=False, transformation=True):
            xp1 = []
            F_int = self.integration(dt, sensitivity, transformation)
            if sensitivity:
                Fk            = F_int(x0=vertcat(x, xp1), p=vertcat(u, theta))
                x11 = Fk['xf'][0:self.nx]
                xp1 = Fk['xf'][self.nx:]
                return x11, xp1
            else:
                Fk            = F_int(x0=vertcat(x, xp1), p=vertcat(u, theta))
                x11 = Fk['xf'][0:self.nx]
                return x11


