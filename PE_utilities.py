from casadi import *
import numpy as np

class Reactor_pfr_model:
    def __init__(self, A):

        self.A = A
        self.nx = A.shape[0]
        self.ntheta = A.shape[1]*2
        self.nu = 4
    def plant_model_real(self, sensitivity=False, transformation=True):
        """
        Define the model that is meant to describe the physical system
        :return: model f
        """
        nx = self.nx
        ntheta = self.ntheta
        nu = self.nu
        A  = self.A

        x = MX.sym('x', nx)
        u = MX.sym('u', nu)
        theta = MX.sym('theta', ntheta)

        x_p = MX.sym('xp', np.shape(x)[0] * np.shape(theta)[0])

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
        L = []  # x1 ** 2 + x2 ** 2 + 1*u1 ** 2 + 1*u2**2
        # Algebraic
        alg = []

        # Calculate on the fly dynamic sensitivities without the need of perturbations
        if sensitivity:
            xpdot = []
            for i in range(np.shape(theta)[0]):
                xpdot = vertcat(xpdot, jacobian(xdot, x) @ (x_p[nx * i: nx * i + nx])
                                + jacobian(xdot, theta)[nx * i: nx * i + nx])
                f = Function('f', [x, u, theta, x_p], [xdot, L, xpdot],
                             ['x', 'u', 'theta', 'xp'], ['xdot', 'L', 'xpdot'])
        else:
            f = Function('f', [x, u, theta], [xdot, L], ['x', 'u', 'theta'], ['xdot', 'L'])

        nu = u.shape
        nx = x.shape
        ntheta = theta.shape

        return f, nu, nx, ntheta
