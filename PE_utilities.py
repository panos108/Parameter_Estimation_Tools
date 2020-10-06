from casadi import *
import numpy as np
import scipy.stats as stats

def construct_polynomials_basis(d, poly_type):

    # Get collocation points
    tau_root = np.append(0, collocation_points(d, poly_type))

    # Coefficients of the collocation equation
    C = np.zeros((d + 1, d + 1))

    # Coefficients of the continuity equation
    D = np.zeros(d + 1)

    # Coefficients of the quadrature function
    B = np.zeros(d + 1)

    # Construct polynomial basis
    for j in range(d + 1):
        # Construct Lagrange polynomials to get the polynomial basis at the collocation point
        p = np.poly1d([1])
        for r in range(d + 1):
            if r != j:
                p *= np.poly1d([1, -tau_root[r]]) / (tau_root[j] - tau_root[r])

        # Evaluate the polynomial at the final time to get the coefficients of the continuity equation
        D[j] = p(1.0)
        # Evaluate the time derivative of the polynomial at all collocation points to get the coefficients of the continuity
        # equation
        pder = np.polyder(p)
        for r in range(d + 1):
            C[j, r] = pder(tau_root[r])

        # Evaluate the integral of the polynomial to get the coefficients of the quadrature function
        pint = np.polyint(p)
        B[j] = pint(1.0)

    return C, D, B

class ParameterEstimation:

    def __init__(self, Model_Def, collocation_degree = 8, intermediate_elements=25,
                 N_exp_max=80, t_span=2):
        self.Model_def = Model_Def             # Take class of the dynamic system
        self.dc        = collocation_degree    # Define the degree of collocation
        self.int_elem  = intermediate_elements # Define the Horizon of the problem
        self.N_exp_max = N_exp_max
        self.t_span    = t_span
        self.f, self.nu, self.nx, self.ntheta = self.Model_def.Generate_fun_integrator(sensitivity=False)
        self.measured  = self.Model_def.measured

        theta0        = self.Model_def.bounds_theta[0]

        # Define options for solver
        opts = {}
        opts["expand"] = True
        opts["ipopt.print_level"] = 5
        opts["ipopt.max_iter"] = 1000
        opts["ipopt.tol"] = 1e-12
        opts["calc_lam_p"] = False
        opts["calc_multipliers"] = False
        opts["ipopt.print_timing_statistics"] = "no"
        opts["print_time"] = True
        self.opts = opts

        self.Construct_col(self.Model_def.bounds_theta, theta0)

        self.Construct_NLP(self.Model_def.bounds_theta, theta0)


    def Construct_NLP(self, bounds_theta, theta0):
        # def construct_NLP_collocation(N_exp, f, x_0, x_init, lbx, ubx, lbu, ubu, lbtheta, ubtheta,
        #                               dt, N, x_meas, theta0, d, ms):

        nx = self.nx
        ntheta = self.ntheta
        nu = self.nu
        lbtheta = bounds_theta[0]
        ubtheta = bounds_theta[1]

        # xmin = np.zeros(nx - 1)
        # xmax = np.zeros(nx - 1)  # -1)
        # x_meas_norm = x_meas.copy()
        # for i in range(nx - 1):
        #     xmax[i] = np.max(x_meas[:, i, :])
        #     if xmax[i] > 1e-9:
        #         x_meas_norm[:, i, :] = x_meas[:, i, :] / xmax[i]
        #     else:
        #         x_meas_norm[:, i, :] = x_meas[:, i, :]
        #         xmax[i] = 1.

        C, D, B = construct_polynomials_basis(self.dc, 'radau')
        # Start with an empty NLP
        # x0      = MX.sym('x_0', ((self.t_span-1) * self.N_exp_max, nx))
        # x_meas  = MX.sym('x_exp', ((self.t_span-1) * self.N_exp_max, nx))
        # u_meas  = MX.sym('u_exp', ((self.t_span-1) * self.N_exp_max), nu)
        # dt      = MX.sym('dt_exp', (self.N_exp_max * (self.t_span-1)))
        # shrink  = MX.sym('p_shrink', (self.N_exp_max*(self.t_span-1)))

        x01      = MX.sym('x_0', (self.N_exp_max * nx))
        x_meas1  = MX.sym('x_exp', ((self.t_span-1) * self.N_exp_max* nx))
        u_meas1  = MX.sym('u_exp', ((self.t_span-1) * self.N_exp_max)* nu)
        dt      = MX.sym('dt_exp', (self.N_exp_max * (self.t_span-1)))
        shrink  = MX.sym('p_shrink', (self.N_exp_max*(self.t_span-1)))




        x0 = reshape(x01, (nx, self.N_exp_max))
        x_meas = reshape(x_meas1, (nx,( self.t_span-1) * self.N_exp_max))
        u_meas = reshape(u_meas1, (nu, (self.t_span-1) * self.N_exp_max))


        w = []
        w0 = []
        lbw = []
        ubw = []

        g = []
        lbg = []
        ubg = []

        # For plotting x and u given w
        x_plot     = []
        x_measured =[]
        x_plotp    = []
        u_plot     = []
        dt_plot    = []
        mle = 0
        chi2 = 0
        s = 0

        thetak = MX.sym('theta', ntheta)
        w += [thetak]

        lbw.extend(lbtheta)
        ubw.extend(ubtheta)

        w0.extend(theta0)
        for k_exp in range(self.N_exp_max):
            if s > 0:
                s += 1
            # "Lift" initial conditions
            Xk = MX.sym('X_' + str(s), nx)
            w += [Xk]
            lbw.extend([-inf]*nx)#x_init[k_exp][:].tolist())
            ubw.extend([+inf]*nx)#x_init[k_exp][:].tolist())
            w0.extend([0]*nx)#x_init[k_exp][:].tolist())
            g += [(Xk - x0.T[k_exp,:].T)]
            lbg += [*np.zeros([nx])]
            ubg += [*np.zeros([nx])]

            x_plot += [Xk]

            # Uk = MX.sym('U_' + str(k_exp), nu)
            # w += [Uk]
            # # lbw.extend(lbu[0][k_exp][:].tolist())
            # # ubw.extend(ubu[0][k_exp][:].tolist())
            # # w0.extend(ubu[0][k_exp][:].tolist())
            # g += [(Uk - u_meas[k_exp][:,])*shrink[k_exp]]
            # lbg += [*np.zeros([nx])]
            # ubg += [*np.zeros([nx])]
            # u_plot += [Uk]
            # Formulate the NLP
            ms = self.int_elem
            for k in range((self.t_span-1)):
                if s > 0:
                    s += 1
                # New NLP variable for the control
                Uk = MX.sym('U_' + str(k_exp*(self.t_span-1)+k), nu)
                w += [Uk]
                lbw.extend([-inf]*nu)
                ubw.extend([+inf]*nu)
                w0.extend([0.]*nu)
                g += [(Uk - u_meas.T[k_exp*(self.t_span-1)+k,:].T) * shrink[k_exp]]
                lbg += [*np.zeros([nu])]
                ubg += [*np.zeros([nu])]
                u_plot += [Uk]


                h = dt[k_exp*(self.t_span-1)+k] / ms
                s=0
                for k_in in range(ms):
                    # --------------------
                    # State at collocation points
                    # ---------------------------------


                    lbw, ubw, w0, w, lbg, ubg, g, Xk = self.collocation(self.f, self.dc, s, nx, self.Model_def.lower_bound,
                                                                   self.Model_def.upper_bound, lbw, ubw, w0, w,
                                                                   lbg, ubg, g, Xk, shrink[k], Uk, thetak, h, C,
                                                                   D)

                    # lbw, ubw, w0, w, lbg, ubg, g, Xk = self.collocation(
                    #     self.f, self.dc, s, nx, nu, [-inf]*nx, [inf]*nx, lbw, ubw, w0, w,
                    #     lbg, ubg, g, np.zeros(nx), Xk, k_exp, 0, Uk, thetak, h, C, D)
                    # k1, _ = self.f(Xk, Uk, thetak)
                    # k2, _ = self.f(Xk + h / 2 * k1, Uk, thetak)
                    # k3, _ = self.f(Xk + h / 2 * k2, Uk, thetak)
                    # k4, _ = self.f(Xk + h * k3, Uk, thetak)
                    # Xk = Xk + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
                    # ---------------------------------
                    # if divmod(k + 1, ms)[1] == 0:
                    x_plot += [Xk]
                    s += 1
                x_measured += [Xk]
                mle += self.maximum_likelihood_est(k_exp, Xk, x_meas,
                                                   self.Model_def.normalize_data, k)*shrink[k_exp]
                        # chi2 += 2 * maximum_likelihood_est(k_exp, Xk[:-1], x_meas, [0.005, 0.005, 0.003, 0.003], m,
                        #                                    [1.] * 4)*shrink[k_exp]

        # hs, _ = hessian(mle, vertcat(*w))
        # h_fn = Function('mle', [vertcat(*w)], [hs], ['w'], ['hss'])
        # Concatenate vectors
        # w = vertcat(*w)
        # g = vertcat(*g)
        # x_plot = horzcat(*x_plot)
        # u_plot = horzcat(*u_plot)
        # w0 = np.concatenate(w0)
        # lbw = np.concatenate(lbw)
        # ubw = np.concatenate(ubw)
        # lbg = np.concatenate(lbg)
        # ubg = np.concatenate(ubg)

        p =  []
        p += [x01]
        p += [x_meas1]
        p += [u_meas1]
        p += [dt]
        p += [shrink]
        # Create an NLP solver
        problem = {'f': mle, 'x': vertcat(*w), 'p': vertcat(*p),'g': vertcat(*g)}
        trajectories = Function('trajectories', [vertcat(*w)]
                                , [horzcat(*x_plot), horzcat(*u_plot), horzcat(*x_plotp), chi2, thetak, horzcat(*x_measured)], ['w'],
                                ['x', 'u', 'xp',
                                 'chi2', 'theta', 'x_measured'])

        solver = nlpsol('solver', 'ipopt', problem, self.opts)  # 'bonmin', prob, {"discrete": discrete})#'ipopt', prob, {'ipopt.output_file': 'error_on_fail'+str(ind)+'.txt'})#
        self.solver, self.trajectories, self.w0, self.lbw, self.ubw, self.lbg, self.ubg = \
            solver, trajectories, w0, lbw, ubw, lbg, ubg

        return problem, w0, lbw, ubw, lbg, ubg, trajectories#, h_fn


    def collocation(self, f, d, s, nx, lbx, ubx, lbw, ubw, w0, w,
                lbg, ubg, g, Xk, shrink, Uk, thetak, h, C, D):

        Xc = []

        for j in range(d):
            Xkj = MX.sym('X_' + str(s) + '_' + str(j), nx)
            Xc += [Xkj]
            w += [Xkj]
            lbw.extend(lbx)
            ubw.extend(ubx)
            #                ubw.extend([u_meas[k_exp][1]])
            w0.extend([0.]*nx)#, m])
        #                w0.extend([u_meas[k_exp][1]])

        # Loop over collocation points
        Xk_end = D[0] * Xk

        for j in range(1, d + 1):
            # Expression for the state derivative at the collocation point
            xp = C[0, j] * Xk

            for r in range(d):
                xp = xp + C[r + 1, j] * Xc[r]

            # Append collocation equations
            fj, qj = f(Xc[j - 1], Uk, thetak) # Xpc[j - 1])
            g += [(h * fj*shrink  - xp)]
            lbg.extend([-1e-9] * nx)
            ubg.extend([1e-9] * nx)

            # Add contribution to the end state
            Xk_end = Xk_end + D[j] * Xc[j - 1]

        # New NLP variable for state at end of interval
        Xk = MX.sym('X_' + str(s + 1), nx)
        w += [Xk]
        lbw.extend(lbx)
        ubw.extend(ubx)  # [:-1])
        #            ubw.extend([u_meas[k_exp][1]])
        w0.extend([0.]*nx)#, m])
        #            w0.extend([u_meas[k_exp][1]])

        # Add equality constraint
        g += [Xk_end - Xk]
        lbg.extend([-1e-9] * nx)
        ubg.extend([1e-9] * nx)
        return lbw, ubw, w0, w, lbg, ubg, g, Xk
    #
    # def collocation(self, f, d, s, nx, nu, lbx, ubx, lbw, ubw, w0, w,
    #                 lbg, ubg, g, x_meas, Xk, k_exp, m, Uk, thetak, h, C, D):
    #     Xc = []
    #
    #     for j in range(d):
    #         Xkj = MX.sym('X_' + str(s) + '_' + str(j), nx)
    #         Xc += [Xkj]
    #         w += [Xkj]
    #         lbw.extend(lbx)
    #         ubw.extend(ubx)
    #         #                ubw.extend([u_meas[k_exp][1]])
    #         w0.extend(np.zeros(np.shape(lbx)))  # , m])
    #     #                w0.extend([u_meas[k_exp][1]])
    #
    #     # Loop over collocation points
    #     Xk_end = D[0] * Xk
    #
    #     for j in range(1, d + 1):
    #         # Expression for the state derivative at the collocation point
    #         xp = C[0, j] * Xk
    #
    #         for r in range(d):
    #             xp = xp + C[r + 1, j] * Xc[r]
    #
    #         # Append collocation equations
    #         fj, qj = f(Xc[j - 1], Uk, thetak)  # Xpc[j - 1])
    #         g += [(h * fj - xp)]
    #         lbg.extend([-1e-9] * nx)
    #         ubg.extend([1e-9] * nx)
    #
    #         # Add contribution to the end state
    #         Xk_end = Xk_end + D[j] * Xc[j - 1]
    #
    #     # New NLP variable for state at end of interval
    #     Xk = MX.sym('X_' + str(s + 1), nx)
    #     w += [Xk]
    #     lbw.extend(lbx)
    #     ubw.extend(ubx)  # [:-1])
    #     #            ubw.extend([u_meas[k_exp][1]])
    #     w0.extend(x_meas * 0)  # , m])
    #     #            w0.extend([u_meas[k_exp][1]])
    #
    #     # Add equality constraint
    #     g += [Xk_end - Xk]
    #     lbg.extend([-1e-9] * nx)
    #     ubg.extend([1e-9] * nx)
    #     return lbw, ubw, w0, w, lbg, ubg, g, Xk

    def maximum_likelihood_est(self,i, y, y_meas, sigma, k):
        """
        This is a function that computes the MLE for a given set of experiments
        """
        #    N = 100#y_meas.shape[0]
        M = sum(self.measured)#y_meas.shape[1]

        MLE = 0
        s = 0
        for j in range(self.nx):
            if self.measured[j]:
                MLE += 0.5 * (y[j] - y_meas.T[i*(self.t_span-1)+k, j]) ** 2 / sigma[j] ** 2
        return MLE


    def solve_pe(self, x_meas, u_meas, dt):

        # ADD CHANGE OF DT FROM T VALUES

        #Change this! to maximum x_meas
        if u_meas.ndim == 2:
            u_meas = np.expand_dims(u_meas, axis=2)
        else:
            u_meas = u_meas
        measured = self.Model_def.measured


        N_exp        = sum(x_meas[:, 0, 0] > 0)
        x_zeros      = np.zeros([self.N_exp_max, self.nx, self.t_span-1])
        u_zeros      = np.zeros([self.N_exp_max, u_meas.shape[1],self.t_span-1])
        dt_zeros     = np.zeros([self.N_exp_max, self.t_span-1])
        x0_zeros     = np.zeros([self.N_exp_max, self.nx])
        for i in range(self.nx):
            x0_zeros[:N_exp, i] = x_meas[:N_exp, i, 0]
            if measured[i]:
                x_zeros[:N_exp,i,:]  = x_meas[:N_exp,i,1:]

        u_zeros[:N_exp,:,:]  = u_meas[:N_exp,:,:]
        dt_zeros[:N_exp,:]   = dt[:N_exp,:]


        # dt            = dt[:N_exp]


        #x0_exp_pe_tep = x0_zeros.transpose([1, 0])
        #x0_exp_pe      = np.reshape(x0_exp_pe_tep.reshape((-1,x_meas.shape[1])),(-1,1),)
        x0_exp_pe      = np.reshape(x0_zeros.reshape((-1,x_meas.shape[1])),(-1,1),)

        #x_exp_pe_temp = x_zeros.transpose([1, 0, 2])
        #x_exp_pe      = np.reshape(x_exp_pe_temp.reshape((-1,x_meas.shape[1])),(-1,1),)
        x_exp_pe      = np.reshape(x_zeros.reshape((-1,x_meas.shape[1])),(-1,1),)

        #-------u_exp_pe_temp0= u_meas.reshape((-1,-1,))------------#
        #u_exp_pe_temp = u_zeros.transpose([1, 0, 2])
        #u_exp_pe      = np.reshape(u_exp_pe_temp.reshape((-1,x_meas.shape[1])),(-1,1),)
        u_exp_pe      = np.reshape(u_zeros.reshape((-1,u_meas.shape[1])),(-1,1),)

        dt_pe         = dt_zeros.reshape((-1,1))

        shrink        = np.concatenate((np.ones([N_exp]*(self.t_span-1)),
                                        np.zeros([self.N_exp_max-N_exp]*(self.t_span-1)))).reshape((-1,1))
        self.dof = sum(self.measured) * sum(shrink) * (self.t_span - 1) - self.ntheta

        p0      = np.concatenate((x0_exp_pe, x_exp_pe, u_exp_pe, dt_pe, shrink))
        self.p0 = p0
        #---------------------
        sol = self.solver(x0=self.w0, lbx=self.lbw, ubx=self.ubw, lbg=self.lbg, ubg=self.ubg,
                          p=p0)

        w_opt = sol['x'].full().flatten()
        x_opt, u_opt, xp_opt, chi2, theta, x_measured = self. trajectories(sol['x'])
        if self.solver.stats()['return_status'] != 'Solve_Succeeded':
            print('Opt failed')

        self.obj   = sol['f'].full().flatten()
        self.theta = np.array(theta)
        return u_opt, x_opt, w_opt, x_measured, theta


    def hessian_compute(self, theta=None):
        if theta == None:
            theta = self.theta
        self.hessian = self.h_fn(theta, self.p0)

    def Confidence_intervals(self, theta=None, confidence=0.95):
        if theta == None:
            theta = self.theta
        self.hessian = self.h_fn(theta, self.p0)
        inverse_hessian = np.linalg.inv(self.hessian)
        t = np.zeros([self.ntheta])
        for i in range(self.ntheta):
            t[i] = theta[i] / (np.sqrt(inverse_hessian[i, i]) * stats.t.ppf((1 - (1-confidence) / 2), self.dof))
        t_ref = stats.t.ppf(confidence, self.dof)
        CI    = np.sqrt(diag(inverse_hessian)) * stats.t.ppf((1 - (1 - confidence) / 2), self.dof)
        chi2  = self.mle(theta, self.p0)

        statistics = {'CI':CI, 't':t, 'chi2':chi2, 't_ref':t_ref}
        return CI, t, t_ref, chi2, statistics


    def t_values(self, theta=None, confidence=0.95):
        if theta == None:
            theta = self.theta
        self.hessian = self.h_fn(theta, self.p0)
        np.sqrt(diag(self.hessian))* stats.t.ppf((1 - (1-confidence) / 2), self.dof)



    def Construct_col(self, bounds_theta, theta0):
            # def construct_NLP_collocation(N_exp, f, x_0, x_init, lbx, ubx, lbu, ubu, lbtheta, ubtheta,
            #                               dt, N, x_meas, theta0, d, ms):

        nx = self.nx
        ntheta = self.ntheta
        nu = self.nu
        lbtheta = bounds_theta[0]
        ubtheta = bounds_theta[1]

        x01 = MX.sym('x_0', (self.N_exp_max * nx))
        x_meas1 = MX.sym('x_exp', ((self.t_span - 1) * self.N_exp_max * nx))
        u_meas1 = MX.sym('u_exp', ((self.t_span - 1) * self.N_exp_max) * nu)
        dt = MX.sym('dt_exp', (self.N_exp_max * (self.t_span - 1)))
        shrink = MX.sym('p_shrink', (self.N_exp_max * (self.t_span - 1)))

        x0 = reshape(x01, (nx, self.N_exp_max))
        x_meas = reshape(x_meas1, (nx, (self.t_span - 1) * self.N_exp_max))
        u_meas = reshape(u_meas1, (nu, (self.t_span - 1) * self.N_exp_max))

        w = []
        w0 = []
        lbw = []
        ubw = []


        x_plot = []
        u_plot = []
        mle = 0
        s = 0

        thetak = MX.sym('theta', ntheta)
        w += [thetak]

        lbw.extend(lbtheta)
        ubw.extend(ubtheta)

        w0.extend(theta0)
        for k_exp in range(self.N_exp_max):
            if s > 0:
                s += 1
            # "Lift" initial conditions
            Xk = x0.T[k_exp, :].T#MX.sym('X_' + str(s), nx)
            w += [Xk]
            lbw.extend([-inf] * nx)  # x_init[k_exp][:].tolist())
            ubw.extend([+inf] * nx)  # x_init[k_exp][:].tolist())
            w0.extend([0] * nx)  # x_init[k_exp][:].tolist())

            x_plot += [Xk]

            ms = self.int_elem*2
            for k in range((self.t_span - 1)):
                if s > 0:
                    s += 1
                # New NLP variable for the control
                Uk = u_meas.T[k_exp * (self.t_span - 1) + k, :].T * shrink[k_exp]#MX.sym('U_' + str(k_exp * (self.t_span - 1) + k), nu)
                w += [Uk]
                lbw.extend([-inf] * nu)
                ubw.extend([+inf] * nu)
                w0.extend([0.] * nu)

                u_plot += [Uk]

                h = dt[k_exp * (self.t_span - 1) + k] / ms
                s = 0
                for k_in in range(ms):
                    # --------------------
                    # State at collocation points
                    # ---------------------------------

                    k1, _ = self.f(Xk, Uk, thetak)
                    k2, _ = self.f(Xk + h / 2 * k1, Uk, thetak)
                    k3, _ = self.f(Xk + h / 2 * k2, Uk, thetak)
                    k4, _ = self.f(Xk + h * k3, Uk, thetak)
                    Xk = Xk + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

                    x_plot += [Xk]
                    s += 1

                mle += self.maximum_likelihood_est(k_exp, Xk, x_meas, self.Model_def.standard_deviation, k) * shrink[k_exp]

        p = []
        p += [x01]
        p += [x_meas1]
        p += [u_meas1]
        p += [dt]
        p += [shrink]


        hs, _ = hessian(mle, thetak)

        self.h_fn = Function('h',   [thetak,vertcat(*p)], [hs], ['w','p'], ['hss'])
        self.mle = Function('mle',  [thetak, vertcat(*p)], [mle], ['w', 'p'], ['mle'])


