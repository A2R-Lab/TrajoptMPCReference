import importlib
import numpy as np
import copy
import enum
from TrajoptPlant import TrajoptPlant, DoubleIntegratorPlant, PendulumPlant, CartPolePlant, URDFPlant
from TrajoptCost import TrajoptCost, QuadraticCost
from TrajoptConstraint import TrajoptConstraint, BoxConstraint
PCG = importlib.import_module("GBD-PCG-Python").PCG
BCHOL = importlib.import_module("BCHOL-python").BCHOL
##delete later##
buildBCHOL = importlib.import_module("BCHOL-python").buildBCHOL

np.set_printoptions(precision=4, suppress=True, linewidth = 100)

class SQPSolverMethods(enum.Enum):
    N = "N"
    S = "S"
    PCG_J = "PCG-J"
    PCG_BJ = "PCG-BJ"
    PCG_SS = "PCG-SS"
    #Should I ADD BCHOL HERE?
    # BCHOL = "BCHOL"


class MPCSolverMethods(enum.Enum):
    iLQR = "iLQR"
    DDP = "DDP"
    QP_N = "QP-N"
    QP_S = "QP-S"
    QP_PCG_J = "QP-PCG-J"
    QP_PCG_BJ = "QP-PCG-BJ"
    QP_PCG_SS = "QP-PCG-SS"
    #Should I ADD BCHOL HERE?
    # BCHOL = "BCHOL"

class TrajoptMPCReference:
    def __init__(self, plantObj:TrajoptPlant, costObj: TrajoptCost, constraintObj: TrajoptConstraint = None):
        if (not issubclass(type(plantObj),TrajoptPlant) or not issubclass(type(costObj),TrajoptCost)):
            print("Must pass in a TrajoptPlant and TrajoptCost object to TrajoptMPCReference.")
            exit()
        if constraintObj is None:
            constraintObj = TrajoptConstraint()
        elif not issubclass(type(constraintObj),TrajoptConstraint):
            print("If passing in additional constraints must pass in a TrajoptConstraint object to TrajoptMPCReference.")
            exit()
        self.plant = plantObj
        self.cost = costObj
        self.other_constraints = constraintObj

    def update_cost(self, costObj: TrajoptCost):
        assert issubclass(type(costObj),TrajoptCost), "Must pass in a TrajoptCost object to update_cost in TrajoptMPCReference."
        self.cost = costObj

    def update_plant(self, plantObj: TrajoptPlant):
        assert issubclass(type(plantObj),TrajoptPlant), "Must pass in a TrajoptPlant object to update_plant in TrajoptMPCReference."
        self.plant = plantObj

    def update_constraints(self, constraintObj: TrajoptConstraint):
        assert issubclass(type(constraintObj),TrajoptConstraint), "Must pass in a TrajoptConstraint object to update_constraints in TrajoptMPCReference."
        self.other_constraints = constraintObj

    def set_default_options(self, options: dict):
        # LinSys options (mostly for PCG)
        options.setdefault('exit_tolerance_linSys', 1e-6)
        options.setdefault('max_iter_linSys', 100)
        options.setdefault('DEBUG_MODE_linSys', False)
        options.setdefault('RETURN_TRACE_linSys', False)
        # DDP/SQP options
        options.setdefault('exit_tolerance_SQP_DDP', 1e-6)
        options.setdefault('max_iter_SQP_DDP', 25)
        options.setdefault('DEBUG_MODE_SQP_DDP', False)
        options.setdefault('alpha_factor_SQP_DDP', 0.5)
        options.setdefault('alpha_min_SQP_DDP', 0.005)
        options.setdefault('rho_factor_SQP_DDP', 4)
        options.setdefault('rho_min_SQP_DDP', 1e-3)
        options.setdefault('rho_max_SQP_DDP', 1e3)
        options.setdefault('rho_init_SQP_DDP', 0.001)
        options.setdefault('expected_reduction_min_SQP_DDP', 0.05)
        options.setdefault('expected_reduction_max_SQP_DDP', 3)
        # DDP/iLQR lag
        options.setdefault('DDP_flag', False)
        # SQP only options
        options.setdefault('merit_factor_SQP', 1.5)
        options.setdefault('RETURN_TRACE_SQP', False)
        # DDP only options
        options.setdefault('state_regularization_DDP', True)
        options.setdefault('print_full_trajectory_DDP', True)
        # AL and ADMM options
        options.setdefault('exit_tolerance_softConstraints', 1e-6)
        options.setdefault('max_iter_softConstraints', 10)
        options.setdefault('DEBUG_MODE_Soft_Constraints', False)
        # MPC options
        options.setdefault('num_timesteps_per_solve_mpc',  1)
        options.setdefault('simulator_steps_mpc', 1)

    def formKKTSystemBlocks(self, x: np.ndarray, u: np.ndarray, xs: np.ndarray, N: int, dt: float):
        nq = self.plant.get_num_pos()
        nv = self.plant.get_num_vel()
        nu = self.plant.get_num_cntrl()
        nx = nq + nv
        n = nx + nu
        # G,g are cost hessian and gradient with n*(N-1) + nx state and control variables 
        total_states_controls = n*(N-1) + nx
        G = np.zeros((total_states_controls, total_states_controls))
        g = np.zeros((total_states_controls, 1))
        # C,c are constraint gradient (Jacobian) and value and depends on both the
        #     N-1 dynamics constarints, the initial state constaint, and any additional constraints
        total_dynamics_intial_state_constraints = nx*N
        total_other_constraints = self.other_constraints.total_hard_constraints(x, u)
        total_constraints = total_dynamics_intial_state_constraints + total_other_constraints
        C = np.zeros((total_constraints, total_states_controls))
        c = np.zeros((total_constraints, 1))

        # start filling from the top left of the matricies (and top of vectors)
        constraint_index = 0
        state_control_index = 0
        # begin with the initial state constraint
        C[constraint_index:constraint_index + nx, state_control_index:state_control_index + nx] = np.eye(nx)
        c[constraint_index:constraint_index + nx, 0] = x[:,0] - xs
        constraint_index += nx

        for k in range(N-1):
            # first load in the cost hessian and gradient
            G[state_control_index:state_control_index + n, \
              state_control_index:state_control_index + n] = self.cost.hessian(x[:,k], u[:,k], k)
            g[state_control_index:state_control_index + n, 0] = self.cost.gradient(x[:,k], u[:,k], k)
            # add soft constraints if applicable
            if self.other_constraints.total_soft_constraints(timestep = k) > 0:
                gck = self.other_constraints.jacobian_soft_constraints(x[:,k], u[:,k], k)
                g[state_control_index:state_control_index + n, :] += gck
                G[state_control_index:state_control_index + n, \
                  state_control_index:state_control_index + n] += np.outer(gck,gck)

            # then load in the constraints for this timestep starting with dynamics
            Ak, Bk = self.plant.integrator(x[:,k], u[:,k], dt, return_gradient = True)
            C[constraint_index:constraint_index + nx, \
              state_control_index:state_control_index + n + nx] = np.hstack((-Ak, -Bk, np.eye(nx)))
            xkp1 = self.plant.integrator(x[:,k], u[:,k], dt)
            c[constraint_index:constraint_index + nx, 0] = x[:,k+1] - xkp1
            constraint_index += nx
            
            # and then other constraints
            if total_other_constraints > 0 and self.other_constraints.total_hard_constraints(x, u, k):
                jac = self.other_constraints.jacobian_hard_constraints(x[:,k], u[:,k], k)
                val = self.other_constraints.value_hard_constraints(x[:,k], u[:,k], k)
                if val is not None and len(val):
                    num_active_const_k = len(val)
                    C[constraint_index:constraint_index + num_active_const_k, \
                      state_control_index:state_control_index + n] = np.reshape(jac, (num_active_const_k,n))
                    c[constraint_index:constraint_index + num_active_const_k] = np.reshape(val, (num_active_const_k,1))
                    constraint_index += num_active_const_k
            
            # then update the state_control_index
            state_control_index += n

        # finish with the final cost
        G[state_control_index:state_control_index + nx, \
          state_control_index:state_control_index + nx] = self.cost.hessian(x[:,N-1], timestep = N-1)
        g[state_control_index:state_control_index + nx, 0] = self.cost.gradient(x[:,N-1], timestep = N-1)
        # add soft constraints if applicable
        if self.other_constraints.total_soft_constraints(timestep = N-1) > 0:
            gcNm1 = self.other_constraints.jacobian_soft_constraints(x[:,N-1], timestep = N-1)
            g[state_control_index:state_control_index + nx, :] += gcNm1
            G[state_control_index:state_control_index + nx, \
              state_control_index:state_control_index + nx] += np.outer(gcNm1,gcNm1)

        # and the final constraint
        if total_other_constraints > 0 and self.other_constraints.total_hard_constraints(x, u, N-1):
            jac = self.other_constraints.jacobian_hard_constraints(x[:,N-1], timestep = N-1)
            val = self.other_constraints.value_hard_constraints(x[:,N-1], timestep = N-1)
            if val is not None and len(val):
                num_active_const_k = len(val)
                C[constraint_index:constraint_index + num_active_const_k, \
                  state_control_index:state_control_index + nx] = np.reshape(jac, (num_active_const_k,nx))
                c[constraint_index:constraint_index + num_active_const_k] = np.reshape(val, (num_active_const_k,1))

        return G, g, C, c

    def totalHardConstraintViolation(self, x: np.ndarray, u: np.ndarray, xs: np.ndarray, N: int, dt: float, mode = None):
        mode_func = sum
        if mode == "MAX":
            mode_func = max
        # first do initial state and dynamics
        x_err = x[:,0] - xs
        err = list(map(abs,x_err))
        c = mode_func(err)
        for k in range(N-1):
            xkp1 = self.plant.integrator(x[:,k], u[:,k], dt)
            x_err = x[:,k+1] - xkp1
            c += mode_func(list(map(abs,x_err)))
        # then do all other constraints
        if self.other_constraints.total_hard_constraints(x, u) > 0:
            for k in range(N-1):
                if self.other_constraints.total_hard_constraints(x, u, k):
                    c_err = self.other_constraints.value_hard_constraints(x[:,k], u[:,k], k)
                    c += mode_func(list(map(abs,c_err)))
            if self.other_constraints.total_hard_constraints(x, u, N-1):
                c_err = self.other_constraints.value_hard_constraints(x[:,N-1], N-1)
                c += mode_func(list(map(abs,c_err)))
        return c

    def totalCost(self, x: np.ndarray, u: np.ndarray, N: int):
        J = 0
        for k in range(N-1):
            J += self.cost.value(x[:,k], u[:,k], k)
        J += self.cost.value(x[:,N-1], timestep = N-1)
        # add soft constraints if applicable
        if self.other_constraints.total_soft_constraints() > 0:
            for k in range(N-1):
                J += self.other_constraints.value_soft_constraints(x[:,k], u[:,k], k)
            J += self.other_constraints.value_soft_constraints(x[:,N-1], timestep = N-1)
        return J

    def solveKKTSystem(self, x: np.ndarray, u: np.ndarray, xs: np.ndarray, N: int, dt: float, rho: float = 0.0, options = {}):
        nq = self.plant.get_num_pos()
        nv = self.plant.get_num_vel()
        nu = self.plant.get_num_cntrl()
        nx = nq + nv
        n = nx + nu
        
        G, g, C, c = self.formKKTSystemBlocks(x, u, xs, N, dt)
        #write them into file 

        ###

        total_dynamics_intial_state_constraints = nx*N
        total_other_constraints = self.other_constraints.total_hard_constraints(x, u)
        total_constraints = total_dynamics_intial_state_constraints + total_other_constraints
        BR = np.zeros((total_constraints,total_constraints))
        if rho != 0:
            G += rho * np.eye(G.shape[0])

        KKT = np.hstack((np.vstack((G, C)),np.vstack((C.transpose(), BR))))
        kkt = np.vstack((g, c))

        print("C\n", C)
        print("G\n", G)
        print("c\n", c)
        print("g\n", g)
        
        print("KKT\n")
        for row in KKT:
            np.set_printoptions(linewidth=np.inf)  # Ensure the output is on a single line
            print(row)
        print("hi")
        print("kkt\n", kkt)    

        #add 0s for last timetep of nu
        g_yana=np.append(g,np.zeros((nu)))
        Q,R,q,r,A,B,d = buildBCHOL(G,g_yana,C,c,N,nx,nu)
        #maybe print Q,R,q,r,A,B,d

        #solve with BCHOL
        b_dxul = BCHOL(N,nu,nx,Q,R,q,r,A,B,d)
        print("cholesky_soln:\n", b_dxul)
        # BCHOL(Q,R,A,B...)

        # ###YANA
        # print("KKT~~\n")
        # print(KKT)
        # print("kkt\n")
        # print(kkt)
        # print("sol\n")
        # print(np.linalg.solve(KKT, kkt))
        # ###
        breakpoint()

        try:
            print("solving with linalg")
            dxul = np.linalg.solve(KKT, kkt)
            print("solved\n")
        except:
            if options.get('DEBUG_MODE'):
                print("Warning singular KKT system -- solving with least squares.")
            dxul, _, _, _ = np.linalg.lstsq(KKT, kkt, rcond=None)
        print("soln\n ", dxul)
        if not (np.isclose(dxul,b_dxul).all()):
            print("solutions are different\n")
        else:
            print("solns are the same!")
        return dxul
    
    ###############
    ''''ADD HERE solveBCHOLSystem_Schur
    The following method serves as a bridge and translator between MPC vars and BCHOL
    nx = nstates
    nu = ninputs'''
    # def solveBCHOL(self,x:np.ndarray, u:np.ndarray,xs:np.ndarray,N:int,dt:float, rho:float = 0.0) :
        
        # nq = self.plant.get_num_pos()
        # nv = self.plant.get_num_vel()
        # nu = self.plant.get_num_cntrl()
        # nx = nq + nv
        

        # G,g,C,c = self.formKKTSystemBlocks(x,u,xs,N,dt)
       
        # dxul = solve_build.buildBCHOL(G,g,C,c,N,nx,nu)
        
        # #q,r,d are the solution vector, need to return q and r
        # return dxul


    
    ################

    def solveKKTSystem_Schur(self, x: np.ndarray, u: np.ndarray, xs: np.ndarray, N: int, dt: float, rho: float = 0.0, use_PCG = False, options = {}):
        nq = self.plant.get_num_pos()
        nv = self.plant.get_num_vel()
        nu = self.plant.get_num_cntrl()
        nx = nq + nv
        
        G, g, C, c = self.formKKTSystemBlocks(x, u, xs, N, dt)
        
        print("breakpoint")
        breakpoint()
        total_dynamics_intial_state_constraints = nx*N
        total_other_constraints = self.other_constraints.total_hard_constraints(x, u)
        total_constraints = total_dynamics_intial_state_constraints + total_other_constraints
        BR = np.zeros((total_constraints,total_constraints))

        if rho != 0:
            G += rho * np.eye(G.shape[0])

        invG = np.linalg.inv(G)
        S = BR - np.matmul(C, np.matmul(invG, C.transpose()))
        gamma = c - np.matmul(C, np.matmul(invG, g))

        if not use_PCG:
            try:
                l = np.linalg.solve(S, gamma)
            except:
                if options.get('DEBUG_MODE'):
                    print("Warning singular Schur system -- solving with least squares.")
                l, _, _, _ = np.linalg.lstsq(S, gamma, rcond=None)
        else:
            pcg = PCG(S, gamma, nx, N, options = options)
            if 'guess' in options.keys():
                pcg.update_guess(options['guess'])
            if options.get('RETURN_TRACE'):
                l, traces = pcg.solve()
            else:
                l = pcg.solve()

        gCl = g - np.matmul(C.transpose(), l)
        dxu = np.matmul(invG, gCl)

        dxul = np.vstack((dxu,l))

        if options.get('RETURN_TRACE'):
            return dxul, traces
        else:
            return dxul

    def reduce_regularization(self, rho: float, drho: float, options: dict):
        self.set_default_options(options)
        drho = min(drho/options['rho_factor_SQP_DDP'], 1/options['rho_factor_SQP_DDP'])
        rho = max(rho*drho, options['rho_min_SQP_DDP'])
        return rho, drho

    def check_for_exit_or_error(self, error: bool, delta_J: float, iteration: int, rho: float, drho: float, options):
        self.set_default_options(options)
        exit_flag = False
        if error:
            drho = max(drho*options['rho_factor_SQP_DDP'], options['rho_factor_SQP_DDP'])
            rho = max(rho*drho, options['rho_min_SQP_DDP'])
            if rho > options['rho_max_SQP_DDP']:
                if options['DEBUG_MODE_SQP_DDP']:
                    print("Exiting for max_rho")
                exit_flag = True
        elif delta_J < options['exit_tolerance_SQP_DDP']:
            if options['DEBUG_MODE_SQP_DDP']:
                print("Exiting for exit_tolerance_SQP_DDP")
            exit_flag = True
        
        if iteration == options['max_iter_SQP_DDP'] - 1:
            if options['DEBUG_MODE_SQP_DDP']:
                print("Exiting for max_iter")
            exit_flag = True
        else:
            iteration += 1
        return exit_flag, iteration, rho, drho

    def check_and_update_soft_constraints(self, x: np.ndarray, u: np.ndarray, iteration: int, options):
        exit_flag = False
        # check for exit for constraint convergence
        max_c = self.other_constraints.max_soft_constraint_value(x,u)
        if max_c < options['exit_tolerance_softConstraints']:
            if options['DEBUG_MODE_Soft_Constraints']:
                print("Exiting for Soft Constraint Convergence")
            exit_flag = True
        # check for exit for iterations
        if iteration == options['max_iter_softConstraints'] - 1:
            if options['DEBUG_MODE_Soft_Constraints']:
                print("Exiting for Soft Constraint Max Iters")
            exit_flag = True
        else:
            iteration += 1
        # if we are not exiting update soft constraint constants
        if not exit_flag:
            all_mu_over_limit_flag = self.other_constraints.update_soft_constraint_constants(x,u)
            # check if we need to exit for mu over the limit
            if all_mu_over_limit_flag:
                if options['DEBUG_MODE_Soft_Constraints']:
                    print("Exiting for Mu over limit for all soft constraints")
                exit_flag = True
        return exit_flag, iteration

    def SQP(self, x: np.ndarray, u: np.ndarray, N: int, dt: float, LINEAR_SYSTEM_SOLVER_METHOD: SQPSolverMethods = SQPSolverMethods.N, options = {}):
        self.set_default_options(options)
        options_linSys = {'DEBUG_MODE': options['DEBUG_MODE_linSys']}

        USING_PCG = LINEAR_SYSTEM_SOLVER_METHOD in [SQPSolverMethods.PCG_J, SQPSolverMethods.PCG_BJ, SQPSolverMethods.PCG_SS]
        if USING_PCG:
            options_linSys['exit_tolerance'] = options['exit_tolerance_linSys']
            options_linSys['max_iter'] = options['max_iter_linSys']
            options_linSys['RETURN_TRACE'] = options['RETURN_TRACE_linSys']
            options_linSys['preconditioner_type'] = LINEAR_SYSTEM_SOLVER_METHOD.value[4:]

        nq = self.plant.get_num_pos()
        nv = self.plant.get_num_vel()
        nu = self.plant.get_num_cntrl()
        nx = nq + nv
        n = nx + nu

        xs = copy.deepcopy(x[:,0])

        # Start the main loops (soft constraint outer loop)
        soft_constraint_iteration = 0
        while 1:
            # Initialize the QP solve
            J = 0
            c = 0
            rho = options['rho_init_SQP_DDP']
            drho = 1

            # Compute initial cost and constraint violation
            J = self.totalCost(x, u, N)
            c = self.totalHardConstraintViolation(x, u, xs, N, dt)

            # L1 merit function with balanced J and c
            mu = J/c if c != 0 else 10
            mu = 10
            merit = J + mu*c
            if options['DEBUG_MODE_SQP_DDP']:
                print("Initial Cost, Constraint Violation, Merit Function: ", J, c, merit)
            if options['RETURN_TRACE_SQP']:
                inner_iters = 0
                trace = [[J,c,0,inner_iters]]

            # Start the main loop (SQP main loop)
            iteration = 0
            while 1:

                #
                # Solve QP to get step direction
                #
                if LINEAR_SYSTEM_SOLVER_METHOD == SQPSolverMethods.N: # standard backslash
                    print("solveKKT\n")
                    dxul = self.solveKKTSystem(x, u, xs, N, dt, rho, options_linSys)
                    # print("dxul", dxul) #soln

                elif LINEAR_SYSTEM_SOLVER_METHOD == SQPSolverMethods.S: # schur complement backslash
                    print("Solve KKTSHUR")
                    dxul = self.solveKKTSystem_Schur(x, u, xs, N, dt, rho, False, options_linSys)
                 
                elif USING_PCG: # PCG
                    dxul = self.solveKKTSystem_Schur(x, u, xs, N, dt, rho, True, options_linSys)
                    print(f"dxul: {dxul}\n")
                
                else:
                    print("Valid QP Solver options are:\n", \
                          "N      : Standard Backslash\n", \
                          "S      : Schur Complement Backslash\n", \
                          "PCG-X  : PCG with Preconditioner X (see PCG for valid preconditioners)\n")
                    print("If calling from SQP the solver must be called QP-X where X is a solver option above.")
                    exit()

                if USING_PCG and options['RETURN_TRACE_linSys']:
                    inner_trace = dxul[1][1]
                    dxul = dxul[0]
                    inner_iters = len(inner_trace)
                else:
                    inner_iters = 1

                #
                # Do line search and accept iterate or regularize the problem
                #
                alpha = 1
                error = False
                while 1:
                    #
                    # Apply the update
                    #
                    x_new = copy.deepcopy(x)
                    u_new = copy.deepcopy(u)

                    for k in range(N):
                        x_new[:,k] -= alpha*dxul[n*k : n*k+nx, 0]
                        if k < N-1:
                            u_new[:,k] -= alpha*dxul[n*k+nx : n*(k+1), 0]
                    
                    #
                    # Compute the cost, constraint violation, and directional derivative
                    #
                    J_new = self.totalCost(x_new, u_new, N)
                    c_new = self.totalHardConstraintViolation(x_new, u_new, xs, N, dt)
                    
                ###
                    
                    #
                    # Directional derivative = grad_J*p - mu|c|
                    #
                    D = 0 
                    for k in range(N-1):
                        D += np.dot(self.cost.gradient(x_new[:,k], u_new[:,k], k), dxul[n*k : n*(k+1), 0])
                        # Add soft constraints if applicable
                        if self.other_constraints.total_soft_constraints(timestep = k) > 0:
                            D += np.dot(self.other_constraints.jacobian_soft_constraints(x_new[:,k], u_new[:,k], k)[:,0], dxul[n*k : n*(k+1), 0])
                    D += np.dot(self.cost.gradient(x_new[:,N-1], timestep = N-1), dxul[n*(N-1) : n*(N-1)+nx, 0])
                    # Add soft constraints if applicable
                    if self.other_constraints.total_soft_constraints(timestep = N-1) > 0:
                        D += np.dot(self.other_constraints.jacobian_soft_constraints(x_new[:,N-1], timestep = N-1)[:,0], dxul[n*(N-1) : n*(N-1)+nx, 0])
                    
                    #
                    # Compute totals for line search test
                    #
                    merit_new = J_new + mu * c_new
                    delta_J = J - J_new
                    delta_c = c - c_new
                    delta_merit = merit -  merit_new
                    expected_reduction = alpha * (D - mu * c_new)
                    reduction_ratio = delta_merit/expected_reduction
                    
                    #
                    # If succeeded accept new trajectory according to Nocedal and Wright 18.3
                    #
                    if (delta_merit >= 0 and reduction_ratio >= options['expected_reduction_min_SQP_DDP'] and \
                                             reduction_ratio <= options['expected_reduction_max_SQP_DDP']):
                        x = x_new
                        u = u_new
                        J = J_new
                        c = c_new
                        merit = merit_new
                        if options['DEBUG_MODE_SQP_DDP']:
                            print("Iter[", iteration, "] Cost[", J_new, "], Constraint Violation[", c_new, "], mu [", mu, "], Merit Function[", merit_new, "] and Reduction Ratio[", reduction_ratio, "] and rho [", rho, "]")

                        # update regularization
                        rho, drho = self.reduce_regularization(rho, drho, options)
                        # Check feasability gain vs. optimality gain and adjust mu accordingly
                        # if delta_J/J > delta_c/c:
                        #     mu = min(mu * merit_factor_SQP, 1000)
                        # else:
                        #     mu = max(mu / merit_factor_SQP, 1)
                        # merit = J + mu * c
                        if options['DEBUG_MODE_SQP_DDP']:
                            print("      updated merit: ", merit, " <<< delta J vs c: ", delta_J, " ", delta_c)
                        if options['RETURN_TRACE_SQP']:
                            trace.append([J,c,int(-np.log(alpha)/np.log(2))+1,inner_iters])
                        # end line search
                        break
                    
                    #
                    # If failed iterate decrease alpha and try line search again
                    #
                    elif alpha > options['alpha_min_SQP_DDP']:
                        if options['DEBUG_MODE_SQP_DDP']:
                            print("Alpha[", alpha, "] Rejected with Cost[", J_new, "], Constraint Violation[", c_new, "], mu [", mu, "], Merit Function[", merit_new, "] and Reduction Ratio[", reduction_ratio, "]")
                        alpha *= options['alpha_factor_SQP_DDP']
                    
                    #
                    # If failed the whole line search report the error
                    #
                    else:
                        error = True
                        if options['DEBUG_MODE_SQP_DDP']:
                            print("Line search failed")
                        if options['RETURN_TRACE_SQP']:
                            trace.append([J,c,-1,inner_iters])
                        break
                #
                # Check for exit (or error) and adjust accordingly
                #
                exit_flag, iteration, rho, drho = self.check_for_exit_or_error(error, delta_J, iteration, rho, drho, options)
                if exit_flag:
                    break

            #
            # Outer loop updates of soft constraint hyperparameters (where appropriate)
            #
            exit_flag, soft_constraint_iteration = self.check_and_update_soft_constraints(x, u, soft_constraint_iteration, options)
            if exit_flag:
                break

        if options['RETURN_TRACE_SQP']:
            return x, u, trace   
        
        return x, u

    def next_iteration_setup(self, x, u, dt, N, A, B, H, g, J, fxx = None, fux = None):
        nx = self.plant.get_num_pos() + self.plant.get_num_vel()
        #
        # compute new gradients
        #
        for k in range(N-1):
            A[:,:,k], B[:,:,k] = self.plant.integrator(x[:,k], u[:,k], dt, return_gradient = True)
            H[:,:,k] = self.cost.hessian(x[:,k], u[:,k], k)
            g[:,k] = self.cost.gradient(x[:,k], u[:,k], k)
            if (fxx is not None) and (fux is not None):
                fxx[:,:,:,k], fux[:,:,:,k] = self.plant.integrator(x[:,k], u[:,k], dt, return_hessian = True)
        H[0:nx,0:nx,N-1] = self.cost.hessian(x[:,N-1], timestep = N-1)
        g[0:nx,N-1] = self.cost.gradient(x[:,N-1], timestep = N-1)
        #
        # add soft constraints (if applicable)
        #
        for k in range(N-1):
            if self.other_constraints.total_soft_constraints(timestep = k) > 0:
                J += self.other_constraints.value_soft_constraints(x[:,k], u[:,k], k)
                gck = self.other_constraints.jacobian_soft_constraints(x[:,k], u[:,k], k)
                g[:,k] += gck[:,0]
                H[:,:,k] += np.outer(gck,gck)
        if self.other_constraints.total_soft_constraints(timestep = N-1) > 0:
            J += self.other_constraints.value_soft_constraints(x[:,N-1], timestep = N-1)
            gcNm1 = self.other_constraints.jacobian_soft_constraints(x[:,N-1], timestep = N-1)
            g[0:nx,N-1] += gcNm1[:,0]
            H[0:nx,0:nx,N-1] += np.outer(gcNm1,gcNm1)
        return J

    def backward_pass(self, K_new, du_new, A, B, H, g, rho, N, options, fxx = None, fux = None):
        error = False
        delta_J_expected_1 = 0
        delta_J_expected_2 = 0   
        nx = self.plant.get_num_pos() + self.plant.get_num_vel()
        #
        # Initialize CTG
        #
        P = H[0:nx,0:nx,N-1]
        p = g[0:nx,N-1]
        for k in range(N-2,-1,-1):
            #
            # Backpropogate CTG
            #
            Hxxk = np.matmul(A[:,:,k].transpose(),np.matmul(P,A[:,:,k])) + H[0:nx,0:nx,k]
            gxk = np.matmul(A[:,:,k].transpose(),p) + g[0:nx,k]
            guk = np.matmul(B[:,:,k].transpose(),p) + g[nx:,k]
            if options['state_regularization_DDP']:
                Preg = P + rho*np.eye(nx)
                Huuk = np.matmul(B[:,:,k].transpose(),np.matmul(Preg,B[:,:,k])) + H[nx:,nx:,k]
                Huxk = np.matmul(B[:,:,k].transpose(),np.matmul(Preg,A[:,:,k])) + H[nx:,0:nx,k]
            else:
                Huuk = np.matmul(B[:,:,k].transpose(),np.matmul(P,B[:,:,k])) + H[nx:,nx:,k] + rho*np.eye(nu)
                Huxk = np.matmul(B[:,:,k].transpose(),np.matmul(P,A[:,:,k])) + H[nx:,0:nx,k]
            #
            # Add Hessians (optional)
            if (fxx is not None) and (fux is not None):
                Hxxk += np.tensordot(p, fxx[:,:,:,k], axes=1)
                Huxk += np.tensordot(p, fux[:,:,:,k], axes=1)
                # FYI the tensordot with axes=1 is the same as the below
                # def tensordot_axes1(vec,tens):
                #     output = np.zeros((tens.shape[1],tens.shape[2]))
                #     for tensId in range(tens.shape[2]):
                #         for colId in range(tens.shape[1]):
                #             output[colId,tensId] = np.dot(p,tens[:,colId,tensId])
                #     return output
            #
            #
            # Invert Huu block
            #
            try:
                HuukInv = np.linalg.inv(Huuk)
            except np.linalg.LinAlgError:
                if options['DEBUG_MODE_SQP_DDP']:
                    print("Error in backward pass!")
                error = True
                break
            #
            # Compute feedback and next CTG as well as expected cost
            #
            K_new[:,:,k] = -np.matmul(HuukInv,Huxk)
            du_new[:,k] = -np.matmul(HuukInv,guk[:])
            P = Hxxk + np.matmul(K_new[:,:,k].transpose(),np.matmul(Huuk,K_new[:,:,k])) + np.matmul(K_new[:,:,k].transpose(),Huxk) + np.matmul(Huxk.transpose(),K_new[:,:,k])
            p = gxk + np.matmul(K_new[:,:,k].transpose(),np.matmul(Huuk,du_new[:,k])) + np.matmul(K_new[:,:,k].transpose(),guk) + np.matmul(Huxk.transpose(),du_new[:,k])
            delta_J_expected_1 += np.matmul(du_new[:,k].transpose(),guk)
            delta_J_expected_2 +=  np.matmul(du_new[:,k].transpose(),np.matmul(Huuk,du_new[:,k]))

        return error, delta_J_expected_1, delta_J_expected_2
    
    def forward_pass(self, x, u, K, du, J, dt, N, rho, drho, delta_J_expected_1, delta_J_expected_2, options):
        error = False
        alpha = 1
        while 1:
            #
            # Simulate forward
            #
            x_new = copy.deepcopy(x)
            u_new = copy.deepcopy(u)
            J_new = 0
            for k in range(N-1):
                u_new_k = u[:,k] + alpha*du[:,k] + np.matmul(K[:,:,k],(x_new[:,k] - x[:,k]))
                u_new[:,k:k+1] = np.reshape(u_new_k, (u_new_k.shape[0],1))
                
                x_new_k = self.plant.integrator(x_new[:,k], u_new[:,k], dt)
                x_new[:,k+1:k+2] = np.reshape(x_new_k, (x_new_k.shape[0],1))
                
                J_new += self.cost.value(x_new[:,k], u_new[:,k], k)
            J_new += self.cost.value(x_new[:,N-1], timestep = N-1)
            #
            # Add soft constraints if applicable
            #
            if self.other_constraints.total_soft_constraints() > 0:
                for k in range(N-1):
                    J_new += self.other_constraints.value_soft_constraints(x_new[:,k], u_new[:,k], k)
                J_new += self.other_constraints.value_soft_constraints(x_new[:,N-1], timestep = N-1)
            #
            # Compute change in cost
            #
            delta_J = J - J_new
            delta_J_expected = -alpha*delta_J_expected_1 + 0.5*alpha*alpha*delta_J_expected_2
            if not delta_J_expected == 0:
                delta_J_ratio = delta_J / delta_J_expected
            else:
                delta_J_ratio = options["expected_reduction_min_SQP_DDP"]
            #
            # If succeeded accept new trajectory
            #
            if delta_J >= 0 and delta_J_ratio >= options['expected_reduction_min_SQP_DDP'] \
                            and delta_J_ratio <= options['expected_reduction_max_SQP_DDP']:
                x = x_new
                u = u_new
                J = J_new
                rho, drho = self.reduce_regularization(rho, drho, options)
                if options['DEBUG_MODE_SQP_DDP']:
                    print("Iteration[", iteration, "] with cost[", J, "] and delta_J_ratio[", delta_J_ratio, "]")
                    print("x final:")
                    print(x[:,N-1])
                break
            #
            # If failed decrease alpha and try line search again
            #
            elif alpha > options['alpha_min_SQP_DDP']:
                alpha *= options['alpha_factor_SQP_DDP']
                if options['DEBUG_MODE_SQP_DDP']:
                    print("Rejected with delta_J[", delta_J, "] and delta_J_ratio[", delta_J_ratio, "]")
            #
            # If failed the whole line search report the error
            #
            else:
                error = True
                if options['DEBUG_MODE_SQP_DDP']:
                    print("Line search failed")
                break
        return error, x, u, J, delta_J, rho, drho

    def DDP(self, x: np.ndarray, u: np.ndarray, N: int, dt: float, options):
        options["DDP_flag"] = True
        return self.iLQR(x, u, N, dt, options)

    def iLQR(self, x: np.ndarray, u: np.ndarray, N: int, dt: float, options):
        self.set_default_options(options)

        nq = self.plant.get_num_pos()
        nv = self.plant.get_num_vel()
        nu = self.plant.get_num_cntrl()
        nx = nq + nv

        # Start the main loops (soft constraint outer loop)
        soft_constraint_iteration = 0
        while 1:

            #
            # compute initial cost and gradients and placeholders
            #
            rho = options['rho_init_SQP_DDP']
            drho = 1
            J = 0
            iteration = 0
            du = np.zeros((nu,N-1))
            K = np.zeros((nu,nq+nv,N-1))
            H = np.zeros((nq+nv+nu,nq+nv+nu,N))
            g = np.zeros((nq+nv+nu,N))
            A = np.zeros((nq+nv,nq+nv,N-1))
            B = np.zeros((nq+nv,nu,N-1))
            fxx = None
            fux = None
            if options["DDP_flag"]:
                fxx = np.zeros((nq+nv,nq+nv,nq+nv,N-1))
                fux = np.zeros((nq+nv,nu,nq+nv,N-1))
            # get initial cost
            for k in range(N-1):
                J += self.cost.value(x[:,k], u[:,k], k)
            J += self.cost.value(x[:,N-1], timestep = N-1)
            # get initial gradients and apply soft constraints (if applicable)
            J = self.next_iteration_setup(x, u, dt, N, A, B, H, g, J, fxx, fux)
            delta_J = J

            if options['DEBUG_MODE_SQP_DDP']:
                print("Initial Cost: ", J)

            # start the main loop
            while 1:
                
                #
                # Do backwards pass to compute du and K and expected cost reduction
                #
                K_new = np.zeros(K.shape)
                du_new = np.zeros(du.shape)
                error, delta_J_expected_1_new, delta_J_expected_2_new = self.backward_pass(K_new, du_new, A, B, H, g, rho, N, options, fxx, fux)
                if not error:
                    K = K_new
                    du = du_new
                    delta_J_expected_1 = delta_J_expected_1_new
                    delta_J_expected_2 = delta_J_expected_2_new
                
                #
                # Do forwards pass to compute new x, u, J (with line search)
                #
                if not error:
                    error, x, u, J, delta_J, rho, drho = self.forward_pass(x, u, K, du, J, dt, N, rho, drho, delta_J_expected_1, delta_J_expected_2, options)
                    
                #
                # Check for exit (or error) and adjust accordingly
                #
                exit_flag, iteration, rho, drho = self.check_for_exit_or_error(error, delta_J, iteration, rho, drho, options)
                if exit_flag:
                    break
                #
                # If doing new loop compute new gradients and add soft constraints (if applicable)
                #
                J = self.next_iteration_setup(x, u, dt, N, A, B, H, g, J, fxx, fux)
            
            #
            # Outer loop updates of soft constraint hyperparameters (where appropriate)
            #
            exit_flag, soft_constraint_iteration = self.check_and_update_soft_constraints(x, u, soft_constraint_iteration, options)
            if exit_flag:
                break

        if options['DEBUG_MODE_SQP_DDP']:
            print("Final Trajectory")
            print(x)
            print(u)
        return x, u, K

    def LQR_tracking(self, x: np.ndarray, u: np.ndarray, xs: np.ndarray, N: int, dt: float):
        nq = self.plant.get_num_pos()
        nv = self.plant.get_num_vel()
        nu = self.plant.get_num_cntrl()
        nx = nq + nv
        n = nx + nu
        K = np.zeros((nu,nx,N-1))

        P = self.cost.hessian(x[:,N-1], timestep = N-1)
        for k in range(N-2,-1,-1):
            H = self.cost.hessian(x[:,k], u[:,k], k)
            Q = H[0:nx, 0:nx]
            R = H[nx:n, nx:n]
            A, B = self.plant.integrator(x[:,k], u[:,k], dt, return_gradient = True)
            
            PA = np.matmul(P,A)
            PB = np.matmul(P,B)
            ATPA = np.matmul(A.transpose(),PA)
            ATPB = np.matmul(A.transpose(),PB)
            BTPB = np.matmul(B.transpose(),PB)  
            BTPA = np.matmul(B.transpose(),PA)
            invTerm = np.linalg.inv(R + BTPB)
            
            Kterm = np.matmul(invTerm,BTPA)
            K[:,:,k] = -Kterm

            P = Q + ATPA - np.matmul(ATPB,Kterm)

        return K


    def integrate_and_shift_trajectory(self, x: np.ndarray, u: np.ndarray, K: int, xs: np.ndarray, dt: float, options = {}):
        self.set_default_options(options)
        # compute new start state
        xs_new = copy.deepcopy(xs)
        adj = 0
        for k in range(options['num_timesteps_per_solve_mpc']):
            for i in range(options['simulator_steps_mpc']):
                if not isinstance(K, type(None)):
                    x_err = xs_new - x[:,k]
                    adj = np.matmul(K[:,:,k],x_err)
                u_new = u[:,k] + adj
                xs_new = self.plant.integrator(xs_new, u_new, dt/options['simulator_steps_mpc'], 0)
        xs = copy.deepcopy(xs_new)

        # shift trajectory
        x[:,:-options['num_timesteps_per_solve_mpc']] = x[:,options['num_timesteps_per_solve_mpc']:]
        u[:,:-options['num_timesteps_per_solve_mpc']] = u[:,options['num_timesteps_per_solve_mpc']:]

        # Copy in new start state
        x[:,0] = copy.deepcopy(xs)

        # Fill end
        copy_step = -options['num_timesteps_per_solve_mpc'] - 1
        for step in range(options['num_timesteps_per_solve_mpc']):
            load_step = -step - 1 # starts at 0 but we need to start at -1
            integrate_from_step = load_step - 1
            u[:,load_step] = u[:,copy_step]
            x[:,load_step] = self.plant.integrator(x[:,integrate_from_step], u[:,load_step], dt, 0)

        # Finally shift soft constraint constants if applicable and fill end with zeros
        self.other_constraints.shift_soft_constraint_constants(options['num_timesteps_per_solve_mpc'])

        return x, u, xs


    def MPC(self, x: np.ndarray, u: np.ndarray, N: int, dt: float, SOLVER_METHOD = MPCSolverMethods.QP_N, options = {}):
        self.set_default_options(options)

        last_err = float('inf')
        xs = copy.deepcopy(x[:,0])
        x_trace = np.reshape(xs,(xs.shape[0],1))
        u_trace = None
        if options['DEBUG_MODE_SQP_DDP']:
            print(xs)

        while True:

            if SOLVER_METHOD in [MPCSolverMethods.QP_N, MPCSolverMethods.QP_S, MPCSolverMethods.QP_PCG_J, MPCSolverMethods.QP_PCG_BJ, MPCSolverMethods.QP_PCG_SS]:
                options['max_iter_SQP_DDP'] = 5
                sqp_solver = SQPSolverMethods(SOLVER_METHOD.value[3:])
                x, u = self.SQP(x, u, N, dt, sqp_solver, options)
                K = self.LQR_tracking(x, u, xs, N, dt)
            elif SOLVER_METHOD == MPCSolverMethods.iLQR:
                x, u, K = self.iLQR(x, u, N, dt, options)
            elif SOLVER_METHOD == MPCSolverMethods.DDP:
                x, u, K = self.DDP(x, u, N, dt, options)
            else:
                print("Invalid solver options are:\n", \
                          "iLQR      : Iterated Linear Quadratic Regulator\n", \
                          "DDP      : Differential Dynamic Programming\n", \
                          "QP-N      : SQP with Standard Backslash\n", \
                          "QP-S      : SQP with Schur Complement Backslash\n", \
                          "QP-PCG-0  : SQP with PCG with no preconditioner\n", \
                          "QP-PCG-J  : SQP with PCG with Jacobi preconditioner\n", \
                          "QP-PCG-BJ : SQP with PCG with Block-Jacobi preconditioner\n", \
                          "QP-PCG-SS : SQP with PCG with Stair preconditioner\n")
                exit()

            if isinstance(u_trace, type(None)):
                u_trace = np.reshape(u[:,0],(u.shape[0],1))
            else:
                u_trace = np.append(u_trace,np.reshape(u[:,0],(u.shape[0],1)),1)

            x, u, xs = self.integrate_and_shift_trajectory(x, u, K, xs, dt, options)

            x_trace = np.append(x_trace,np.reshape(xs,(xs.shape[0],1)),1)
            if options['DEBUG_MODE_SQP_DDP'] or 1:
                print(xs)

            err = xs - self.cost.xg
            err = np.dot(err,err)
            delta_err = abs(last_err - err)
            
            if delta_err < 1e-4:
                break
        
            last_err = err

        if options['print_full_trajectory_DDP']:
            print(x_trace)
            print(u_trace)

        return x_trace, u_trace