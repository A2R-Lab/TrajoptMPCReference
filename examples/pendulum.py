#!/usr/bin/python3
from exampleHelpers import *

sqp_solver_methods = [SQPSolverMethods.N]#["N", "S", "PCG-J", "PCG-BJ", "PCG-SS"]
mpc_solver_methods = [MPCSolverMethods.QP_PCG_SS] #["iLQR", "QP-N", "QP-S", "QP-PCG-J", "QP-PCG-BJ", "QP-PCG-SS"]

plant = PendulumPlant()

N = 20
dt = 0.1

Q = np.diag([3.0,4.0])
QF = np.diag([100.0,100.0])
R = np.diag([0.1])
xg = np.array([3.14159,0])
cost = QuadraticCost(Q,QF,R,xg)

hard_constraints = TrajoptConstraint(plant.get_num_pos(),plant.get_num_vel(),plant.get_num_cntrl(),N)
hard_constraints.set_torque_limits([7.0],[-7.0],"ACTIVE_SET")

soft_constraints = TrajoptConstraint(plant.get_num_pos(),plant.get_num_vel(),plant.get_num_cntrl(),N)
soft_constraints.set_torque_limits([7.0],[-7.0],"AUGMENTED_LAGRANGIAN")

options = {
    "expected_reduction_min_SQP_DDP":-100, # needed for hard_constraints - TODO debug why
    "display": True
}

runSQPExample(plant, cost, hard_constraints, soft_constraints, N, dt, sqp_solver_methods, options)

options = {
    "expected_reduction_min_SQP_DDP":-100, # needed for hard_constraints - TODO debug why
    "display": False
}

runMPCExample(plant, cost, hard_constraints, soft_constraints, N, dt, mpc_solver_methods, options)