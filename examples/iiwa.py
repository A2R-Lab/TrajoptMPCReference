#!/usr/bin/python3
from exampleHelpers import *

sqp_solver_methods = [SQPSolverMethods.N]#["N", "S", "PCG-J", "PCG-BJ", "PCG-SS"]
mpc_solver_methods = [MPCSolverMethods.DDP] #["iLQR", "QP-N", "QP-S", "QP-PCG-J", "QP-PCG-BJ", "QP-PCG-SS"]

options = {}
options["path_to_urdf"] = "iiwa.urdf"
plant = URDFPlant(options = options)

N = 20
dt = 0.1

Q = np.diag([1.0,1.0,1.0,1.0,1.0,1.0,1.0,0.1,0.1,0.1,0.1,0.1,0.1,0.1])
QF = np.diag([100.0,100.0,100.0,100.0,100.0,100.0,100.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0])
R = np.diag([0.01,0.01,0.01,0.01,0.01,0.01,0.01])
xg = np.array([0,0,0,-0.25*3.14,0,0.25*3.14,0.5*3.14,0,0,0,0,0,0,0])
cost = QuadraticCost(Q,QF,R,xg)

x0 = np.zeros((plant.get_num_pos() + plant.get_num_vel(), N))
u0 = np.zeros((plant.get_num_cntrl(), N))
x0[0,:] = -0.5*3.14
x0[1,:] = 0.25*3.14
x0[2,:] = 0.167*3.14
x0[3,:] = -0.167*3.14
x0[4,:] = 0.125*3.14
x0[5,:] = 0.167*3.14
x0[6,:] = 0.15*3.14
u0[0,:] = 0
u0[1,:] = -102.9832
u0[2,:] = 11.1968
u0[3,:] = 47.0724
u0[4,:] = 2.5993
u0[5,:] = -7.0290
u0[6,:] = -0.0907

hard_constraints = None #TrajoptConstraint(plant.get_num_pos(),plant.get_num_vel(),plant.get_num_cntrl(),N)
# hard_constraints.set_torque_limits([7.0],[-7.0],"ACTIVE_SET")

soft_constraints = None #TrajoptConstraint(plant.get_num_pos(),plant.get_num_vel(),plant.get_num_cntrl(),N)
# soft_constraints.set_torque_limits([7.0],[-7.0],"AUGMENTED_LAGRANGIAN")

options = {
    "expected_reduction_min_SQP_DDP":-100, # needed for hard_constraints - TODO debug why
    "display": False
}

# runSQPExample(plant, cost, hard_constraints, soft_constraints, N, dt, sqp_solver_methods, options)

# options = {
#     "expected_reduction_min_SQP_DDP":-100, # needed for hard_constraints - TODO debug why
#     "display": False
# }

runMPCExample(plant, cost, hard_constraints, soft_constraints, N, dt, mpc_solver_methods, options = options)