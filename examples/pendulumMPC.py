#!/usr/bin/python3
import numpy as np
import copy
from context import *

SOLVER_METHODS = ["QP-PCG-SS"] #["iLQR", "QP-N", "QP-S", "QP-PCG-J", "QP-PCG-BJ", "QP-PCG-SS"]

plant = PendulumPlant()

N = 20
dt = 0.1

Q = np.diag([1.0,1.0])
QF = np.diag([100.0,100.0])
R = np.diag([0.1])
xg = np.array([3.14159,0])
cost = QuadraticCost(Q,QF,R,xg)

def runSolvers(trajoptMPCReference, options = {}):
	for solver in SOLVER_METHODS:
		print("-----------------------------")
		print("Solving with method: ", solver)

		nq = trajoptMPCReference.plant.get_num_pos()
		nv = trajoptMPCReference.plant.get_num_vel()
		nx = nq + nv
		nu = trajoptMPCReference.plant.get_num_cntrl()
		x = np.zeros((nx,N))
		u = np.zeros((nu,N-1))
		xs = copy.deepcopy(x[:,0])

		print("Goal Position")
		print(xg)
		print("-------------")

		trajoptMPCReference.MPC(x, u, N, dt, SOLVER_METHOD = solver)

print("-----------------------------")
print("-----------------------------")
print("Solving Unconstrained Problem")
print("-----------------------------")
trajoptMPCReference = TrajoptMPCReference(plant, cost)
runSolvers(trajoptMPCReference)

print("---------------------------------")
print("---------------------------------")
print(" Solving Constrained Problem Hard")
print("---------------------------------")
constraints = TrajoptConstraint(plant.get_num_pos(),plant.get_num_vel(),plant.get_num_cntrl(),N)
constraints.set_torque_limits([7.0],[-7.0],"ACTIVE_SET")
options = {"expected_reduction_min_SQP_DDP":-100}
trajoptMPCReference = TrajoptMPCReference(plant, cost, constraints)
runSolvers(trajoptMPCReference, options)

print("---------------------------------")
print("---------------------------------")
print(" Solving Constrained Problem Soft")
print("---------------------------------")
constraints.set_torque_limits([7.0],[-7.0],"AUGMENTED_LAGRANGIAN")
trajoptMPCReference = TrajoptMPCReference(plant, cost, constraints)
runSolvers(trajoptMPCReference, options)