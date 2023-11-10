import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TrajoptPlant import TrajoptPlant, DoubleIntegratorPlant, PendulumPlant, CartPolePlant, URDFPlant
from TrajoptCost import TrajoptCost, QuadraticCost
from TrajoptConstraint import TrajoptConstraint, BoxConstraint
from TrajoptMPCReference import TrajoptMPCReference, SQPSolverMethods, MPCSolverMethods

import numpy as np
import copy

def runSolversSQP(trajoptMPCReference: TrajoptMPCReference, N: int, dt: float, solver_methods: list[SQPSolverMethods], options = {}):
	for solver in solver_methods:
		print("-----------------------------")
		print("Solving with method: ", solver)

		nq = trajoptMPCReference.plant.get_num_pos()
		nv = trajoptMPCReference.plant.get_num_vel()
		nx = nq + nv
		nu = trajoptMPCReference.plant.get_num_cntrl()
		x = np.zeros((nx,N))
		u = np.zeros((nu,N-1))
		xs = copy.deepcopy(x[:,0])

		x, u = trajoptMPCReference.SQP(x, u, N, dt, LINEAR_SYSTEM_SOLVER_METHOD = solver, options = options)

		print("Final State Trajectory")
		print(x)
		print("Final Control Trajectory")
		print(u)
		J = 0
		for k in range(N-1):
			J += trajoptMPCReference.cost.value(x[:,k], u[:,k])
		J += trajoptMPCReference.cost.value(x[:,N-1], None)
		print("Cost [", J, "]")
		print("Final State Error vs. Goal")
		print(x[:,-1] - trajoptMPCReference.cost.xg)

def runSQPExample(plant, cost, hard_constraints, soft_constraints, N, dt, solver_methods, options = {}):
	print("-----------------------------")
	print("-----------------------------")
	print("    Running SQP Example      ")
	print("-----------------------------")
	print("-----------------------------")
	print("Solving Unconstrained Problem")
	print("-----------------------------")
	trajoptMPCReference = TrajoptMPCReference(plant, cost)
	runSolversSQP(trajoptMPCReference, N, dt, solver_methods, options)

	print("---------------------------------")
	print("---------------------------------")
	print(" Solving Constrained Problem Hard")
	print("---------------------------------")
	trajoptMPCReference = TrajoptMPCReference(plant, cost, hard_constraints)
	runSolversSQP(trajoptMPCReference, N, dt, solver_methods, options)

	print("---------------------------------")
	print("---------------------------------")
	print(" Solving Constrained Problem Soft")
	print("---------------------------------")
	trajoptMPCReference = TrajoptMPCReference(plant, cost, soft_constraints)
	runSolversSQP(trajoptMPCReference, N, dt, solver_methods, options)

def runSolversMPC(trajoptMPCReference, N, dt, solver_methods, options = {}):
	for solver in solver_methods:
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
		print(trajoptMPCReference.cost.xg)
		print("-------------")

		trajoptMPCReference.MPC(x, u, N, dt, SOLVER_METHOD = solver)


def runMPCExample(plant, cost, hard_constraints, soft_constraints, N, dt, solver_methods, options = {}):
	print("-----------------------------")
	print("-----------------------------")
	print("    Running MPC Example      ")
	print("-----------------------------")
	print("-----------------------------")
	print("Solving Unconstrained Problem")
	print("-----------------------------")
	trajoptMPCReference = TrajoptMPCReference(plant, cost)
	runSolversMPC(trajoptMPCReference, N, dt, solver_methods)

	print("---------------------------------")
	print("---------------------------------")
	print(" Solving Constrained Problem Hard")
	print("---------------------------------")
	trajoptMPCReference = TrajoptMPCReference(plant, cost, hard_constraints)
	runSolversMPC(trajoptMPCReference, N, dt, solver_methods, options)

	print("---------------------------------")
	print("---------------------------------")
	print(" Solving Constrained Problem Soft")
	print("---------------------------------")
	trajoptMPCReference = TrajoptMPCReference(plant, cost, soft_constraints)
	runSolversMPC(trajoptMPCReference, N, dt, solver_methods, options)