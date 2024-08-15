import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TrajoptPlant import TrajoptPlant, DoubleIntegratorPlant, PendulumPlant, CartPolePlant, URDFPlant
from TrajoptCost import TrajoptCost, QuadraticCost
from TrajoptConstraint import TrajoptConstraint, BoxConstraint
from TrajoptMPCReference import TrajoptMPCReference, SQPSolverMethods, MPCSolverMethods

import matplotlib.pyplot as plt
import numpy as np
import copy

def display(x: np.ndarray, x_lim: list[float] = [-20, 20], y_lim: list[float] = [-20, 20], title: str = ""):
	fig = plt.figure()
	ax = fig.add_subplot(111)
	line1, = ax.plot([0, 5, 10], [0, 5, 10], 'b-')
	ax.set_xlim(x_lim)
	ax.set_ylim(y_lim)
	# set suptitle as title
	fig.suptitle(title)
	N = x.shape[1]

	for k in range(N):
		print("State at time step ", k, " is: ", x[:,k])
		# x[:,k] is the state at time step k
		# the first number is the angle of the first joint
		# the second number is the angle of the second joint
		# draw the line with a length of 5
		# add 90 degrees to the angle to make it point up
		first_point = [0, 0]
		second_point = [5*np.cos(x[0,k]-np.pi/2), 5*np.sin(x[0,k]-np.pi/2)]
		third_point = [second_point[0] + 5*np.cos(x[0,k]+x[1,k]-np.pi/2), second_point[1] + 5*np.sin(x[0,k]+x[1,k]-np.pi/2)]
		line1.set_xdata([first_point[0], second_point[0], third_point[0]])
		line1.set_ydata([first_point[1], second_point[1], third_point[1]])
		plt.title("Time Step: " + str(k))
		fig.canvas.draw()
		#fig.canvas.mpl_connect('close_event', _on_close)
		fig.canvas.flush_events()
		plt.pause(0.1)
	
	plt.show()


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

		if options["display"]:
			display(x, title="SQP Solver Method: " + solver.name)

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

	if (hard_constraints is not None):
		print("---------------------------------")
		print("---------------------------------")
		print(" Solving Constrained Problem Hard")
		print("---------------------------------")
		trajoptMPCReference = TrajoptMPCReference(plant, cost, hard_constraints)
		runSolversSQP(trajoptMPCReference, N, dt, solver_methods, options)

	if (soft_constraints is not None):
		print("---------------------------------")
		print("---------------------------------")
		print(" Solving Constrained Problem Soft")
		print("---------------------------------")
		trajoptMPCReference = TrajoptMPCReference(plant, cost, soft_constraints)
		runSolversSQP(trajoptMPCReference, N, dt, solver_methods, options)

def runSolversMPC(trajoptMPCReference, N, dt, solver_methods, x0 = None, u0 = None, options = {}):
	for solver in solver_methods:
		print("-----------------------------")
		print("Solving with method: ", solver)

		nq = trajoptMPCReference.plant.get_num_pos()
		nv = trajoptMPCReference.plant.get_num_vel()
		nx = nq + nv
		nu = trajoptMPCReference.plant.get_num_cntrl()
		x = x0
		u = u0
		if (x0 is None):
			x = np.zeros((nx,N))
		if (u0 is None):
			u = np.zeros((nu,N-1))
		xs = copy.deepcopy(x[:,0])

		print("Goal Position")
		print(trajoptMPCReference.cost.xg)
		print("-------------")

		x, u = trajoptMPCReference.MPC(x, u, N, dt, SOLVER_METHOD = solver)

		if options["display"]:
			display(x, title="MPC Solver Method: " + solver.name)


def runMPCExample(plant, cost, hard_constraints, soft_constraints, N, dt, solver_methods, x0 = None, u0 = None, options = {}):
	print("-----------------------------")
	print("-----------------------------")
	print("    Running MPC Example      ")
	print("-----------------------------")
	print("-----------------------------")
	print("Solving Unconstrained Problem")
	print("-----------------------------")
	trajoptMPCReference = TrajoptMPCReference(plant, cost)
	runSolversMPC(trajoptMPCReference, N, dt, solver_methods, x0, u0, options)

	if (hard_constraints is not None):
		print("---------------------------------")
		print("---------------------------------")
		print(" Solving Constrained Problem Hard")
		print("---------------------------------")
		trajoptMPCReference = TrajoptMPCReference(plant, cost, hard_constraints)
		runSolversMPC(trajoptMPCReference, N, dt, solver_methods, x0, u0, options)

	if (soft_constraints is not None):
		print("---------------------------------")
		print("---------------------------------")
		print(" Solving Constrained Problem Soft")
		print("---------------------------------")
		trajoptMPCReference = TrajoptMPCReference(plant, cost, soft_constraints)
		runSolversMPC(trajoptMPCReference, N, dt, solver_methods, x0, u0, options)