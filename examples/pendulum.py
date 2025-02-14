#!/usr/bin/python3
from exampleHelpers import *

sqp_solver_methods = [SQPSolverMethods.N, SQPSolverMethods.B]#["N", "S","B" "PCG-J", "PCG-BJ", "PCG-SS"]
mpc_solver_methods = [MPCSolverMethods.DDP] #["iLQR", "QP-N", "QP-S", "QP-PCG-J", "QP-PCG-BJ", "QP-PCG-SS"]

plant = PendulumPlant()

#Yana's changes for the BCHOL must be a power of 2
N = 8
dt = 0.1

Q = np.diag([3.0,4.0])
QF = np.diag([100.0,100.0])
R = np.diag([0.5])
xg = np.array([3.14159,0])
cost = QuadraticCost(Q,QF,R,xg)


x0_custom = np.zeros((2, 8))
#x0_custom[0, :] = 0.5   # e.g., set all angles to 0.5 rad
u0_custom = np.zeros((1, 7))
#u0_custom[0, :] = 0.1   # e.g., set all controls to 0.1

hard_constraints = None #TrajoptConstraint(plant.get_num_pos(),plant.get_num_vel(),plant.get_num_cntrl(),N)
# hard_constraints.set_torque_limits([7.0],[-7.0],"ACTIVE_SET")

soft_constraints = None #TrajoptConstraint(plant.get_num_pos(),plant.get_num_vel(),plant.get_num_cntrl(),N)
# soft_constraints.set_torque_limits([7.0],[-7.0],"AUGMENTED_LAGRANGIAN")

options = {
    "expected_reduction_min_SQP_DDP":-100, # needed for hard_constraints - TODO debug why
    "display": False
}
#Is there a difference what to run?

runSQPExample(plant, cost, hard_constraints, soft_constraints,
              N, dt, sqp_solver_methods, options,
              x0=x0_custom, u0=u0_custom)
# options = {
#     "expected_reduction_min_SQP_DDP":-100, # needed for hard_constraints - TODO debug why
#     "display": False
# }

# runMPCExample(plant, cost, hard_constraints, soft_constraints, N, dt, mpc_solver_methods, options = options)


if __name__ == "__main__":

    # the user wants to run with all three
    sqp_solver_methods = [SQPSolverMethods.N, SQPSolverMethods.B]

    plant = PendulumPlant()
    N = 8
    dt = 0.1

    Q = np.diag([3.0,4.0])
    QF = np.diag([100.0,100.0])
    R = np.diag([0.5])
    xg = np.array([3.14159,0])
    cost = QuadraticCost(Q,QF,R,xg)

    # no constraints
    hard_constraints = None
    soft_constraints = None

    options = {
        "expected_reduction_min_SQP_DDP": -100,
        "display": False,
        # optionally:
        "DEBUG_MODE_SQP_DDP": True
    }

    # run the example
    results_unconstrained = runSQPExample(
        plant, cost, hard_constraints, soft_constraints,
        N, dt, sqp_solver_methods, options
    )

    '''
    # Suppose runSQPExample returns a dict of results. Let's do:
    if isinstance(results_unconstrained, dict):
        # Flatten and compare
        if "N" in results_unconstrained and "BCHOL" in results_unconstrained and "S" in results_unconstrained:
            xN, uN, cN = results_unconstrained["N"]
            xB, uB, cB = results_unconstrained["BCHOL"]
            xS, uS, cS = results_unconstrained["S"]

            # flatten
            fN = np.concatenate([xN.flatten(), uN.flatten()])
            fB = np.concatenate([xB.flatten(), uB.flatten()])
            fS = np.concatenate([xS.flatten(), uS.flatten()])

            print("Diff (N-B) = ", np.linalg.norm(fN - fB))
            print("Diff (N-S) = ", np.linalg.norm(fN - fS))
            print("Diff (B-S) = ", np.linalg.norm(fB - fS))
    '''