import numpy as np
import random
from GRiD import RBDReference
from GRiD import URDFParser

class TrajoptPlant:
	def __init__(self, integrator_type: int = 0, options = {}, need_path: bool = False):
		self.validate_integrator_type(integrator_type)
		self.integrator_type = integrator_type
		self.set_default_options(options, need_path)
		self.options = options

	def validate_integrator_type(self, integrator_type: int):
		if not (integrator_type in [0, 1, 2, 3, 4, -1]):
			print("Invalid integrator options are [0 : euler, 1 : semi-implicit euler, 2 : midpoint, 3 : rk3, 4 : rk4, -1 : hard-coded as dynamics")
			exit()

	def set_default_options(self, options: dict, need_path: bool = False):
		options.setdefault('path_to_urdf', None)
		options.setdefault('gravity', -9.81)
		if need_path and (not options.get('path_to_urdf')):
			print("You must include the 'path_to_urdf' in the options.")
			exit()

	##############################
	# Child class must implement #
	##############################

	def forward_dynamics(self):
		raise NotImplementedError

	def forward_dynamics_gradient(self):
		raise NotImplementedError

	def get_num_pos(self):
		raise NotImplementedError

	def get_num_vel(self):
		raise NotImplementedError

	def get_num_cntrl(self):
		raise NotImplementedError

	##############################
	# Child class must implement #
	##############################

	#  [ v ;
	#   qdd ]
	def qdd_to_xdot(self, xk: np.ndarray, qdd: np.ndarray):
		nq = self.get_num_pos()
		nv = self.get_num_vel()
		nu = self.get_num_cntrl()
		return np.vstack((xk[nq:], qdd)).flatten()

	# [ 0       ; eye     ; 0
	#   dqdd/dq ; dqdd/dv ; dqdd/du ]
	def dqdd_to_dxdot(self, dqdd: np.ndarray):
		nq = self.get_num_pos()
		nv = self.get_num_vel()
		m = self.get_num_cntrl()
		top = np.hstack((np.zeros((nq,nq)), np.eye(nv), np.zeros((nq,m))))
		return np.vstack((top, dqdd))

	def integrator(self, xk: np.ndarray, uk: np.ndarray, dt: float, return_gradient: bool = False):
		n = len(xk)

		if self.integrator_type == -1: # hard coded into model
			if not return_gradient:
				return self.integrator(xk,uk)
			else:
				return self.integrator_gradient(xk,uk)

		if self.integrator_type == 0: # euler
			#  xkp1 = xk + dt * [vk,qddk]
			# dxkp1 = [Ix | 0u ] + dt*[ 0q, Iv, 0u; dqdd]
			qdd = self.forward_dynamics(xk,uk)
			xdot = self.qdd_to_xdot(xk, qdd)
			xkp1 = xk + dt*xdot
			if not return_gradient:
				return xkp1 #np.reshape(xkp1, (xkp1.shape[0],1))[:,0]
			else:
				dqdd = self.forward_dynamics_gradient(xk,uk)
				dxdot = self.dqdd_to_dxdot(dqdd)
				A = np.eye(n) + dt*dxdot[:,0:n]
				B = dt*dxdot[:,n:]
				return A, B
		
		elif self.integrator_type == 1: # semi-implicit euler
			#  vkp1 = vk + dt*qddk
			#  qkp1 = qk  + dt*vkp1
			#  xkp1 = [qkp1; vkp1]
			# dxkp1 = [Ix | 0u ] + dt*[[0q, Iv, 0u] + dt*dqdd; dqdd]
			nq = self.get_num_pos()
			nv = self.get_num_vel()
			nu = self.get_num_cntrl()
			qdd = self.forward_dynamics(xk,uk)
			vkp1 = xk[nq:]  + dt*qdd
			qkp1 = xk[0:nq] + dt*vkp1
			if not return_gradient:
				return np.hstack((qkp1,vkp1)).transpose()
			else:
				dqdd = self.forward_dynamics_gradient(xk,uk)
				zIz = np.hstack((np.zeros((nq,nq)),np.eye(nq),np.zeros((nq,nu))))
				Iz = np.hstack((np.eye(nq+nv),np.zeros((nq+nv,nu))))
				AB = Iz + dt*np.vstack((zIz + dt*dqdd, dqdd))
				return AB[:,0:nq+nv], AB[:,nq+nv:]
		
		elif self.integrator_type == 2: # midpoint
			xdot1 = self.qdd_to_xdot(xk, self.forward_dynamics(xk,uk))
			midpoint = xk + 0.5*dt*xdot1
			xdot2 = self.qdd_to_xdot(xk, self.forward_dynamics(midpoint,uk))
			xkp1 = xk + dt*xdot2
			if not return_gradient:
				return xkp1
			else:
				dxdot1 = self.dqdd_to_dxdot(self.forward_dynamics_gradient(xk,uk))
				A1 = np.eye(n) + 0.5*dt*dxdot1[:,0:n]
				B1 = 0.5*dt*dxdot1[:,n:]
				
				dxdot2 = self.dqdd_to_dxdot(self.forward_dynamics_gradient(midpoint,uk))
				A2 = np.eye(n) + 0.5*dt*dxdot2[:,0:n]
				B2 = 0.5*dt*dxdot2[:,n:]

				A = np.matmul(A2,A1)
				B = np.matmul(A2,B1) + B2
				return A, B

		elif self.integrator_type == 3: # rk3
			xdot1 = self.qdd_to_xdot(xk, self.forward_dynamics(xk,uk))
			point1 = xk + 0.5*dt*xdot1
			xdot2 = self.qdd_to_xdot(xk, self.forward_dynamics(point1,uk))
			point2 = xk + 0.75*dt*xdot2
			xdot3 = self.qdd_to_xdot(xk, self.forward_dynamics(point2,uk))
			xkp1 = xk + (dt/9)*(2*xdot1 + 3*xdot2 + 4*xdot3)
			if not return_gradient:
				return xkp1
			else:
				dxdot1 = self.dqdd_to_dxdot(self.forward_dynamics_gradient(xk,uk))
				A1 = np.eye(n) + 2/9*dt*dxdot1[:,0:n]
				B1 = 2/9*dt*dxdot1[:,n:]

				dxdot2 = self.dqdd_to_dxdot(self.forward_dynamics_gradient(point1,uk))
				A2 = np.eye(n) + 1/3*dt*dxdot2[:,0:n]
				B2 = 1/3*dt*dxdot1[:,n:]                
				
				dxdot3 = self.dqdd_to_dxdot(self.forward_dynamics_gradient(point2,uk))
				A3 = np.eye(n) + 4/9*dt*dxdot3[:,0:n]
				B3 = 4/9*dt*dxdot1[:,n:]                
				
				A = np.matmul(A3,np.matmul(A2,A1))
				B = np.matmul(A3,np.matmul(A2,B1)) + np.matmul(A3,B2) + B3
				return A,B
		
		elif self.integrator_type == 4: # rk4
			xdot1 = self.qdd_to_xdot(xk, self.forward_dynamics(xk,uk))
			point1 = xk + 0.5*dt*xdot1
			xdot2 = self.qdd_to_xdot(xk, self.forward_dynamics(point1,uk))
			point2 = xk + 0.5*dt*xdot2
			xdot3 = self.qdd_to_xdot(xk, self.forward_dynamics(point2,uk))
			point3 = xk + dt*xdot3
			xdot4 = self.qdd_to_xdot(xk, self.forward_dynamics(point3,uk))
			xkp1 = xk + (dt/6)*(xdot1 + 2*xdot2 + 2*xdot3 + xdot4)
			if not return_gradient:
				return xkp1
			else:
				dxdot1 = self.dqdd_to_dxdot(self.forward_dynamics_gradient(xk,uk))
				A1 = np.eye(n) + 1/6*dt*dxdot1[:,0:n]
				B1 = 1/6*dt*dxdot1[:,n:]

				dxdot2 = self.dqdd_to_dxdot(self.forward_dynamics_gradient(point1,uk))
				A2 = np.eye(n) + 1/3*dt*dxdot2[:,0:n]
				B2 = 1/3*dt*dxdot1[:,n:]                
				
				dxdot3 = self.dqdd_to_dxdot(self.forward_dynamics_gradient(point2,uk))
				A3 = np.eye(n) + 1/3*dt*dxdot3[:,0:n]
				B3 = 1/3*dt*dxdot1[:,n:]

				dxdot4 = self.dqdd_to_dxdot(self.forward_dynamics_gradient(point3,uk))
				A4 = np.eye(n) + 1/6*dt*dxdot4[:,0:n]
				B4 = 1/6*dt*dxdot1[:,n:]
				
				A = np.matmul(A4,np.matmul(A3,np.matmul(A2,A1)))
				B = np.matmul(A4,np.matmul(A3,np.matmul(A2,B1))) + np.matmul(A4,np.matmul(A3,B2)) + np.matmul(A4,B3) + B4
				
				return A,B

class DoubleIntegratorPlant(TrajoptPlant):
	def __init__(self, integrator_type: int = 0, options = {}):
		super().__init__(integrator_type, options)

	def forward_dynamics(self, x, u):
		return u

	def forward_dynamics_gradient(self, x, u):
		return np.array([0, 0, 1])

	def get_num_pos(self):
		return 1

	def get_num_vel(self):
		return 1

	def get_num_cntrl(self):
		return 1

class PendulumPlant(TrajoptPlant):
	def __init__(self, integrator_type = 0, options = {}):
		super().__init__(integrator_type, options)

	def forward_dynamics(self, x: np.ndarray, u: np.ndarray):
		# m * l^2 * theta_dd   +   b * theta_d   +   m * g * l * sin(theta) = u
		# assume 0 damping and m = l = 1
		# theta_dd = u - g * sin(theta)
		return u - 9.81 * np.sin(x[0])

	def forward_dynamics_gradient(self, x: np.ndarray, u: np.ndarray):
		return np.array([- 9.81 * np.cos(x[0]), 0, 1])

	def get_num_pos(self):
		return 1

	def get_num_vel(self):
		return 1

	def get_num_cntrl(self):
		return 1

# http://www.matthewpeterkelly.com/tutorials/cartPole/index.html
class CartPolePlant(TrajoptPlant):
	def __init__(self, integrator_type = 0, options = {}):
		super().__init__(integrator_type, options)

	def forward_dynamics(self, x: np.ndarray, u: np.ndarray):
		gravity = self.options['gravity']
		# assuming m_cart = m_pole = l_pole = 1
		q = x[0] # position of cart on track
		theta = x[1] # angle of pole
		q_d = x[2]
		theta_d = x[3]
		st = np.sin(theta)
		ct = np.cos(theta)

		LHS = np.array([[ct,1],[2, ct]])
		rhs = np.array([[gravity*st],[u[0] + theta_d*theta_d*st]])

		LHS_inv = np.linalg.inv(LHS)

		xdd = np.matmul(LHS_inv,rhs)

		return xdd.flatten()

	def forward_dynamics_gradient(self, x: np.ndarray, u: np.ndarray):
		gravity = self.options['gravity']
		# assuming m_cart = m_pole = l_pole = 1
		q = x[0] # position of cart on track
		theta = x[1] # angle of pole
		q_d = x[2]
		theta_d = x[3]
		st = np.sin(theta)
		ct = np.cos(theta)

		LHS = np.array([[ct,1],[2, ct]])
		rhs = np.array([[gravity*st],[u[0] + theta_d*theta_d*st]])
		
		LHS_inv = np.linalg.inv(LHS)

		LHS_dtheta = np.array([[-st,0],[0, -st]])
		rhs_dtheta = np.array([[gravity*ct],[theta_d*theta_d*ct]])
		rhs_dtheta_d = np.array([[0],[2*theta_d*st]])
		rhs_du = np.array([[0],[1]])

		LHS_inv_dtheta = -np.matmul(LHS_inv,np.matmul(LHS_dtheta,LHS_inv))

		result = np.zeros((2,5))
		result[:,1:2] = np.matmul(LHS_inv_dtheta,rhs) + np.matmul(LHS_inv,rhs_dtheta)
		result[:,3:4] = np.matmul(LHS_inv,rhs_dtheta_d)
		result[:,4:5] = np.matmul(LHS_inv,rhs_du)

		return result

	def get_num_pos(self):
		return 2

	def get_num_vel(self):
		return 2

	def get_num_cntrl(self):
		return 1

class URDFPlant(TrajoptPlant):
	def __init__(self, integrator_type = 0, options = {}):
		super().__init__(integrator_type, options, True)
		parser = URDFParser()
		self.robot = parser.parse(options['path_to_urdf'])
		self.rbdReference = RBDReference(self.robot)

	def forward_dynamics(self, x: np.ndarray, u: np.ndarray):
		nq = self.get_num_pos()
		q = x[0:nq]
		qd = x[nq:]
		(c, _, _, _) = self.rbdReference.rnea(q, qd, None, self.options['gravity'])
		Minv = self.rbdReference.minv(q)
		qdd = np.matmul(Minv,(u-c))
		return qdd

	def forward_dynamics_gradient(self, x: np.ndarray, u: np.ndarray):
		nq = self.get_num_pos()
		q = x[0:nq]
		qd = x[nq:]
		(c, _, _, _) = self.rbdReference.rnea(q, qd, None, self.options['gravity'])
		Minv = self.rbdReference.minv(q)
		qdd = np.matmul(Minv,(u-c))
		dc_du = self.rbdReference.rnea_grad(q, qd, qdd, self.options['gravity'])
		df_du = np.matmul(-Minv,dc_du)
		return np.hstack((df_du,Minv))

	def get_num_pos(self):
		return self.robot.get_num_pos()

	def get_num_vel(self):
		return self.robot.get_num_vel()

	def get_num_cntrl(self):
		return self.robot.get_num_cntrl()