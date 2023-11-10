import numpy as np

class TrajoptCost:
	def value():
		raise NotImplementedError

	def gradient():
		raise NotImplementedError

	def hessian():
		raise NotImplementedError

class QuadraticCost(TrajoptCost):
	def __init__(self, Q_in: np.ndarray, QF_in: np.ndarray, R_in: np.ndarray, xg_in: np.ndarray, QF_start = None):
		self.Q = Q_in
		self.QF = QF_in
		self.R = R_in
		self.xg = xg_in
		self.increaseCount_Q = 0
		self.increaseCount_QF = 0
		self.QF_start = QF_start

	def get_currQ(self, u = None, timestep = None):
		last_state = isinstance(u,type(None))
		shifted_QF = (not isinstance(timestep,type(None)) \
			          and not isinstance(self.QF_start,type(None)) \
			          and timestep >= self.QF_start)
		use_QF =  last_state or shifted_QF
		currQ = self.QF if use_QF else self.Q
		return currQ

	def value(self, x: np.ndarray, u: np.ndarray = None, timestep: int = None):
		delta_x = x - self.xg
		currQ = self.get_currQ(u,timestep)
		cost = 0.5*np.matmul(delta_x.transpose(),np.matmul(currQ,delta_x))
		if not isinstance(u, type(None)):
			cost += 0.5*np.matmul(u.transpose(),np.matmul(self.R,u))
		return cost

	def gradient(self, x: np.ndarray, u: np.ndarray = None, timestep: int = None):
		delta_x = x - self.xg
		currQ = self.get_currQ(u,timestep)
		top = np.matmul(delta_x.transpose(),currQ)
		if u is None:
			return top
		else:
			bottom = np.matmul(u.transpose(),self.R)
			return np.hstack((top,bottom))

	def hessian(self, x: np.ndarray, u: np.ndarray = None, timestep: int = None):
		nx = self.Q.shape[0]
		nu = self.R.shape[0]
		currQ = self.get_currQ(u,timestep)
		if u is None:
			return currQ
		else:
			top = np.hstack((currQ,np.zeros((nx,nu))))
			bottom = np.hstack((np.zeros((nu,nx)),self.R))
			return np.vstack((top,bottom))

	def increase_QF(self, multiplier: float = 2.0):
		self.QF *= multiplier
		self.increaseCount_QF += 1
		return self.increaseCount_QF

	def increase_Q(self, multiplier: float = 2.0):
		self.Q *= multiplier
		self.increaseCount_Q += 1
		return self.increaseCount_Q

	def reset_increase_count_QF(self):
		self.increaseCount_QF = 0

	def reset_increase_count_Q(self):
		self.increaseCount_Q = 0

	def shift_QF_start(self, shift: float = -1.0):
		self.QF_start += shift
		self.QF_start = max(self.QF_start, 0)
		return self.QF_start