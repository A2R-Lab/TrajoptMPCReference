import numpy as np

class BoxConstraint:
	def __init__(self, constraint_size: int = 0, num_timesteps: int = 0, upper_bounds: list[float] = [], lower_bounds: list[float] = [], mode: str = "NONE", options = {}):
		# constants
		self.constraint_size = constraint_size
		self.num_timesteps = num_timesteps
		self.num_constraints = 2*self.constraint_size * self.num_timesteps
		# bounds
		lblen = len(lower_bounds)
		ublen = len(upper_bounds)
		if (lblen != constraint_size and lblen != 1) or (ublen != constraint_size and ublen != 1):
			print("[!]ERROR please enter bounds of the size of constraint or constant 1")
			exit()
		self.bounds = np.zeros((2*self.constraint_size))
		self.bounds[:self.constraint_size] = lower_bounds
		self.bounds[self.constraint_size:] = upper_bounds
		# mode
		self.validate_constraint_mode(mode, options)
		self.quadratic_penalty_mu = self.options['quadratic_penalty_mu_init'] * np.ones((2*self.constraint_size,self.num_timesteps))
		self.augmented_lagrangian_lambda = np.zeros((2*self.constraint_size,self.num_timesteps))
		self.augmented_lagrangian_phi = self.options['augmentated_lagrangian_phi_init'] * np.ones((2*self.constraint_size,self.num_timesteps))

	def is_hard_constraint_mode(self, mode: str = None):
		if mode is None:
			mode = self.mode
		return (mode in ["ACTIVE_SET", "FULL_SET"])

	def is_soft_constraint_mode(self, mode: str = None):
		if mode is None:
			mode = self.mode
		return (mode in ["QUADRATIC_PENALTY", "AUGMENTED_LAGRANGIAN", "ADMM_PROJECTION"])

	def validate_constraint_mode(self, mode: str, options = {}):
		self.mode = mode
		if self.is_hard_constraint_mode() or self.is_soft_constraint_mode():
			options.setdefault('quadratic_penalty_mu_init', 1e-2)
			options.setdefault('quadratic_penalty_mu_factor', 10.0)
			options.setdefault('quadratic_penalty_mu_max', 1e12)
			options.setdefault('augmentated_lagrangian_phi_init', 1e-2)
			options.setdefault('augmentated_lagrangian_phi_factor', 10.0)
			options.setdefault('jacobian_extra_columns_head', 0)
			options.setdefault('jacobian_extra_columns_tail', 0)
			self.options = options
		else:
			print("[!Error] Invalid Constraint Mode")
			print("Options are [ACTIVE_SET, FULL_SET, QUADRATIC_PENALTY, AUGMENTED_LAGRANGIAN, ADMM_PROJECTION]")
			exit()

	def value(self, xk: np.ndarray, timestep: int = None, mode: str = None):
		if mode is None:
			mode = self.mode
		delta_lb = xk[:self.constraint_size] - self.bounds[:self.constraint_size]
		delta_ub = self.bounds[self.constraint_size:] - xk[:self.constraint_size]
		full_value = np.vstack((delta_lb,delta_ub))
		# return hard constraint value
		if self.is_hard_constraint_mode(mode):
			if mode == "ACTIVE_SET":
				return full_value[full_value < 0]
			elif mode == "FULL_SET":
				return full_value
		# else return soft constraint jacobian
		elif self.is_soft_constraint_mode(mode):
			if mode == "QUADRATIC_PENALTY" or mode == "AUGMENTED_LAGRANGIAN":
				if timestep is None:
					print("[!]ERROR Need Timestep for Soft Constraint Mode")
					exit()
				# squared term
				sq_err = np.square(full_value)
				value = np.sum(np.dot(self.quadratic_penalty_mu[:,timestep],sq_err))
				# AL term
				if mode == "AUGMENTED_LAGRANGIAN":
					value += np.dot(self.augmented_lagrangian_lambda[:,timestep],full_value)
				return value
			elif mode == "ADMM_PROJECTION":
				print("[!] ERROR NOT IMPLEMENTED YET")
				exit()
				# TBD

	def jacobian(self, xk: np.ndarray, timestep: int = None, mode: str = None):
		if mode is None:
			mode = self.mode
		# compute jacobian
		value = self.value(xk, mode = "FULL_SET")
		active_values = value < 0
		base = np.diag(np.hstack((np.ones(self.constraint_size),-np.ones(self.constraint_size))))
		vec = np.matmul(base,active_values)
		full_jac = np.vstack((np.diag(vec[:self.constraint_size]),np.diag(vec[self.constraint_size:])))
		# add head or tail columns as needed
		if self.options['jacobian_extra_columns_head'] > 0:
			full_jac = np.hstack((np.zeros((full_jac.shape[0],self.options['jacobian_extra_columns_head'])), full_jac))
		if self.options['jacobian_extra_columns_tail'] > 0:
			full_jac = np.hstack((full_jac, np.zeros((full_jac.shape[0],self.options['jacobian_extra_columns_tail']))))
		# return hard constraint jacobian
		if self.is_hard_constraint_mode(mode):
			# then return the correct rows
			if mode == "ACTIVE_SET":
				return full_jac[~np.all(full_jac == 0, axis=1)]
			elif mode == "FULL_SET":
				return full_jac
		# else return soft constraint jacobian
		elif self.is_soft_constraint_mode(mode):
			if mode == "QUADRATIC_PENALTY" or mode == "AUGMENTED_LAGRANGIAN":
				# squared term
				if timestep is None:
					print("[!]ERROR Need Timestep for Soft Constraint Mode")
					exit()
				sq_term = np.multiply(value,full_jac)
				jac = 2*np.matmul(np.reshape(self.quadratic_penalty_mu[:,timestep],value.T.shape),sq_term)
				if mode == "AUGMENTED_LAGRANGIAN":
					jac += np.matmul(self.augmented_lagrangian_lambda[:,timestep],full_jac)
				return jac.T
			elif mode == "ADMM_PROJECTION":
				print("[!] ERROR NOT IMPLEMENTED YET")
				exit()
				# TBD

	def max_soft_constraint_value(self, x: np.ndarray):
		max_value = 0
		for timestep in range(self.num_timesteps):
			value = self.value(x[:,timestep], mode = "FULL_SET")
			max_value = max(max_value, abs(min(value))) # if active value < 0 so the min is the biggest violation
		return max_value

	def update_soft_constraint_constants(self, x: np.ndarray):
		# loop through the constaints at each timestep
		mu_max_flag = True
		for timestep in range(self.num_timesteps):
			# first compute the errors
			value = self.value(x[:,timestep], mode = "FULL_SET")
			active_values = value < 0
			# then determine if mu or lambda update
			lambda_flag = abs(value) < np.reshape(self.augmented_lagrangian_phi[:,timestep], value.shape)
			lambda_update_flag = np.logical_and(active_values,lambda_flag)
			mu_update_flag = np.logical_and(active_values,np.logical_not(lambda_flag))
			# update each constraint accordingly
			for cnstr_ind in range(len(value)):
				# update mu
				if mu_update_flag[cnstr_ind]:
					# check if at max
					curr_mu = self.quadratic_penalty_mu[cnstr_ind,timestep]
					if curr_mu < self.options['quadratic_penalty_mu_max']:
						mu_max_flag = False
						self.quadratic_penalty_mu[cnstr_ind,timestep] = min(self.options['quadratic_penalty_mu_max'], \
																	curr_mu * self.options['quadratic_penalty_mu_factor'])
				# update lambda and phi
				elif lambda_update_flag[cnstr_ind]:
					mu_max_flag = False
					self.augmented_lagrangian_lambda[cnstr_ind,timestep] += self.quadratic_penalty_mu[cnstr_ind,timestep] * value[cnstr_ind]
					self.augmented_lagrangian_phi[cnstr_ind,timestep] /= self.options['augmentated_lagrangian_phi_factor']
		return mu_max_flag

	def shift_soft_constraint_constants(self, shift_steps: int):
		# first shift
		self.quadratic_penalty_mu[:,:-shift_steps] = self.quadratic_penalty_mu[:,shift_steps:]
		self.augmented_lagrangian_lambda[:,:-shift_steps] = self.augmented_lagrangian_lambda[:,shift_steps:]
		self.augmented_lagrangian_phi[:,:-shift_steps] = self.augmented_lagrangian_phi[:,shift_steps:]
		# then load with init (and 0 for lambda)
		self.quadratic_penalty_mu[:,shift_steps:] = self.options['quadratic_penalty_mu_init']
		self.augmented_lagrangian_lambda[:,shift_steps:] = 0.0
		self.augmented_lagrangian_phi[:,shift_steps:] = self.options['augmentated_lagrangian_phi_init']

class TrajoptConstraint:
	def __init__(self, nq: int = 0, nv: int = 0, nu: int = 0, num_timesteps: int = 0):
		self.nq = nq
		self.nv = nv
		self.nu = nu
		self.num_timesteps = num_timesteps
		# specialized constraint helpers
		self.joint_limits = None
		self.velocity_limits = None
		self.torque_limits = None
		# generic constraints
		# TBD

	def set_joint_limits(self, upper_bounds: list[float], lower_bounds: list[float], mode: str, options = {}):
		# construct box constraint object
		options['jacobian_extra_columns_tail'] = self.nv + self.nu
		self.joint_limits = BoxConstraint(self.nq, self.num_timesteps-1, upper_bounds, lower_bounds, mode, options)

	def set_velocity_limits(self, upper_bounds: list[float], lower_bounds: list[float], mode: str, options = {}):
		# construct box constraint object
		options['jacobian_extra_columns_head'] = self.nq
		options['jacobian_extra_columns_tail'] = self.nu
		self.velocity_limits = BoxConstraint(self.nv, self.num_timesteps, upper_bounds, lower_bounds, mode, options)

	def set_torque_limits(self, upper_bounds: list[float], lower_bounds: list[float], mode: str, options = {}):
		# construct box constraint object
		options['jacobian_extra_columns_head'] = self.nq + self.nv
		self.torque_limits = BoxConstraint(self.nu, self.num_timesteps - 1, upper_bounds, lower_bounds, mode, options)

	def value_hard_constraints(self, xk: np.ndarray, uk: np.ndarray = None, timestep: int = None):
		constraint_index = 0
		if timestep is None:
			timestep = self.num_timesteps - 1
		ck = None
		# add special constraints
		if (not (self.joint_limits is None)) and self.joint_limits.is_hard_constraint_mode():
			ck = self.joint_limits.value(xk, timestep = timestep)
			constraint_index += 2*self.nq
		if (not (self.velocity_limits is None)) and self.velocity_limits.is_hard_constraint_mode():
			val = self.velocity_limits.value(xk, timestep = timestep)
			if ck is None:
				ck = val
			else:
				ck = np.vstack((ck,val))
			constraint_index += 2*self.nv
		if (not (self.torque_limits is None)) and self.torque_limits.is_hard_constraint_mode() and timestep < self.num_timesteps - 1:
			val = self.torque_limits.value(uk, timestep = timestep)
			if ck is None:
				ck = val
			else:
				ck = np.vstack((ck,val))
		return ck

	def jacobian_hard_constraints(self, xk: np.ndarray, uk: np.ndarray = None, timestep: int = None):
		constraint_index = 0
		if timestep is None:
			timestep = self.num_timesteps - 1
		Ck = None
		# add special constraints
		if (not (self.joint_limits is None)) and self.joint_limits.is_hard_constraint_mode():
			Ck = self.joint_limits.jacobian(xk, timestep = timestep)	
			constraint_index += 2*self.nq
		if (not (self.velocity_limits is None)) and self.velocity_limits.is_hard_constraint_mode():
			jac = self.velocity_limits.jacobian(xk, timestep = timestep)	
			if Ck is None:
				Ck = jac
			else:
				Ck = np.vstack((Ck,jac))
			constraint_index += 2*self.nv
		if (not (self.torque_limits is None)) and self.torque_limits.is_hard_constraint_mode():
			jac = self.torque_limits.jacobian(uk, timestep = timestep)
			if Ck is None:
				Ck = jac
			else:
				Ck = np.vstack((Ck,jac))
		return Ck

	def len_or_none(self, x):
		if x is None:
			return 0
		return len(x)
		
	def total_hard_constraints(self, x: np.ndarray, u: np.ndarray, timestep: int = None):
		total = 0
		if timestep is None:
			for k in range(self.num_timesteps - 1):
				total += self.len_or_none(self.value_hard_constraints(x[:,k], u[:,k], k))
			total += self.len_or_none(self.value_hard_constraints(x[:,self.num_timesteps - 1], timestep = self.num_timesteps - 1))
		else:
			if timestep == self.num_timesteps - 1:
				total += self.len_or_none(self.value_hard_constraints(x[:,self.num_timesteps - 1], timestep = self.num_timesteps - 1))	
			else:
				total += self.len_or_none(self.value_hard_constraints(x[:,timestep], u[:,timestep], timestep))
		return total

	def value_soft_constraints(self, xk: np.ndarray, uk: np.ndarray = None, timestep: int = None):
		if timestep is None:
			timestep = self.num_timesteps - 1
		value = 0
		# add special constraints
		if (not (self.joint_limits is None)) and self.joint_limits.is_soft_constraint_mode():
			value += self.joint_limits.value(xk, timestep = timestep)
		if (not (self.velocity_limits is None)) and self.velocity_limits.is_soft_constraint_mode():
			value += self.velocity_limits.value(xk, timestep = timestep)
		if (not (self.torque_limits is None)) and self.torque_limits.is_soft_constraint_mode() and timestep < self.num_timesteps - 1:
			value += self.torque_limits.value(uk, timestep = timestep)
		return value

	def jacobian_soft_constraints(self, xk: np.ndarray, uk: np.ndarray = None, timestep: int = None):
		if timestep is None:
			timestep = self.num_timesteps - 1
		jacobian = None
		# add special constraints
		if (not (self.joint_limits is None)) and self.joint_limits.is_soft_constraint_mode():
			jacobian = self.joint_limits.jacobian(xk, timestep = timestep)
		if (not (self.velocity_limits is None)) and self.velocity_limits.is_soft_constraint_mode():
			jk = self.velocity_limits.jacobian(xk, timestep = timestep)
			if jacobian is None:
				jacobian = jk
			else:
				jacobian = np.vstack((jacobian,jk))
		if (not (self.torque_limits is None)) and self.torque_limits.is_soft_constraint_mode() and timestep < self.num_timesteps - 1:
			jk = self.torque_limits.jacobian(uk, timestep = timestep)
			if jacobian is None:
				jacobian = jk
			else:
				jacobian = np.vstack((jacobian,jk))
		return jacobian

	def total_soft_constraints(self, timestep: int = None):
		total = 0
		if timestep is None:
			if (not (self.joint_limits is None)) and self.joint_limits.is_soft_constraint_mode():
				total += self.joint_limits.num_constraints
			if (not (self.velocity_limits is None)) and self.velocity_limits.is_soft_constraint_mode():
				total += self.velocity_limits.num_constraints
			if (not (self.torque_limits is None)) and self.torque_limits.is_soft_constraint_mode():
				total += self.torque_limits.num_constraints
		else:
			if (not (self.joint_limits is None)) and self.joint_limits.is_soft_constraint_mode():
				total += self.joint_limits.constraint_size
			if (not (self.velocity_limits is None)) and self.velocity_limits.is_soft_constraint_mode():
				total += self.velocity_limits.constraint_size
			if (not (self.torque_limits is None)) and self.torque_limits.is_soft_constraint_mode() and (timestep < self.num_timesteps - 1):
				total += self.torque_limits.constraint_size
		return total

	def max_soft_constraint_value(self, x: np.ndarray, u: np.ndarray):
		max_value = 0
		if (not (self.joint_limits is None)) and self.joint_limits.is_soft_constraint_mode():
			max_value = max(max_value, self.joint_limits.max_soft_constraint_value(x))
		if (not (self.velocity_limits is None)) and self.velocity_limits.is_soft_constraint_mode():
			max_value = max(max_value, self.velocity_limits.max_soft_constraint_value(x))
		if (not (self.torque_limits is None)) and self.torque_limits.is_soft_constraint_mode():
			max_value = max(max_value, self.torque_limits.max_soft_constraint_value(u))
		return max_value

	def update_soft_constraint_constants(self, x: np.ndarray, u: np.ndarray):
		all_mu_over_limit_flag = True
		if not (self.joint_limits is None):
			all_mu_over_limit_flag = all_mu_over_limit_flag and self.joint_limits.update_soft_constraint_constants(x)
		if not (self.velocity_limits is None):
			all_mu_over_limit_flag = all_mu_over_limit_flag and self.velocity_limits.update_soft_constraint_constants(x)
		if not (self.torque_limits is None):
			all_mu_over_limit_flag = all_mu_over_limit_flag and self.torque_limits.update_soft_constraint_constants(u)
		return all_mu_over_limit_flag

	def shift_soft_constraint_constants(self, shift_steps: int):
		if not (self.joint_limits is None):
			self.joint_limits.shift_soft_constraint_constants(shift_steps)
		if not (self.velocity_limits is None):
			self.velocity_limits.shift_soft_constraint_constants(shift_steps)
		if not (self.torque_limits is None):
			self.torque_limits.shift_soft_constraint_constants(shift_steps)