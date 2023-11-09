# TrajoptMPCReference

A Python reference implementation of trajectory optimization (trajopt) algorithms and model predictive control (MPC).

This package is designed to enable rapid prototyping and testing of new algorithms and algorithmic optimizations. If your favorite trajectory optimizatio algorithm is not yet implemented please submit a PR with the implementation. We can then then try to get a GPU implementation designed as soon as possible. Please see our other repositories for available GPU implementations.

**This package contains submodules make sure to run ```git submodule update --init --recursive```** after cloning!

## Usage and API:
This package relies on a series of objects encapsulating different components of trajectory optimization algorithms and model predictive control pipelines including:
+ ```TrajoptPlant```: which wraps a series of canonical systems (double integrator, pendlume, cart pole) as well as parsed URDFs via the ```URDFParser``` and ```RBDReference``` packages from ```GRiD```.
+ ```TrajoptCost```: which wraps canonical cost function types and currently supports quadratic costs.
+ ```TrajoptConstraint```: which wraps canonical constraint types and currently supports joint, velocity, and torque limits via box constraints implemented via hard constraints (Full Sets, Active Sets) and soft constraints (Quadratic Penalty Methods, Augmented Lagrangian Methods).

Currently implemented algorithms include:
+ Sequential Quadratic Programming through the use of KKT factorization, Schur factorization, or PCG solves of the Scher complement
+ iLQR (only supports soft constraints at this time)

## Instalation Instructions::
In order to support the wrapped packages there are 4 required external packages ```beautifulsoup4, lxml, numpy, sympy``` which can be automatically installed by running:
```shell
pip3 install -r requirements.txt
```
Wrapped packages include: [GRiD](https://github.com/A2R-Lab/GRiD) and [GBD-PCG-Python](https://github.com/A2R-Lab/GBD-PCG-Python).

### Citing
To cite this work in your research, please use the following bibtexs depending on the parts of this package you use:
```
@misc{adabag2023mpcgpu,
      title={MPCGPU: Real-Time Nonlinear Model Predictive Control through Preconditioned Conjugate Gradient on the GPU}, 
      author={Emre Adabag and Miloni Atal and William Gerard and Brian Plancher},
      year={2023},
      eprint={2309.08079},
      archivePrefix={arXiv},
      primaryClass={cs.RO}
}
@inproceedings{plancher2022grid,
  title={GRiD: GPU-Accelerated Rigid Body Dynamics with Analytical Gradients}, 
  author={Brian Plancher and Sabrina M. Neuman and Radhika Ghosal and Scott Kuindersma and Vijay Janapa Reddi},
  booktitle={IEEE International Conference on Robotics and Automation (ICRA)}, 
  year={2022}, 
  month={May}
}
```