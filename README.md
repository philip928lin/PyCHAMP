[![Documentation Status](https://readthedocs.org/projects/dises-pychamp/badge/?version=latest)](https://dises-pychamp.readthedocs.io/en/latest/?badge=latest)

# PyCHAMP

<<<<<<< Updated upstream
The Crop-Hydrological-Agent Modeling Platform (PyCHAMP) is a Python-based open-source package designed for modeling agro-hydrological systems. The modular design, incorporating aquifer, crop field, groundwater well, finance, and behavior modules, enables users to simulate and analyze the interactions between human and natural systems, considering both environmental and socio-economic factors. PyCHAMP serves as a useful tool for human-water research and sustainable groundwater management, offering prospects for future integration with detailed sub-models and systematic evaluation of model structural uncertainty.

### Preprint
[http://dx.doi.org/10.2139/ssrn.4814225](http://dx.doi.org/10.2139/ssrn.4814225)

### Manual
[https://diese-pychamp.readthedocs.io](https://diese-pychamp.readthedocs.io/en/latest/?badge=latest#)

## To install PyCHAMP, proceed with the following steps:
=======
## Overview
>>>>>>> Stashed changes

PyCHAMP (Crop-Hydrological-Agent Modeling Platform) is an open-source Python package specifically developed for analyzing policies and assessing decision-making processes in agro-hydrological systems. This platform is particularly useful for systems characterized by complex interactions between human activities and natural resources. PyCHAMP is structured into five core components—**Aquifer**, **Field**, **Well**, **Finance**, and **Behavior**. Each component acts as a module, encapsulating classes that represent different agent types within these systems.

<<<<<<< Updated upstream
2. Install the PyCHAMP package. Running the subsequent line of code will ensure the inclusion of all necessary dependencies for the package, such as gurobipy, mesa (version 2.1.1), joblib, scipy, numpy, and pandas.
	
	`pip install git+https://github.com/philip928lin/PyCHAMP.git`


=======
In PyCHAMP, a **component** is implemented as a module, which is a file containing Python statements and definitions. This includes classes and functions relevant to that component. An **agent type** is implemented as a class, serving as a template for creating agent objects. These objects encapsulate both attributes (data) and methods (functions) that define their behavior. PyCHAMP includes five default agent types, one for each component: **Aquifer**, **Field**, **Well**, **Finance**, and **Behavior**.

## Preprint
[http://dx.doi.org/10.2139/ssrn.4814225](http://dx.doi.org/10.2139/ssrn.4814225)

## Manual
[https://diese-pychamp.readthedocs.io](https://diese-pychamp.readthedocs.io/en/latest/?badge=latest#)

## Installation

### Prerequisites

Before installing PyCHAMP, you must install the Gurobi optimization software, which PyCHAMP depends on for certain computations.

1. **Install Gurobi**:
   - Gurobi is available under a free academic license for qualified users. Detailed instructions on how to download and install Gurobi can be found at the [Gurobi website](https://www.gurobi.com/academia/academic-program-and-licenses).

### Installing PyCHAMP

You can install PyCHAMP directly from the source or via the repository on GitHub:

- **From GitHub**:
  ```bash
  pip install .
  ```

- **From the source**:
  ```bash
  pip install git+https://github.com/philip928lin/PyCHAMP.git
  ```
>>>>>>> Stashed changes
