.. _installation:

Installation Steps
--------------------

1. To utilize the PyCHAMP package, begin by installing the Gurobi solver (version 11.0.2 or higher). Gurobi is commercial software; however, it offers free licenses for academic purposes and recent graduates. For detailed instructions on how to download and install Gurobi with a free license, please refer to the Gurobi website (https://www.gurobi.com/academia/academic-program-and-licenses).

2. Proceed with the installation of the PyCHAMP package. This process will automatically ensure the inclusion of all necessary dependencies for the package, such as gurobipy, mesa (version 2.1.1), joblib, scipy, numpy, and pandas. To install the latest version of PyCHAMP from Github, use either of the following lines of code.

**From GitHub**:

.. code-block:: console

   (.venv) $ pip install .


**From the source**:

.. code-block:: console

   (.venv) $ pip install git+https://github.com/philip928lin/PyCHAMP.git