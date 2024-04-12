.. _tutorial:

###########
Tutorial
###########

Introduction
==============

This tutorial will guide you through the basics of using the PyCHAMP model. The tutorial is based on running the simulation as per environmental and agricultural settings for the Sheridan 6 Local Enhanced Management Area (SD6 LEMA); users have the leeway to develop their own model to meet their specific requirements. This model simulates actions, interactions, and decisions of the agents, i.e., fields, wells, aquifers, finance, and behavior over a specified peiod of time in the SD6 region.

.. note:: 
   Ensure you have installed PyCHAMP Package as per the instructions in the :doc:`usage`.
Getting Started
=================

Start by importing the necessary modules:

.. code-block:: python

    import os
    import dill
    import numpy as np
    import pandas as pd
    from tqdm import tqdm
    from dateutil.relativedelta import relativedelta
    from scipy.stats import truncnorm
    import geopy.distance
    import py_champ

Inputs
--------

Simulation through an SD6 model will require the following inputs:

 1. Initial, Start, and End Years:
    The initial year sets up the model for simulating through required years, i.e., the span between the start and end years. For eg., the model updates a farmer's CONSUMAT state for the start year based on the inputs for the initial year (for more info on this go to :doc:`py_champ.entities.behavior`).
     
 2. Crop Options and Growing Seasons:
    Crop options include different types of crops that can be grown in the region of study, including an option to leave the field fallow.
    The growing seasons are given as an input to facilitate available precipitation calculation for each crop.

 3. Random number generator:
    A random number generator with the same seed value should be used if reproducibility of the model result is a primary concern. 

.. code-block:: python

    init = 2007
    start = 2008
    end = 2022
    crop_options = ["corn", "sorghum", "soybeans", "wheat", "fallow"]
    growing_seasons = {
        "corn":     ["5/1", "10/3"],
        "sorghum":  ["6/2", "11/3"],
        "soybeans": ["6/2", "10/15"],
        "wheat":    ["10/3", "6/27"]
        }
    nprng = np.random.default_rng(12345)

 4. General information for the region of the study:

    Gridded input data:
    Csv files with climate, subsurface, crop, and irrigation data are read. Climate data includes annual precipitation data extracted from gridMET, the IDs of which are assigned to corresponding Grid IDs of the entire study area. Hydraulic conductivity (m/d), specific yield, well depth elevation (m), water level elevation (m), and saturated thickness of the aquifer are the subsurface data fed in a yearly basis. The type of crops grown in the region each year and whether each grid, which is basically a field, was irrigated each year, along with the irrigation frequency are also included in the grid information. 

    Non-gridded input data:
    Yearly irrigation technology ratio  







Basic Usage
=============

Explain the basic usage of your package:

.. code-block:: python

    # Example code snippet
    result = your_package.some_function()
    print(result)

Advanced Topics
=================

Discuss more advanced topics or tips and tricks here.

Conclusion
============

Wrap up the tutorial and perhaps provide some links to further resources or next steps.
