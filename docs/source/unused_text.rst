.. _example_SD6:

##################################
Introduction to the PyCHAMP Model
##################################

.. _general_intro:  



1. Aquifer Module and Aquifer Class:

The aquifer class inside the aquifer module simulates the changes in the groundwater level on an annual basis using the Kansas Geological Survey - Water Balance Method (KGS-WBM). The following KGS-WBM was proposed by Butler et al. (2018): 

.. math::

   \Delta WL = \frac{1}{Area \times S_y} - \frac{Q}{area \times S_y} \approx b_{aq} - a_{aq}Q

Where:

- :math:`\Delta WL` is the annual groundwater level change :math:`[m]`,
- :math:`I` is net inflow :math:`[10^4 m^3]`,
- :math:`Q` is the annual pumping amount :math:`[10^4 m^3]`.
- :math:`S_y` is the specific yield :math:`[-]` and
- :math:`Area` represents the area of the aquifer :math:`[10^4 m^2]`.

For places, where :math:`S_y` and :math:`I` change little with the time, :math:`\frac{1}{Area \times S_y}` and :math:`\frac{1}{area \times S_y}` can be replaced by :math:`a_{aq}` and :math:`b_{aq}`, respectively.

Where:

:math:`a_{aq}` and :math:`b_{aq}`, are linear regression coefficients based on the regression analysis of the annual groundwater level change and pumping data.

The major output of this class is the annual change in the water level (denoted as dwl within the package).

For a more detailed breakdown of the aquifer module, refer to :ref:`py_champ_entities_aquifer`.


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


Aquifer Class:
^^^^^^^^^^^^^^

The Aquifer class within the aquifer module models the annual variations in groundwater levels using the Kansas Geological Survey – Water Balance Method (KGS-WBM).

This class requires a unique identifier and an input "settings" dictionary as parameters. The settings dictionary allows users to define aquifer-specific attributes, such as the aquifer's specific yield.

Aquifer class calculates and returns the annual change in water level (m), denoted as dwl within the package. This calculation takes into account the aquifer's inflow and the water extracted from it. It also utilizes constant values derived from a regression analysis of the annual changes in groundwater levels and the data on water extraction.

*insert class diagram here*

For an in-depth explanation of the aquifer module, please refer to :ref:`py_champ_entities_aquifer`.

2. Field Module 
----------------
Field Class:
^^^^^^^^^^^^

The Field class, located within the field module, employs a water-yield curve to simulate crop growth annually. It calculates the annual yield for each crop type, a key input for revenue assessment in the finance module. Additionally, it determines the daily pumping rate necessary for the well energy consumption calculations performed in the well module.

The parameters for this class include a unique identifier for the field agent and an input "settings" dictionary, which contains data for field attributes such as water yield curves for different crop types.

Moreover, the class is responsible for updating the irrigation technology and crop type at each time step.

The Field class returns the total yield in bushels per hectare (bu/ha), the yield rate per field (bu/ha), and the annual volume of irrigation (m-ha) utilized by the field.

*insert class diagram here*

For an in-depth overview of the field module, please refer to :ref:`py_champ_entities_field`.

3. Well Module 
---------------
Well Class:
^^^^^^^^^^^^

The Well class inside the well module simulates the dynamics of each well defined in the model. It computes the total effective annual lift, a figure that informs the calculation of the well's yearly energy demands. This total lift includes several components: the head needed to raise water from the water table to the ground surface, adjustments for aquifer depletion using the Cooper-Jacob method, allowances for losses within the well, and the additional lift needed for water pressurization and overcoming pipe friction, which varies with the irrigation technology employed.

The parameters for the class are unique identifier and input "settings dictionary", which includes well attributes such as the well's radius.

Ultimately, the well class returns the annual energy consumption for each well, measured in petajoules (PJ).

*insert class diagram here*

For an expansive analysis of the well module, refer to :ref:`py_champ_entities_well`.

4. Finance Module 
------------------
Finance Class:
^^^^^^^^^^^^^^

Within the finance module, the Finance class is tasked with calculating the annual revenue, derived from the total yield of all fields owned by a farmer agent and the effective price of crops, which is the price earned after subtracting the costs incurred in crop production from the crop price. Additionally, it computes the profit by deducting expenses related to energy for well operation, technology maintenance, and costs associated with changing crops and irrigation technologies from the total revenue.

The parameters for this class include a unique identifier and an input "settings dictionary" that specifies various financial attributes, such as the cost of each crop type.

The Finance class returns the profit in units of ten thousand dollars (1e4 $), a crucial metric for assessing farmer satisfaction.

*insert class diagram here*

For additional information on the finance module, refer to :ref:`py_champ_entities_finance`.

5. Behavior Module 
-------------------
Behavior Class:
^^^^^^^^^^^^^^^

The Behavior class, within the behavior module, is designed to model the decision-making process of a farmer according to the CONSUMAT framework. The class conducts a series of crucial operations as outlined below:

    a. It evaluates the perceived risk for each crop across different fields, considering the given risk parameters.
    b. It updates the perceived precipitation levels, integrating forecasted values, the farmer's confidence in these forecasts, and the previously assessed risk.
    c. It executes simulations for well, field, and finance modules for a single time step, reflecting the farmer's current CONSUMAT state, and then revises the CONSUMAT state for the subsequent step in response to new satisfaction and uncertainty levels.
    d. Depending on the prevailing CONSUMAT state, it resolves an optimization problem within the Optimization class and refreshes the decision-making solutions (referred to as dm_sols in the package), which will inform the agent’s actions in future time steps.
    e. It assesses the agent's satisfaction levels against those of their network peers to solve the optimization problem when the agent’s CONSUMAT state involves social comparison or imitation.

Class parameters are as follows:

    a. A unique identifier for the agent.
    b. An input "settings dictionary" that encompasses the farmer's attributes and Gurobi optimization settings, detailing the farmer's fields, wells, social network, decision-making elements such as optimization target variables, water rights, and CONSUMAT framework parameters.
    c. A dictionary for decision-making parameters, for instance, the degree of trust in weather forecasts.
    d. Separate dictionaries detailing the specifics of the farmer agent’s fields, wells, financial aspects, and the aquifer beneath the fields.

The Behavior class stores the decision-making outcomes as a dictionary for subsequent analysis. For example, the irrigation depth defined in the dictionary is utilized to compute the irrigation volume for each field, which, when aggregated for all fields owned by a farmer, determines the total water extraction from the aquifer within the simulated model.

*insert class diagram here*

For further insights into the behavior module, refer to :ref:`py_champ_entities_behavior`.


6. Optimization Class 
------------------------
The Optimization class is responsible for resolving a non-linear mixed integer optimization problem annually using the Gurobi solver. This class makes important decisions regarding the selection of crop types, irrigation technologies, and irrigation options (either rainfed or irrigated fields) for a farmer agent, tailored to the agent's existing CONSUMAT state. The class executes several critical tasks:

    a. It prepares the initial setup for the optimization problem, sourcing parameters from user inputs for aforementioned five classes. If any user inputs is missing, the class substitutes them with default values.
    b. It establishes constraints for all fields under the farmer's ownership through iteration. The decision variables such as crop type, irrigation technology, and the choice between irrigated or rainfed cultivation may be user-specified or optimized for each field, depending on the farmer's CONSUMAT state.
    c. It formulates constraints for every well operated by the farmer, with each iteration calculating the well's drawdown and energy consumption.
    d. It arranges the financial constraints for the farmer agent, aligning with the finance input dictionary to set parameters for revenue, energy costs, and other yearly expenses, including costs for changing crops and technologies, as well as annual technology operation fees.
    e. It iteratively imposes constraints concerning the water rights held by the farmer agent.
    f. It delineates the objective for the optimization, aiming to maximize satisfaction based on the target variable specified in the behavior settings dictionary.
    g. Finally, it concludes the setup, solves the optimization problem while considering all constraints, and records the solutions within a dictionary.
    
*insert class diagram here*

The solution dictionary derived from the Optimization class informs the behavior class within the package to update the CONSUMAT state of the farmer agent.

General Structure of an agent based model based on PyCHAMP
==========================================================

PyCHAMP, utilizing the Mesa 2.1.1 Agent-Based Modeling (ABM) framework, initiates by defining a MyModel class, which is an extension of mesa.Model. A new instance of this model is created, taking various input dictionaries as parameters. Within the newly created instance of the class, different agents like aquifer, field, well, finance, and behavior, each based on their respective input dictionaries are initialized. It is important to note the flexibility in the number of agent instances; for example, there can be multiple instances of well agents, ranging from 1 to n. Upon initialization, these agents are added to the Mesa scheduler. Also, within the __init__ constructor of MyModel, a DataCollector is set up to gather and record both model-level and agent-level data during the simulation.

The model includes a step method to progress the simulation by one time unit. This method updates the attributes of the agents and specifically advances the state of the Behavior agent type through the Mesa scheduler. In a unique setup, other agent types are simulated within the Behavior agent. The aquifer agents are then iteratively processed to calculate the total annual water withdrawal, aggregating the withdrawal from each well. The withdrawal for each aquifer is then updated within the step method of the aquifer class. Additionally, the step method directs the previously initialized DataCollector to capture and store data related to the various agents.

To run the simulation, a model object is instantiated with the necessary input settings dictionaries, and the step method is called repeatedly for the desired number of iterations.

The pseudocode of a Model created with PyCHAMP modules is illustrated below:

.. code-block:: python

    Class MyModel(mesa.Model):
        Constructor __init__(settings):
            Initialize scheduler as new Mesa.Scheduler()
            For each agent_type in [aquifer, field, well, finance, behavior]:
                Initialize agent of agent_type with settings
                Add agent to scheduler with self.schedule.add(agent)
            Initialize DataCollector for storing model-level and agent-level data

        Method step():
            For each agent in self.schedule.agents:
                Update agent attributes
            Call self.schedule.step(agt_type="Behavior") to update Behavior agents
            For each aquifer_agent in aquifer_agents:
                Calculate total annual withdrawal
                Call aquifer_agent.step(withdrawal) to update withdrawal information
            Collect model and agent data with self.datacollector.collect()

    # Initialize a new instance of MyModel with settings
    model_instance = MyModel(settings)

    # Run the simulation for a predetermined number of steps
    For step in range(number_of_simulation_steps):
        model_instance.step()




Basic Usage
===========

Explain the basic usage of your package:

.. code-block:: python

    # Example code snippet
    result = your_package.some_function()
    print(result)

Advanced Topics
===============

Discuss more advanced topics or tips and tricks here.

Conclusion
==========

Wrap up the tutorial and perhaps provide some links to further resources or next steps.