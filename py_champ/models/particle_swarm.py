# Import standard library
import logging
import os
import sys

import dill

sys.setrecursionlimit(10000)
# Import modules
import numpy as np
from joblib import Parallel, delayed
from joblib.externals.loky import set_loky_pickler

set_loky_pickler("dill")

from collections import deque

import matplotlib.pyplot as plt
from pyswarms.backend.handlers import BoundaryHandler, OptionsHandler, VelocityHandler
from pyswarms.backend.operators import compute_pbest
from pyswarms.backend.topology import Star
from pyswarms.base import SwarmOptimizer
from pyswarms.utils.plotters import plot_cost_history
from pyswarms.utils.reporter import Reporter


class GlobalBestPSO(SwarmOptimizer):
    def __init__(
        self,
        n_particles,
        dimensions,
        options,
        bounds=None,
        oh_strategy=None,
        bh_strategy="periodic",
        velocity_clamp=None,
        vh_strategy="unmodified",
        center=1.00,
        ftol=-np.inf,
        ftol_iter=1,
        init_pos=None,
        wd=None,
        load_dict=None,
    ):
        """Initialize the swarm.

        Attributes
        ----------
        n_particles : int
            number of particles in the swarm.
        dimensions : int
            number of dimensions in the space.
        options : dict with keys :code:`{'c1', 'c2', 'w'}`
            a dictionary containing the parameters for the specific
            optimization technique.
                * c1 : float
                    cognitive parameter
                * c2 : float
                    social parameter
                * w : float
                    inertia parameter
        bounds : tuple of numpy.ndarray, optional
            a tuple of size 2 where the first entry is the minimum bound while
            the second entry is the maximum bound. Each array must be of shape
            :code:`(dimensions,)`.
        oh_strategy : dict, optional, default=None(constant options)
            a dict of update strategies for each option.
        bh_strategy : str
            a strategy for the handling of out-of-bounds particles.
        velocity_clamp : tuple, optional
            a tuple of size 2 where the first entry is the minimum velocity and
            the second entry is the maximum velocity. It sets the limits for
            velocity clamping.
        vh_strategy : str
            a strategy for the handling of the velocity of out-of-bounds particles.
        center : list (default is :code:`None`)
            an array of size :code:`dimensions`
        ftol : float
            relative error in objective_func(best_pos) acceptable for
            convergence. Default is :code:`-np.inf`
        ftol_iter : int
            number of iterations over which the relative error in
            objective_func(best_pos) is acceptable for convergence.
            Default is :code:`1`
        init_pos : numpy.ndarray, optional
            option to explicitly set the particles' initial positions. Set to
            :code:`None` if you wish to generate the particles randomly.
        """
        super().__init__(
            n_particles=n_particles,
            dimensions=dimensions,
            options=options,
            bounds=bounds,
            velocity_clamp=velocity_clamp,
            center=center,
            ftol=ftol,
            ftol_iter=ftol_iter,
            init_pos=init_pos,
        )

        if oh_strategy is None:
            oh_strategy = {}
        # Initialize logger
        self.rep = Reporter(
            log_path=os.path.join(wd, "Report.log"), logger=logging.getLogger(__name__)
        )
        self.wd = wd
        # Initialize the resettable attributes
        if load_dict is None:
            self.reset()
        # Initialize the topology
        self.top = Star()
        self.bh = BoundaryHandler(strategy=bh_strategy)
        self.vh = VelocityHandler(strategy=vh_strategy)
        self.oh = OptionsHandler(strategy=oh_strategy)
        self.name = __name__

        # Populate memory of the handlers
        self.bh.memory = self.swarm.position
        self.vh.memory = self.swarm.position
        # Initialize cost
        if load_dict is None:
            self.swarm.pbest_cost = np.full(self.swarm_size[0], np.inf)
            self.culmulated_iter = 0
        else:
            self.swarm = load_dict["swarm"]
            self.pos_history = load_dict["pos_history"]
            self.cost_history = load_dict["cost_history"]
            self.culmulated_iter = load_dict["culmulated_iter"]

    def to_dict(self):
        dict_to_save = {
            "swarm": self.swarm,
            "pos_history": self.pos_history,
            "cost_history": self.cost_history,
            "culmulated_iter": self.culmulated_iter,
        }
        return dict_to_save

    def optimize(self, objective_func, iters, n_processes=None, verbose=60, **kwargs):
        """Optimize the swarm for a number of iterations.

        Performs the optimization to evaluate the objective
        function :code:`f` for a number of iterations :code:`iter.`

        Parameters
        ----------
        objective_func : callable
            objective function to be evaluated
        iters : int
            number of iterations
        n_processes : int
            number of processes to use for parallel particle evaluation (default: None = no parallelization)
        verbose : bool
            enable or disable the logs and progress bar (default: True = enable logs)
        kwargs : dict
            arguments for the objective function

        Returns
        -------
        tuple
            the global best cost and the global best position.
        """
        # Apply verbosity
        if verbose > 0:
            log_level = logging.INFO
        else:
            log_level = logging.NOTSET

        self.rep.log(f"Obj. func. args: {kwargs}", lvl=logging.DEBUG)
        self.rep.log(
            f"Optimize for {iters} iters with {self.options}",
            lvl=log_level,
        )

        ftol_history = deque(maxlen=self.ftol_iter)
        for i_iter in self.rep.pbar(iters, self.name) if verbose > 0 else range(iters):
            culmulated_iter = self.culmulated_iter
            # Part 1: Evaluation
            positions = self.swarm.position
            if n_processes is None:
                current_cost = []
                for i_particle in range(self.swarm.n_particles):
                    current_cost.append(
                        objective_func(
                            positions[i_particle, :],
                            **kwargs,
                            i_iter=culmulated_iter,
                            i_particle=i_particle,
                        )
                    )
            else:
                current_cost = Parallel(n_jobs=n_processes, verbose=verbose)(
                    delayed(objective_func)(
                        positions[i_particle, :],
                        **kwargs,
                        i_iter=culmulated_iter,
                        i_particle=i_particle,
                    )
                    for i_particle in range(self.swarm.n_particles)
                )
            self.swarm.current_cost = np.array(current_cost)

            self.swarm.pbest_pos, self.swarm.pbest_cost = compute_pbest(self.swarm)
            # Set best_cost_yet_found for ftol
            best_cost_yet_found = self.swarm.best_cost
            self.swarm.best_pos, self.swarm.best_cost = self.top.compute_gbest(
                self.swarm
            )
            # fmt: on
            if verbose > 0:
                self.rep.hook(best_cost=self.swarm.best_cost)

            # Save to history
            hist = self.ToHistory(
                best_cost=self.swarm.best_cost,
                mean_pbest_cost=np.mean(self.swarm.pbest_cost),
                mean_neighbor_cost=self.swarm.best_cost,
                position=self.swarm.position,
                velocity=self.swarm.velocity,
            )
            self._populate_history(hist)
            # Verify stop criteria based on the relative acceptable cost ftol
            relative_measure = self.ftol * (1 + np.abs(best_cost_yet_found))
            delta = (
                np.abs(self.swarm.best_cost - best_cost_yet_found) < relative_measure
            )
            if i_iter < self.ftol_iter:
                ftol_history.append(delta)
            else:
                ftol_history.append(delta)
                if all(ftol_history):
                    break
            # Perform options update  (ignore this for continue optimize)
            self.swarm.options = self.oh(self.options, iternow=i_iter, itermax=iters)
            # Perform velocity and position updates
            self.swarm.velocity = self.top.compute_velocity(
                self.swarm, self.velocity_clamp, self.vh, self.bounds
            )
            self.swarm.position = self.top.compute_position(
                self.swarm, self.bounds, self.bh
            )

            if verbose > 0:
                # Plot cost
                fig, ax = plt.subplots()
                plot_cost_history(cost_history=self.cost_history, ax=ax)
                fig.savefig(os.path.join(self.wd, "cost_history.png"))
                plt.close()

            # save
            dict_item = self.to_dict()
            with open(os.path.join(self.wd, f"PSO_it{culmulated_iter}.pkl"), "wb") as f:
                dill.dump(dict_item, f)

            self.culmulated_iter += 1

        # Obtain the final best_cost and the final best_position
        final_best_cost = self.swarm.best_cost.copy()
        final_best_pos = self.swarm.pbest_pos[self.swarm.pbest_cost.argmin()].copy()
        # Write report in log and return final cost and position
        self.rep.log(
            "Optimization finished | best cost: {}, best pos: {}".format(
                final_best_cost, final_best_pos
            ),
            lvl=log_level,
        )
        return (final_best_cost, final_best_pos)
