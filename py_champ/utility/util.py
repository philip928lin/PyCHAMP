import time
import warnings

import numpy as np
import pandas as pd
from mesa.time import BaseScheduler


class BaseSchedulerByTypeFiltered(BaseScheduler):
    """
    A scheduler that overrides the step method to allow for filtering
    of agents by .agt_type.

    Example:
    -------
    >>> scheduler = BaseSchedulerByTypeFiltered(model)
    >>> scheduler.step(agt_type="Behavior")
    """

    def step(self, agt_type=None) -> None:
        """Execute the step of all the agents, one at a time."""
        # To be able to remove and/or add agents during stepping
        # it's necessary for the keys view to be a list.
        self.do_each(method="step", agt_type=agt_type)
        self.steps += 1
        self.time += 1

    def do_each(self, method, agent_keys=None, shuffle=False, agt_type=None):
        if agent_keys is None:
            agent_keys = self.get_agent_keys()
        if agt_type is not None:
            agent_keys = [i for i in agent_keys if self._agents[i].agt_type == agt_type]
        if shuffle:
            self.model.random.shuffle(agent_keys)
        for agent_key in agent_keys:
            if agent_key in self._agents:
                getattr(self._agents[agent_key], method)()

def get_nested_attr(obj, attr_str):
    """A patch to collect a nested attribute using MESA's datacollector.
    
    This function is used to get a nested attribute from an object.
    For example, if we have an object with an attribute "a" that is a dictionary
    with a key "b", we can get the value of "b" by calling get_nested_attr(obj, "a.b").
    This is useful when collecting data from agents in a model.

    Parameters
    ----------
    obj : object
        An object.
    attr_str : str
        A string of nested attributes separated by a period.

    Returns
    -------
    object
        The nested attribute.
    """
    attrs = attr_str.split(".", 1)
    current_attr = getattr(obj, attrs[0], None)
    if len(attrs) == 1 or current_attr is None:
        return current_attr
    return get_nested_attr(current_attr, attrs[1])

def get_agt_attr(attr_str):
    """ Get a nested attribute from an agent object.

    This replaces, e.g., lambda a: getattr(a, "satisfaction", None)
    We have to do this to return None if the attribute is not exist
    in the given agent type.
    def func(agent):
        return getattr(agent, attr, None).

    Parameters
    ----------
    attr_str : str
        A string of nested attributes separated by a period.

    Returns
    -------
    function
        A function that returns the nested attribute.
    """

    def get_nested_attr(obj):
        def get_nested_attr_(obj, attr_str):
            attrs = attr_str.split(".", 1)
            current_attr = getattr(obj, attrs[0], None)
            if len(attrs) == 1 or current_attr is None:
                return current_attr
            return get_nested_attr_(current_attr, attrs[1])

        return get_nested_attr_(obj, attr_str)

    return get_nested_attr

def dict_to_string(dictionary, prefix="", indentor="  ", level=2):
    """Ture a dictionary into a printable string.

    Parameters
    ----------
    dictionary : dict
        A dictionary.
    prefix : str, optional
        Prefix, by default "".
    indentor : str, optional
        Indentor, by default "  ".
    level : int, optional
        Level of indentation, by default 2.

    Returns
    -------
    str
        A printable string.
    """

    def dict_to_string_list(dictionary, indentor="  ", count=1, string=None):
        """Convert dictionary to string list."""
        if string is None:
            string = []
        for key, value in dictionary.items():
            string.append(prefix + indentor * count + str(key))
            if isinstance(value, dict) and count < level:
                string = dict_to_string_list(value, indentor, count + 1, string)
            elif isinstance(value, dict) is False and count == level:
                string[-1] += ":\t" + str(value)
            else:
                string.append(prefix + indentor * (count + 1) + str(value))
        return string

    return "\n".join(dict_to_string_list(dictionary, indentor))


class TimeRecorder:
    """A class for recording time."""

    def __init__(self):
        self.start = time.monotonic()
        self.records = {}

    def get_elapsed_time(self, event=None, strf=True):
        """Get elapsed time since the start of the recorder.

        Parameters
        ----------
        event : str, optional
            Record event, by default None.
        strf : bool, optional
            Convert seconds to string format, by default True.

        Returns
        -------
        float or str
        Elapsed time or string format of the elapsed time.
        """
        elapsed_time = time.monotonic() - self.start
        if strf:
            elapsed_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
        if event is not None:
            self.records[event] = elapsed_time
        return elapsed_time

    @staticmethod
    def sec2str(secs, fmt="%H:%M:%S"):
        """Convert seconds to string format.

        Parameters
        ----------
        secs : int
            Seconds.
        fmt : str, optional
            Format, by default "%H:%M:%S".

        Returns
        -------
        str
            A string.
        """
        return time.strftime(fmt, time.gmtime(secs))


# Indicator module
# adopted from Chung-Yi Lin @ Lehigh University (philip928lin@gmail.com)
# Last update at 2021/12/23.
class Indicator:
    """A class for calculating indicators."""

    def __init__(self) -> None:
        """Initialize an Indicator class to manage various hydrological calculations.

        r   : Correlation of correlation
        r2  : Coefficient of determination
        rmse: Root mean square error
        nse : Nash-Sutcliffe efficiency
        inse: nse with inverse transformed Q.
        cp  : Correlation of persistence
        rsr : RMSE-observations standard deviation ratio
        kge : Kling-Gupta efficiency
        ikge: kge with inverse transformed Q.

        Note
        ----
        The code is adopted from HydroCNHS (Lin et al., 2022).
        Lin, C. Y., Yang, Y. C. E., & Wi, S. (2022). HydroCNHS: A Python Package of
        Hydrological Model for Coupled Natural-Human Systems. Journal of Water
        Resources Planning and Management, 148(12), 06022005.
        """
        pass

    @staticmethod
    def remove_na(x_obv, y_sim):
        """Remove nan in x_obv and y_sim.

        This function makes sure there is no nan involves in the indicator
        calculation. If nan is detected, data points will be remove from x_obv
        and y_sim simultaneously.

        Parameters
        ----------
        x_obv : array
            Observation data.
        y_sim : array
            Simulation data.

        Returns
        -------
        tuple
            Updated (x_obv, y_sim)
        """
        x_obv = np.array(x_obv)
        y_sim = np.array(y_sim)
        index = [
            True if np.isnan(x) is False and np.isnan(y) is False else False
            for x, y in zip(x_obv, y_sim, strict=True)
        ]
        x_obv = x_obv[index]
        y_sim = y_sim[index]
        # print("Usable data ratio = {}/{}.".format(len(index), len(x_obv)))
        return x_obv, y_sim

    @staticmethod
    def cal_indicator_df(
        x_obv, y_sim, index_name="value", indicators_list=None, r_na=True
    ):
        """Calculate indicators and return as a DataFrame.

        Parameters
        ----------
        x_obv : array
            Observation data.
        y_sim : array
            Simulation data.
        index_name : str, optional
            Index name, by default "value".
        indicators_list : list, optional
            List of indicators, by default None.
        r_na : bool, optional
            Remove nan, by default True.

        Returns
        -------
        DataFrame
            A DataFrame of indicators.
        """
        if r_na:
            x_obv, y_sim = Indicator.remove_na(x_obv, y_sim)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            indicators_dict = {
                "r": Indicator.get_r(x_obv, y_sim, False),
                "r2": Indicator.get_r2(x_obv, y_sim, False),
                "rmse": Indicator.get_rmse(x_obv, y_sim, False),
                "nse": Indicator.get_nse(x_obv, y_sim, False),
                "inse": Indicator.get_inse(x_obv, y_sim, False),
                "kge": Indicator.get_kge(x_obv, y_sim, False),
                "ikge": Indicator.get_ikge(x_obv, y_sim, False),
                "cp": Indicator.get_cp(x_obv, y_sim, False),
                "rsr": Indicator.get_rsr(x_obv, y_sim, False),
            }
            df = pd.DataFrame(indicators_dict, index=[index_name])
            if indicators_list is None:
                return df
            else:
                return df.loc[:, indicators_list]

    @staticmethod
    def get_r(x_obv, y_sim, r_na=True):
        """Correlation.

        Parameters
        ----------
        x_obv : array
            Observation data.
        y_sim : array
            Simulation data.
        r_na : bool, optional
            Remove nan, by default True

        Returns
        -------
        float
            r coefficient.
        """
        if r_na:
            x_obv, y_sim = Indicator.remove_na(x_obv, y_sim)
        r = np.corrcoef(x_obv, y_sim)[0, 1]
        if np.isnan(r):
            # We don't consider 2 identical horizontal line as r = 1!
            r = 0
        return r

    @staticmethod
    def get_r2(x_obv, y_sim, r_na=True):
        """Coefficient of determination.

        Parameters
        ----------
        x_obv : array
            Observation data.
        y_sim : array
            Simulation data.
        r_na : bool, optional
            Remove nan, by default True

        Returns
        -------
        float
            r2 coefficient.
        """
        r = Indicator.get_r(x_obv, y_sim, r_na)
        return r**2

    @staticmethod
    def get_rmse(x_obv, y_sim, r_na=False):
        """Root mean square error.

        Parameters
        ----------
        x_obv : array
            Observation data.
        y_sim : array
            Simulation data.
        r_na : bool, optional
            Remove nan, by default True

        Returns
        -------
        float
            Root mean square error.
        """
        if r_na:
            x_obv, y_sim = Indicator.remove_na(x_obv, y_sim)
        return np.nanmean((x_obv - y_sim) ** 2) ** 0.5

    @staticmethod
    def get_nse(x_obv, y_sim, r_na=False):
        """Nash-Sutcliffe efficiency.

        Parameters
        ----------
        x_obv : array
            Observation data.
        y_sim : array
            Simulation data.
        r_na : bool, optional
            Remove nan, by default True

        Returns
        -------
        float
            Nash-Sutcliffe efficiency.
        """
        if r_na:
            x_obv, y_sim = Indicator.remove_na(x_obv, y_sim)

        mu_xObv = np.nanmean(x_obv)
        return 1 - np.nansum((y_sim - x_obv) ** 2) / np.nansum((x_obv - mu_xObv) ** 2)

    @staticmethod
    def get_inse(x_obv, y_sim, r_na=False):
        """Inverse Nash-Sutcliffe efficiency.

        Parameters
        ----------
        x_obv : array
            Observation data.
        y_sim : array
            Simulation data.
        r_na : bool, optional
            Remove nan, by default True

        Returns
        -------
        float
            Inverse Nash-Sutcliffe efficiency.
        """
        if r_na:
            x_obv, y_sim = Indicator.remove_na(x_obv, y_sim)
        # Prevent dividing zero.
        if np.nanmean(x_obv) == 0:
            x_obv = 1 / (x_obv + 0.0000001)
        else:
            x_obv = 1 / (x_obv + 0.01 * np.nanmean(x_obv))

        if np.nanmean(y_sim) == 0:
            y_sim = 1 / (y_sim + 0.0000001)
        else:
            y_sim = 1 / (y_sim + 0.01 * np.nanmean(y_sim))

        mu_xObv = np.nanmean(x_obv)
        return 1 - np.nansum((y_sim - x_obv) ** 2) / np.nansum((x_obv - mu_xObv) ** 2)

    @staticmethod
    def get_cp(x_obv, y_sim, r_na=False):
        """Correlation of persistence.

        Parameters
        ----------
        x_obv : array
            Observation data.
        y_sim : array
            Simulation data.
        r_na : bool, optional
            Remove nan, by default True

        Returns
        -------
        float
            Correlation of persistence.
        """
        if r_na:
            x_obv, y_sim = Indicator.remove_na(x_obv, y_sim)
        a = np.nansum((x_obv[1:] - x_obv[:-1]) ** 2)
        if a == 0:
            a = 0.0000001
        return 1 - np.nansum((x_obv[1:] - y_sim[1:]) ** 2) / a

    @staticmethod
    def get_rsr(x_obv, y_sim, r_na=False):
        """RMSE-observations standard deviation ratio.

        Parameters
        ----------
        x_obv : array
            Observation data.
        y_sim : array
            Simulation data.
        r_na : bool, optional
            Remove nan, by default True

        Returns
        -------
        float
            RMSE-observations standard deviation ratio.
        """
        if r_na:
            x_obv, y_sim = Indicator.remove_na(x_obv, y_sim)
        sig_xObv = np.nanstd(x_obv)
        return Indicator.get_rmse(x_obv, y_sim) / sig_xObv

    @staticmethod
    def get_kge(x_obv, y_sim, r_na=True):
        """Kling-Gupta efficiency.

        Parameters
        ----------
        x_obv : array
            Observation data.
        y_sim : array
            Simulation data.
        r_na : bool, optional
            Remove nan, by default True

        Returns
        -------
        float
            Kling-Gupta efficiency.
        """
        if r_na:
            x_obv, y_sim = Indicator.remove_na(x_obv, y_sim)

        mu_ySim = np.nanmean(y_sim)
        mu_xObv = np.nanmean(x_obv)
        sig_ySim = np.nanstd(y_sim)
        sig_xObv = np.nanstd(x_obv)
        kge = (
            1
            - (
                (Indicator.get_r(x_obv, y_sim, False) - 1) ** 2
                + (sig_ySim / sig_xObv - 1) ** 2
                + (mu_ySim / mu_xObv - 1) ** 2
            )
            ** 0.5
        )
        return kge

    @staticmethod
    def get_ikge(x_obv, y_sim, r_na=True):
        """Inverse Kling-Gupta efficiency.

        Parameters
        ----------
        x_obv : array
            Observation data.
        y_sim : array
            Simulation data.
        r_na : bool, optional
            Remove nan, by default True

        Returns
        -------
        float
            Inverse Kling-Gupta efficiency.
        """
        if r_na:
            x_obv, y_sim = Indicator.remove_na(x_obv, y_sim)

        # Prevent dividing zero.
        if np.nanmean(x_obv) == 0:
            x_obv = 1 / (x_obv + 0.0000001)
        else:
            x_obv = 1 / (x_obv + 0.01 * np.nanmean(x_obv))

        if np.nanmean(y_sim) == 0:
            y_sim = 1 / (y_sim + 0.0000001)
        else:
            y_sim = 1 / (y_sim + 0.01 * np.nanmean(y_sim))

        mu_ySim = np.nanmean(y_sim)
        mu_xObv = np.nanmean(x_obv)
        sig_ySim = np.nanstd(y_sim)
        sig_xObv = np.nanstd(x_obv)
        ikge = (
            1
            - (
                (Indicator.get_r(x_obv, y_sim, False) - 1) ** 2
                + (sig_ySim / sig_xObv - 1) ** 2
                + (mu_ySim / mu_xObv - 1) ** 2
            )
            ** 0.5
        )
        return ikge
