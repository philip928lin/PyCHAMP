# -*- coding: utf-8 -*-
"""
The code is developed by Chung-Yi Lin at Virginia Tech, in April 2023.
Email: chungyi@vt.edu
Last modified on May 1, 2023

WARNING: This code is not yet published, please do not distributed the code
without permission.
"""
import time
import numpy as np
from pandas import to_numeric
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

def dict_to_string(dictionary, prefix="", indentor="  ", level=2):
    """Ture a dictionary into a printable string.
    Parameters
    ----------
    dictionary : dict
        A dictionary.
    indentor : str, optional
        Indentor, by default "  ".

    Returns
    -------
    str
        A printable string.
    """
    def dict_to_string_list(dictionary, indentor="  ", count=1, string=[]):
        for key, value in dictionary.items():
            string.append(prefix + indentor * count + str(key))
            if isinstance(value, dict) and count < level:
                string = dict_to_string_list(value, indentor, count+1, string)
            elif isinstance(value, dict) is False and count == level:
                string[-1] += ":\t" + str(value)
            else:
                string.append(prefix + indentor * (count+1) + str(value))
        return string
    return "\n".join(dict_to_string_list(dictionary, indentor))


def cal_pet_Hamon(temp, lat, dz=None):
    """Calculate potential evapotranspiration (pet) with Hamon (1961) equation.

    Parameters
    ----------
    temp : numpy.ndarray
        Daily mean temperature [degC].
    lat : float
        Latitude [deg].
    dz : float, optional
        Altitude temperature adjustment [m], by default None.

    Returns
    -------
    numpy.ndarray
        Potential evapotranspiration [cm/day]

    Note
    ----
    The code is adopted from HydroCNHS (Lin et al., 2022).
    Lin, C. Y., Yang, Y. C. E., & Wi, S. (2022). HydroCNHS: A Python Package of
    Hydrological Model for Coupled Natural–Human Systems. Journal of Water
    Resources Planning and Management, 148(12), 06022005.
    """
    pdDatedateIndex = temp.index
    temp = temp.values.flatten()
    # Altitude temperature adjustment
    if dz is not None:
        # Assume temperature decrease 0.6 degC for every 100 m elevation.
        tlaps = 0.6
        temp = temp - tlaps*dz/100
    # Calculate Julian days
    # data_length = len(temp)
    # start_date = to_datetime(start_date, format="%Y/%m/%d")
    # pdDatedateIndex = date_range(start=start_date, periods=data_length,
    #                              freq="D")
    JDay = to_numeric(pdDatedateIndex.strftime('%j')) # convert to Julian days
    # Calculate solar declination [rad] from day of year (JDay) based on
    # equations 24 in ALLen et al (1998).
    sol_dec = 0.4093 * np.sin(2. * 3.141592654 / 365. * JDay - 1.39)
    lat_rad = lat*np.pi/180
    # Calculate sunset hour angle from latitude and solar declination [rad]
    # based on equations 25 in ALLen et al (1998).
    omega = np.arccos(-np.tan(sol_dec) * np.tan(lat_rad))
    # Calculate maximum possible daylight length [hr]
    dl = 24 / np.pi * omega
    # From Prudhomme(hess, 2013)
    # https://hess.copernicus.org/articles/17/1365/2013/hess-17-1365-2013-supplement.pdf
    # Slightly different from what we used to.
    pet = (dl / 12) ** 2 * np.exp(temp / 16)
    pet = np.array(pet/10)         # Convert from mm to cm
    pet[np.where(temp <= 0)] = 0   # Force pet = 0 when temperature is below 0.
    return pet      # [cm/day]

class TimeRecorder():
    def __init__(self):
        self.start = time.monotonic()
        self.records = {}
    def get_elapsed_time(self, event=None, strf=True):
        elapsed_time = time.monotonic() - self.start
        if strf:
            elapsed_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
        if event is not None:
            self.records[event] = elapsed_time
        return elapsed_time
    @staticmethod
    def sec2str(secs, fmt="%H:%M:%S"):
        return time.strftime(fmt, time.gmtime(secs))


# Indicator module ( adopt)
# by Chung-Yi Lin @ Lehigh University (philip928lin@gmail.com)
# Last update at 2021/12/23.
class Indicator(object):
    def __init__(self) -> None:
        """A class containing following indicator functions.

        r   : Correlation of correlation
        r2  : Coefficient of determination
        rmse: Root mean square error
        NSE : Nash–Sutcliffe efficiency
        iNSE: NSE with inverse transformed Q.
        CP  : Correlation of persistence
        RSR : RMSE-observations standard deviation ratio
        KGE : Kling–Gupta efficiency
        iKGE: KGE with inverse transformed Q.

        Note
        ----
        The code is adopted from HydroCNHS (Lin et al., 2022).
        Lin, C. Y., Yang, Y. C. E., & Wi, S. (2022). HydroCNHS: A Python Package of
        Hydrological Model for Coupled Natural–Human Systems. Journal of Water
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
        index = [True if np.isnan(x) == False and np.isnan(y) == False \
                    else False for x, y in zip(x_obv, y_sim)]
        x_obv = x_obv[index]
        y_sim = y_sim[index]
        print("Usable data ratio = {}/{}.".format(len(index), len(x_obv)))
        return x_obv, y_sim

    @staticmethod
    def cal_indicator_df(x_obv, y_sim, index_name="value",
                         indicators_list=None, r_na=True):
        if r_na:
            x_obv, y_sim = Indicator.remove_na(x_obv, y_sim)
        dict = {"r"   : Indicator.r(x_obv, y_sim, False),
                "r2"  : Indicator.r2(x_obv, y_sim, False),
                "rmse": Indicator.rmse(x_obv, y_sim, False),
                "NSE" : Indicator.NSE(x_obv, y_sim, False),
                "iNSE": Indicator.iNSE(x_obv, y_sim, False),
                "KGE" : Indicator.KGE(x_obv, y_sim, False),
                "iKGE": Indicator.iKGE(x_obv, y_sim, False),
                "CP"  : Indicator.CP(x_obv, y_sim, False),
                "RSR" : Indicator.RSR(x_obv, y_sim, False)}
        df = pd.DataFrame(dict, index=[index_name])
        if indicators_list is None:
            return df
        else:
            return df.loc[:, indicators_list]

    @staticmethod
    def r(x_obv, y_sim, r_na=True):
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
        r = np.corrcoef(x_obv, y_sim)[0,1]
        if np.isnan(r):
            # We don't consider 2 identical horizontal line as r = 1!
            r = 0
        return r

    @staticmethod
    def r2(x_obv, y_sim, r_na=True):
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
        r = Indicator.r(x_obv, y_sim, r_na)
        return r**2

    @staticmethod
    def rmse(x_obv, y_sim, r_na=False):
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
        return np.nanmean((x_obv - y_sim)**2)**0.5

    @staticmethod
    def NSE(x_obv, y_sim, r_na=False):
        """Nash–Sutcliffe efficiency.

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
            Nash–Sutcliffe efficiency.
        """
        if r_na:
            x_obv, y_sim = Indicator.remove_na(x_obv, y_sim)
        mu_xObv = np.nanmean(x_obv)
        return 1 - np.nansum((y_sim-x_obv)**2) / np.nansum((x_obv-mu_xObv)**2)

    @staticmethod
    def iNSE(x_obv, y_sim, r_na=False):
        """Inverse Nash–Sutcliffe efficiency.

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
            Inverse Nash–Sutcliffe efficiency.
        """
        if r_na:
            x_obv, y_sim = Indicator.remove_na(x_obv, y_sim)
        # Prevent dividing zero.
        if np.nanmean(x_obv) == 0:
            x_obv = 1 / (x_obv + 0.0000001)
        else:
            x_obv = 1 / (x_obv + 0.01*np.nanmean(x_obv))

        if np.nanmean(y_sim) == 0:
            y_sim = 1 / (y_sim + 0.0000001)
        else:
            y_sim = 1 / (y_sim + 0.01*np.nanmean(y_sim))
        mu_xObv = np.nanmean(x_obv)
        return 1 - np.nansum((y_sim-x_obv)**2) / np.nansum((x_obv-mu_xObv)**2)

    @staticmethod
    def CP(x_obv, y_sim, r_na=False):
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
        a = np.nansum((x_obv[1:] - x_obv[:-1])**2)
        if a == 0:
            a = 0.0000001
        return 1 - np.nansum((x_obv[1:] - y_sim[1:])**2) / a

    @staticmethod
    def RSR(x_obv, y_sim, r_na=False):
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
        return Indicator.rmse(x_obv, y_sim) / sig_xObv

    @staticmethod
    def KGE(x_obv, y_sim, r_na=True):
        """Kling–Gupta efficiency.

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
            Kling–Gupta efficiency.
        """
        if r_na:
            x_obv, y_sim = Indicator.remove_na(x_obv, y_sim)

        mu_ySim = np.nanmean(y_sim); mu_xObv = np.nanmean(x_obv)
        sig_ySim = np.nanstd(y_sim); sig_xObv = np.nanstd(x_obv)
        kge = 1 - ((Indicator.r(x_obv, y_sim, False) - 1)**2
                    + (sig_ySim/sig_xObv - 1)**2
                    + (mu_ySim/mu_xObv - 1)**2)**0.5
        return kge

    @staticmethod
    def iKGE(x_obv, y_sim, r_na=True):
        """Inverse Kling–Gupta efficiency.

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
            Inverse Kling–Gupta efficiency.
        """
        if r_na:
            x_obv, y_sim = Indicator.remove_na(x_obv, y_sim)

        # Prevent dividing zero.
        if np.nanmean(x_obv) == 0:
            x_obv = 1/(x_obv + 0.0000001)
        else:
            x_obv = 1/(x_obv + 0.01*np.nanmean(x_obv))

        if np.nanmean(y_sim) == 0:
            y_sim = 1/(y_sim + 0.0000001)
        else:
            y_sim = 1/(y_sim + 0.01*np.nanmean(y_sim))

        mu_ySim = np.nanmean(y_sim); mu_xObv = np.nanmean(x_obv)
        sig_ySim = np.nanstd(y_sim); sig_xObv = np.nanstd(x_obv)
        ikge = 1 - ((Indicator.r(x_obv, y_sim, False) - 1)**2
                    + (sig_ySim/sig_xObv - 1)**2
                    + (mu_ySim/mu_xObv - 1)**2)**0.5
        return ikge


class Visual():
    """Collection of some plotting functions.
    Note
    ----
    The code is adopted from HydroCNHS (Lin et al., 2022).
    Lin, C. Y., Yang, Y. C. E., & Wi, S. (2022). HydroCNHS: A Python Package of
    Hydrological Model for Coupled Natural–Human Systems. Journal of Water
    Resources Planning and Management, 148(12), 06022005.
    """
    @staticmethod
    def plot_reg(x_obv, y_sim, title=None, xy_labal=None, same_xy_limit=True,
                 return_reg_par=False, save_fig_path=None, show=True):
        """Plot regression.

        Parameters
        ----------
        x_obv : array
            Observation data.
        y_sim : array
            Simulation data.
        title : str, optional
            Title, by default None.
        xy_labal : list, optional
            List of x and y labels, by default None.
        same_xy_limit : bool, optional
            If True same limit will be applied to x and y axis, by default True.
        return_reg_par : bool, optional
            If True, slope and interception will be return, by default False.
        save_fig_path : str, optional
            If given, plot will be save as .png, by default None.
        show : bool, optional
            If True, the plot will be shown in the console, by default True.

        Returns
        -------
        ax or list
            axis object or [slope, intercept].
        """
        if title is None:
            title = "Regression"
        else:
            title = title

        if xy_labal is None:
            x_label = "Obv"; y_label = "Sim"
        else:
            x_label = xy_labal[0]; y_label = xy_labal[1]

        # Create figure
        fig, ax = plt.subplots()
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

        # Regression calculation and plot
        x_obv = np.array(x_obv)
        y_sim = np.array(y_sim)
        mask = ~np.isnan(x_obv) & ~np.isnan(y_sim)  # Mask to ignore nan
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            x_obv[mask], y_sim[mask]) # Calculate the regression line
        line = slope*x_obv+intercept  # For plotting regression line
        ax.plot(x_obv, line, 'r', label='y={:.2f}x+{:.2f}'.format(slope,
                                                                  intercept))

        # Plot data point
        ax.scatter(x_obv, y_sim, color="k", s=3.5)
        ax.legend(fontsize=9, loc='upper right')
        if same_xy_limit:
            Max = max([np.nanmax(x_obv), np.nanmax(y_sim)])
            Min = min([np.nanmin(x_obv), np.nanmin(y_sim)])
            ax.set_xlim(Min, Max)
            ax.set_ylim(Min, Max)
            # Add 45 degree line
            interval = (Max - Min) / 10
            diagonal = np.arange(Min, Max+interval, interval)
            ax.plot(diagonal, diagonal, "b", linestyle='dashed', lw=1)


        # PLot indicators
        name = {"r": "$r$",
                "r2":"$r^2$",
                "rmse":"RMSE",
                "NSE": "NSE",
                "CP": "CP",
                "RSR": "RSR",
                "KGE": "KGE"}
        indicators = {}
        indicators["r"] = Indicator.r(x_obv, y_sim)
        indicators["r2"] = Indicator.r2(x_obv, y_sim)
        indicators["rmse"] = Indicator.rmse(x_obv, y_sim)
        indicators["NSE"] = Indicator.NSE(x_obv, y_sim)
        indicators["KGE"] = Indicator.KGE(x_obv, y_sim)

        string = "\n".join(['{:^4}: {}'.format(name[keys], round(values,5))
                            for keys,values in indicators.items()])

        # these are matplotlib.patch.Patch properties
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.85)
        ax.annotate(string, xy=(0.05, 0.95), xycoords='axes fraction',
                    verticalalignment='top', horizontalalignment='left',
                    transform=ax.transAxes, fontsize=9, bbox=props)
        if show:
            plt.show()

        if save_fig_path is not None:
            fig.savefig(save_fig_path)
            plt.close()

        if return_reg_par:
            return [slope, intercept]
        else:
            return ax

    @staticmethod
    def plot_timeseries(x_obv, y_sim, xticks=None, title=None, xy_labal=None,
                       save_fig_path=None, legend=True, show=True, **kwargs):
        """Plot timeseries.

        This function can plot two DataFrames with same column names.

        Parameters
        ----------
        x_obv : array/DataFrame
            Observation data.
        y_sim : array/DataFrame
            Simulation data.
        xticks : list, optional
            Ticks for x-axis, by default None.
        title : str, optional
            Title, by default None.
        xy_labal : list, optional
            List of x and y labels, by default None.
        save_fig_path : str, optional
            If given, plot will be save as .png, by default None.
        legend : bool, optional
            If True, plot legend, by default None.
        show : bool, optional
            If True, the plot will be shown in the console, by default True.
        kwargs : optional
            Other keywords for matplotlib.
        Returns
        -------
        object
            axis object.
        """
        if title is None:
            title = "Timeseries"
        else:
            title = title

        if xy_labal is None:
            x_label = "Obv"; y_label = "Sim"
        else:
            x_label = xy_labal[0]; y_label = xy_labal[1]

        if xticks is None:
            if isinstance(x_obv, pd.DataFrame):
                xticks = x_obv.index
            else:
                xticks = np.arange(0,len(x_obv))
        else:
            assert len(xticks) == len(x_obv), print(
                "Input length of x is not corresponding to the length of data."
                )
        fig, ax = plt.subplots()
        if isinstance(x_obv, pd.DataFrame):
            for i, c in enumerate(list(x_obv)):
                ax.plot(xticks, x_obv[c],
                        label = x_label +"_"+ str(c),
                        color = "C{}".format(i%10),
                        **kwargs)
        else:
            ax.plot(xticks, x_obv, label = x_label, **kwargs)

        if isinstance(y_sim, pd.DataFrame):
            for i, c in enumerate(list(y_sim)):
                ax.plot(xticks, y_sim[c], linestyle='dashed',
                        label = y_label + "_"+ str(c),
                        color = "C{}".format(i%10), alpha = 0.5,
                        **kwargs)
        else:
            ax.plot(xticks, y_sim, linestyle='dashed', label = y_label, **kwargs)
        if legend:
            ax.legend(fontsize=9)
        ax.set_title(title)
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        if show:
            plt.show()

        if save_fig_path is not None:
            fig.savefig(save_fig_path)
            plt.close()

        return ax

    @staticmethod
    def plot_simple_ts(df, title=None, xy_labal=None, data_dots=True,
                     save_fig_path=None, **kwargs):
        """Plot timeseries.

        Parameters
        ----------
        df : DataFrame
            Dataframe.
        title : str, optional
            Title, by default None.
        xy_labal : list, optional
            List of x and y labels, by default None.
        data_dots : bool, optional
            If Ture, show data marker, by default True.
        save_fig_path : str, optional
            If given, plot will be save as .png, by default None.

        Returns
        -------
        object
            axis object.
        """
        if title is None:
            title = ""
        else:
            title = title

        if xy_labal is None:
            x_label = "Time"; y_label = "Value"
        else:
            x_label = xy_labal[0]; y_label = xy_labal[1]

        # Create figure
        fig, ax = plt.subplots()
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

        # Regression calculation and plot
        x = np.arange(1, len(df)+1)
        for i, v in enumerate(df):
            mask = ~np.isnan(df[v])   # Mask to ignore nan
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                x[mask], df[v][mask]) # Calculate the regression line
            line = slope*x+intercept  # For plotting regression line
            ax.plot(df.index, line, color="C{}".format(i%10),
                    label='y={:.2f}x+{:.2f}'.format(slope, intercept),
                    linestyle="dashed", **kwargs)
            if data_dots:
                df[[v]].plot(ax=ax, marker='o', ls='',
                             color="C{}".format(i%10), ms=2, alpha=0.6)
            else:
                df[[v]].plot(ax=ax, color="C{}".format(i%10), alpha=0.5)
        ax.legend()
        plt.show()

        if save_fig_path is not None:
            fig.savefig(save_fig_path)
            plt.close()

        return ax



