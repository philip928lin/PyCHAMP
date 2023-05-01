r"""
The code is developed by Chung-Yi Lin at Virginia Tech, in April 2023.
Email: chungyi@vt.edu
Last modified on May 1, 2023

WARNING: This code is not yet published, please do not distributed the code
without permission.

To do:
    Complete et calculation
"""
import numpy as np
from dotmap import DotMap
from ..util import cal_pet_Hamon

class Field():
    """
    A field simulator.

    Attributes
    ----------
    field_id : str or int
        Field id.
    config : dict or DotMap
        General info of the model.
    te : str, optional
        Irrigation technology. The default is None. The default is "center
        pivot".
    lat : float, optional
        Latitude of the field [deg]. This will be used in calculating
        evapotranspiration. The default is 39.4.
    dz : float, optional
        Gauge height adjustment. This will be used in calculating
        evapotranspiration. The default is None.
    crop_options : list, optional
        A list of crop type options. They must exist in the config. The
        default is ["corn", "sorghum", "soybean", "fallow"].
    tech_options : list, optional
        A list of irrigation technologies. They must exist in the config. The
        default is ["center pivot", "center pivot LEPA"].
    """

    def __init__(self, field_id, config, te="center pivot", lat=39.4, dz=None,
                 crop_options=["corn", "sorghum", "soybean", "fallow"],
                 tech_options=["center pivot", "center pivot LEPA"]):
        # for name_, value_ in vars().items():
        #     if name_ != 'self' and name_ != 'config':
        #         setattr(self, name_, value_)
        self.field_id = field_id
        self.lat = lat
        self.dz = dz
        config = DotMap(config)
        crop = np.array([config.field.crop[c] for c in crop_options])
        self.ymax = crop[:, 0].reshape((-1, 1))     # (n_c, 1)
        self.wmax = crop[:, 1].reshape((-1, 1))     # (n_c, 1)
        self.a = crop[:, 2].reshape((-1, 1))        # (n_c, 1)
        self.b = crop[:, 3].reshape((-1, 1))        # (n_c, 1)
        self.c = crop[:, 4].reshape((-1, 1))        # (n_c, 1)
        self.n_s = config.field.area_split
        self.unit_area = config.field.field_area/self.n_s

        self.tech_options = tech_options
        self.techs = config.field.tech
        self.update_irr_tech(te)

        self.cal_pet_Hamon = cal_pet_Hamon

    def update_irr_tech(self, i_te):
        """
        Update the irri

        Parameters
        ----------
        i_te : 1darray
            An array outputted from OptModel() that represent the indicator
            matrix for the irrigation technology choices in following year. The
            dimension of the array should be (n_te).

        Returns
        -------
        None.

        """
        if isinstance(i_te, str):
            new_te = i_te
        else:
            new_te = self.tech_options[np.where(i_te==1)[0][0]]
        self.a_te, self.b_te, self.l_pr = self.techs[new_te]
        self.te = new_te

    def sim_step(self, irr, i_area, i_te, prec, temp):
        """
        Simulate a single timestep.

        Parameters
        ----------
        irr : 3darray
            An array outputted from OptModel() that represent the irrigation
            depth for the following year. The dimension of the array should be
            (n_s, n_c, 1).
        i_area : 3darray
            An array outputted from OptModel() that represent the indicator
            matrix for the crop choices in following year. The dimension of the
            array should be (n_s, n_c, 1).
        i_te : 1darray
            An array outputted from OptModel() that represent the indicator
            matrix for the irrigation technology choices in following year. The
            dimension of the array should be (n_te).
        prec : float
            The annual precipitation amount.
        temp : DataFrame
            The daily mean temperature storing in DataFrame format with
            datetime index.

        Returns
        -------
        y : 3darray
            Crop yield with the dimension (n_s, n_c, 1).
        y_y : 3darray
            Crop yield/maximum yield with the dimension (n_s, n_c, 1).
        v : float
            Irrigation amount.
        inflow : float
            Inflow to the aquifer.

        """
        # Crop yield
        a = self.a
        b = self.b
        c = self.c
        ymax = self.ymax
        wmax = self.wmax
        n_s = self.n_s
        unit_area = self.unit_area

        w = irr + prec
        w = w * i_area
        w_ = w/wmax
        y_ = (a * w_**2 + b * w_ + c)
        # y_ = yw_ * i_area
        y = y_ * ymax
        v_c = irr * unit_area
        v = np.sum(v_c)
        y_y = np.sum(y_) / n_s

        self.y, self.y_y, self.v = y, y_y, v
        # Can be deleted in the future
        #assert w_ >= 0 and w_ <= 1, f"w_ in [0, 1] expected, got: {w_}"
        #assert y_ >= 0 and y_ <= 1, f"y_ in [0, 1]  expected, got: {y_}"

        # Tech
        self.update_irr_tech(i_te)  # update tech
        a_te = self.a_te
        b_te = self.b_te
        q = a_te * v + b_te
        self.q = q

        # Annual ET for aquifer
        # !!!! Calculate et + remember to average over the corner
        wv = np.sum(w * unit_area)
        pet = self.cal_pet_Hamon(temp, self.lat, self.dz)
        Kc = 1 # not yet decide how

        # Adopt the stress coeficient from GWLF
        Ks = np.ones(w_.shape)
        Ks[w_ <= 0.5] = 2 * w_[w_ <= 0.5]

        et = Ks * Kc * sum(pet) * unit_area
        inflow = wv - et    # volumn
        self.inflow = inflow
        return y, y_y, v, inflow
