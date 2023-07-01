r"""
The code is developed by Chung-Yi Lin at Virginia Tech, in April 2023.
Email: chungyi@vt.edu
Last modified on Jun 9, 2023

WARNING: This code is not yet published, please do not distributed the code
without permission.
"""
import numpy as np

class Field():
    """
    A field simulator.

    Attributes
    ----------
    field_id : str or int
        Field id.
    config : dict
        General configuration of the model.
    te : str
        Initial irrigation technology. The default is None. The default is "center
        pivot".
    crop : str or list
        Initial crop. If given a string, it will apply to all area splits. A
        list of crop can be given to populate initial crop for each area split.
    crop_options : list, optional
        A list of crop type options. They must exist in the config. The
        default is ["corn", "sorghum", "soybeans", "fallow"].
    tech_options : list, optional
        A list of irrigation technologies. They must exist in the config. The
        default is ["center pivot", "center pivot LEPA"].
    """
    # aquifer_id, lat, dz => They are for dynamic inflow calculation (deprecated)
    def __init__(self, field_id, config, te, crop, crop_options, tech_options,
                 **kwargs):

        self.field_id = field_id
        self.crop_options = crop_options
        self.tech_options = tech_options

        self.load_config(config)

        # Init tech
        self.te = te    # serve as the tech in the previous year
        self.update_irr_tech(te)

        # Init crop
        i_crop = np.zeros((self.n_s, self.n_c, 1))
        if isinstance(crop, str):
            i_c = self.crop_options.index(crop)
            i_crop[:, i_c, 0] = 1
        else:
            for s, c in enumerate(crop):
                i_c = self.crop_options.index(c)
                i_crop[s, i_c, 0] = 1

        self.i_crop = i_crop
        self.update_crops(i_crop)

        # Load other kwargs
        for k, v in kwargs.items():
            setattr(self, k, v)

        # Containers
        self.t = 0
        self.crops = []
        self.irr_vol = None

    def load_config(self, config):
        """
        Load config.

        Parameters
        ----------
        config : dict
            General configuration of the model.

        Returns
        -------
        None.

        """

        crop_options = self.crop_options
        config_field = config["field"]
        crop_par = np.array([config_field["crop"][c] for c in crop_options])
        self.ymax = crop_par[:, 0].reshape((-1, 1))     # (n_c, 1)
        self.wmax = crop_par[:, 1].reshape((-1, 1))     # (n_c, 1)
        self.a = crop_par[:, 2].reshape((-1, 1))        # (n_c, 1)
        self.b = crop_par[:, 3].reshape((-1, 1))        # (n_c, 1)
        self.c = crop_par[:, 4].reshape((-1, 1))        # (n_c, 1)
        # self.growing_season = {c: config_field["growing_season"][c] \
        #                           for c in crop_options if c != 'fallow'}
        self.n_s = config_field['area_split']
        self.n_c = len(crop_options)
        self.field_area = config_field["field_area"]
        self.unit_area = self.field_area/self.n_s
        self.tech_par = config_field["tech"]

    def update_irr_tech(self, i_te):
        """
        Update the irrigation technology.

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
            # Use argmax instead of ==1 to avoid float numerical issue
            new_te = self.tech_options[np.argmax(i_te)]
        self.a_te, self.b_te, self.l_pr = self.tech_par[new_te]
        self.pre_te = self.te
        self.te = new_te

    def update_crops(self, i_crop):
        """
        Update the crop for each area split.

        Parameters
        ----------
        i_crop : 3darray
            An array outputted from OptModel() that represent the indicator
            matrix for the crop choices in following year. The dimension of the
            array should be (n_s, n_c, 1).

        Returns
        -------
        None.

        """
        n_s = self.n_s
        crop_options = self.crop_options
        # Use argmax instead of ==1 to avoid float numerical issue
        self.pre_i_crop = self.i_crop
        crops = [crop_options[np.argmax(i_crop[s, :, 0])] for s in range(n_s)]
        self.crops = crops
        self.i_crop = i_crop

    def step(self, irr, i_crop, i_te, prec_aw):
        """
        Simulate a single timestep.

        Parameters
        ----------
        irr : 3darray
            An array outputted from OptModel() that represent the irrigation
            depth for the following year [cm]. The dimension of the array
            should be (n_s, n_c, 1).
        i_crop : 3darray
            An array outputted from OptModel() that represent the indicator
            matrix for the crop choices in following year. The dimension of the
            array should be (n_s, n_c, 1).
        i_te : 1darray
            An array outputted from OptModel() that represent the indicator
            matrix for the irrigation technology choices in following year. The
            dimension of the array should be (n_te).
        prec_aw : float
            The precipitation in the growing season [cm].

        Returns
        -------
        y : 3darray
            Crop yield with the dimension (n_s, n_c, 1) [1e4 bu].
        y_y : 3darray
            Crop yield/maximum yield with the dimension (n_s, n_c, 1).
        v : float
            Irrigation amount [m-ha].
        inflow : float
            Inflow to the aquifer [m-ha].

        """
        self.t +=1

        a = self.a
        b = self.b
        c = self.c
        ymax = self.ymax
        wmax = self.wmax
        n_s = self.n_s
        unit_area = self.unit_area

        ### Yield calculation
        self.update_crops(i_crop)   # update pre_i_crop
        irr = irr.copy()[:,:,[0]]
        prec_aw_ = np.ones(irr.shape)
        for ci, crop in enumerate(self.crop_options):
            prec_aw_[:, ci, :] = prec_aw[crop]

        w = irr + prec_aw_
        w = w * i_crop
        w_ = w/wmax
        w_ = np.minimum(w_, 1)
        y_ = (a * w_**2 + b * w_ + c)
        y_ = np.maximum(0, y_)
        y_ = y_ * i_crop
        y = y_ * ymax * unit_area * 1e-4  # 1e4 bu
        cm2m = 0.01
        v_c = irr * unit_area * cm2m    # m-ha
        v = np.sum(v_c)                 # m-ha
        y_y = np.sum(y_) / n_s

        self.y, self.y_y, self.v = y, y_y, v

        ### Tech (for the pumping cost calculation in Finance module)
        self.update_irr_tech(i_te)  # update tech
        a_te = self.a_te
        b_te = self.b_te
        q = a_te * v + b_te
        self.q = q          # m-ha/day

        # record
        self.irr_vol = v    # m-ha

        return y, y_y, v


# Archive
#     def step(self, irr, i_crop, i_te, prec_aw, prec, temp):
#         """
#         Simulate a single timestep.

#         Parameters
#         ----------
#         irr : 3darray
#             An array outputted from OptModel() that represent the irrigation
#             depth for the following year [cm]. The dimension of the array
#             should be (n_s, n_c, 1).
#         i_crop : 3darray
#             An array outputted from OptModel() that represent the indicator
#             matrix for the crop choices in following year. The dimension of the
#             array should be (n_s, n_c, 1).
#         i_te : 1darray
#             An array outputted from OptModel() that represent the indicator
#             matrix for the irrigation technology choices in following year. The
#             dimension of the array should be (n_te).
#         prec_aw : float
#             The precipitation in the growing season [cm].
#         prec : float
#             The annual precipitation [cm].
#         temp : DataFrame
#             The daily mean temperature storing in a DataFrame format with
#             datetime index [degC].

#         Returns
#         -------
#         y : 3darray
#             Crop yield with the dimension (n_s, n_c, 1) [1e4 bu].
#         y_y : 3darray
#             Crop yield/maximum yield with the dimension (n_s, n_c, 1).
#         v : float
#             Irrigation amount [m-ha].
#         inflow : float
#             Inflow to the aquifer [m-ha].

#         """
#         # Crop yield
#         a = self.a
#         b = self.b
#         c = self.c
#         ymax = self.ymax
#         wmax = self.wmax
#         n_s = self.n_s
#         unit_area = self.unit_area
#         growth_period_ratio = self.growth_period_ratio

#         # Keep the record
#         self.pre_i_crop = self.i_crop
#         self.i_crop = i_crop
#         self.update_crops(i_crop)

#         irr = irr.copy()[:,:,[0]]

#         # Adjust growing period
#         prec_aw_ = np.ones(irr.shape) * prec_aw
#         for ci, crop in enumerate(self.crop_options):
#             prec_aw_[:, ci, :] = prec_aw_[:, ci, :] * growth_period_ratio[crop]

#         w = irr + prec_aw_
#         w = w * i_crop
#         w_ = w/wmax
#         w_ = np.minimum(w_, 1)  # can be removed to create uncertainty.
#         y_ = (a * w_**2 + b * w_ + c)
#         y_ = np.maximum(0, y_)
#         y_ = y_ * i_crop
#         y = y_ * ymax * unit_area * 1e-4  # 1e4 bu
#         cm2m = 0.01
#         v_c = irr * unit_area * cm2m    # m-ha
#         v = np.sum(v_c)                 # m-ha
#         y_y = np.sum(y_) / n_s

#         self.y, self.y_y, self.v = y, y_y, v
#         # Can be deleted in the future
#         #assert w_ >= 0 and w_ <= 1, f"w_ in [0, 1] expected, got: {w_}"
#         #assert y_ >= 0 and y_ <= 1, f"y_ in [0, 1]  expected, got: {y_}"

#         # Tech
#         self.update_irr_tech(i_te)  # update tech
#         a_te = self.a_te
#         b_te = self.b_te
#         q = a_te * v + b_te
#         self.q = q  # m-ha/d

#         # Annual ET for aquifer
#         # Calculate et + (ignore the corner. we will introduce an empirical
#         # coeficient to adjust the inflow~
#         wv = np.sum(w * unit_area) * cm2m
#         pet = self.cal_pet_Hamon(temp, self.lat, self.dz)
#         # We did not consider Kc variation here. Assume to have minor impact.
#         Kc = 1

#         # Adopt the stress coeficient from GWLF
#         Ks = np.ones(w_.shape)
#         Ks[w_ <= 0.5] = 2 * w_[w_ <= 0.5]

#         et = np.sum(Ks * Kc * sum(pet) * (unit_area) * cm2m)  # m-ha sum all n_s
#         inflow = wv - et                            # m-ha
#         inflow = max(0, inflow) # cannot be negative
#         self.inflow = inflow                        # m-ha
#         self.irr_vol = v
#         return y, y_y, v, inflow