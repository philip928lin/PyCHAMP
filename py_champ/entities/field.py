r"""
The code is developed by Chung-Yi Lin at Virginia Tech, in April 2023.
Email: chungyi@vt.edu
Last modified on Sep 6, 2023
"""
import numpy as np
import mesa

class Field(mesa.Agent):
    """
    A class to simulate a field in an agricultural system.

    Attributes
    ----------
    field_id : str or int
        Unique identifier for the field.
    crop_options : list
        List of available crop options.
    tech_options : list
        List of available technology options.
    t : int
        Current time step.
    irr_vol : float
        Irrigation volume.
    yield_rate_per_field : float
        Average yield rate per field.
    irr_vol_per_field : float
        Irrigation volume per field.

    Methods
    -------
    load_config(config)
        Load configuration parameters.
    update_irr_tech(i_te)
        Update irrigation technology.
    update_crops(i_crop)
        Update crops.
    step(irr_depth, i_crop, i_te, prec_aw)
        Simulate the field for one time step based on various parameters.
    """

    # aquifer_id, lat, dz => They are for dynamic inflow calculation (deprecated)
    def __init__(self, field_id, mesa_model, config, ini_crop, ini_te, ini_field_type,
                 crop_options, tech_options, **kwargs):
        """
        Initialize a Field object.

        Parameters
        ----------
        field_id : str or int
            Unique identifier for the field.
        mesa_model : object
            Reference to the overarching MESA model instance.
        config : dict
            General configuration information for the model.
        crop_options : list
            List of available crop options.
        tech_options : list
            List of available technology options.
        ini_te : str
            Initial technology.
        ini_crop : str or list
            Initial crop or list of crops.
        ini_field_type : str
            Initial field type.
        kwargs : dict, optional
            Additional optional arguments.
            
        Notes
        -----
        The `kwargs` could contain any additional attributes that you want to
        add to the Farmer agent. Available keywords include
        block_w_interval_for_corn : list
            An interval of (perceived) avaiable water [w_low, w_high] that 
            the corn crop type cannot be chose.  
        """
        # MESA required attributes => (unique_id, model)
        super().__init__(field_id, mesa_model)
        self.agt_type = "Field"

        # Initialize attributes
        self.field_id = field_id
        self.crop_options = crop_options
        self.tech_options = tech_options
        self.load_config(config)

        # Initialize  tech
        self.te = ini_te    # serve as the tech in the previous year
        self.update_irr_tech(ini_te)

        # Initialize  crop
        i_crop = np.zeros((self.n_s, self.n_c, 1))
        if isinstance(ini_crop, str):
            i_c = self.crop_options.index(ini_crop)
            i_crop[:, i_c, 0] = 1
            self.crops = [ini_crop]*self.n_s
        else:
            self.crops = ini_crop
            for s, c in enumerate(ini_crop):
                i_c = self.crop_options.index(c)
                i_crop[s, i_c, 0] = 1
        self.i_crop = i_crop
        self.update_crops(i_crop)
        
        # Initialize field type
        self.field_type = ini_field_type

        # Additional attributes from kwargs
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.block_w_interval_for_corn = kwargs.get("block_w_interval_for_corn")

        # Initialize other variables
        self.t = 0
        self.irr_vol = None

    def load_config(self, config):
        """
        Load field, crop, and technology parameters from the configuration dictionary.
 
        Parameters
        ----------
        config : dict
            General configuration information for the model.
        
        Returns
        -------
        None
        """

        crop_options = self.crop_options
        config_field = config["field"]
        crop_par = np.array([config_field["crop"][c] for c in crop_options])
        self.ymax = crop_par[:, 0].reshape((-1, 1))     # (n_c, 1)
        self.wmax = crop_par[:, 1].reshape((-1, 1))     # (n_c, 1)
        self.a = crop_par[:, 2].reshape((-1, 1))        # (n_c, 1)
        self.b = crop_par[:, 3].reshape((-1, 1))        # (n_c, 1)
        self.c = crop_par[:, 4].reshape((-1, 1))        # (n_c, 1)
        self.n_s = config_field['area_split']
        self.n_c = len(crop_options)
        self.field_area = config_field["field_area"]
        self.unit_area = self.field_area/self.n_s
        self.tech_par = config_field["tech"]
        
        self.yield_rate_per_field = None
        self.irr_vol_per_field = None

    def update_irr_tech(self, i_te):
        """
        Update the irrigation technology used in the field based on given
        indicator array. The dimension of the array should be (n_te).

        Parameters
        ----------
        i_te : 1darray or str
            Indicator array or string representing the chosen irrigation 
            technology for the next year. The dimension of the array should be
            (n_te).

        Returns
        -------
        None
        """
        if isinstance(i_te, str):
            new_te = i_te
        else:
            # Use argmax instead of "==1" to avoid float numerical issue.
            new_te = self.tech_options[np.argmax(i_te)]
        self.a_te, self.b_te, self.l_pr = self.tech_par[new_te]
        self.pre_te = self.te
        self.te = new_te

    def update_crops(self, i_crop):
        """
        Update the crop types for each area split based on the given indicator
        array. The dimension of the array should be (n_s, n_c, 1).

        Parameters
        ----------
        i_crop : 3darray
            Indicator array representing the chosen crops for the next year for
            each area split. The dimension of the array should be (n_s, n_c, 1).

        Returns
        -------
        None
        """
        n_s = self.n_s
        crop_options = self.crop_options
        # Use argmax instead of ==1 to avoid float numerical issue
        self.pre_i_crop = self.i_crop
        crops = [crop_options[np.argmax(i_crop[s, :, 0])] for s in range(n_s)]
        self.crops = crops
        self.i_crop = i_crop

    def step(self, irr_depth, i_crop, i_te, prec_aw):
        """
        Simulate the field for a single timestep.

        Parameters
        ----------
        irr_depth : 3darray
            Irrigation depth array for the next year. Dimensions: (n_s, n_c, 1).
        i_crop : 3darray
            Indicator array for crop choices for the next year. Dimensions: 
            (n_s, n_c, 1).
        i_te : 1darray
            Indicator array for irrigation technology choices for the next year.
            Dimensions: (n_te).
        prec_aw : dict
            Dictionary containing available precipitation for each crop in cm.

        Returns
        -------
        y : 3darray
            Crop yield in 1e4 bu for each area split and crop type. Dimensions:
            (n_s, n_c, 1).
        avg_y_y : float
            Ratio of actual yield to maximum possible yield (average across splits).
        irr_vol : float
            Total irrigation volume, in m-ha.
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
        irr_depth = irr_depth.copy()[:,:,[0]]
        prec_aw_ = np.ones(irr_depth.shape)
        for ci, crop in enumerate(self.crop_options):
            prec_aw_[:, ci, :] = prec_aw[crop]

        w = irr_depth + prec_aw_
        w = w * i_crop
        w_ = w/wmax
        w_ = np.minimum(w_, 1)
        y_ = (a * w_**2 + b * w_ + c)
        y_ = np.maximum(0, y_)
        y_ = y_ * i_crop
        y = y_ * ymax * unit_area * 1e-4      # 1e4 bu
        cm2m = 0.01
        v_c = irr_depth * unit_area * cm2m    # m-ha
        irr_vol = np.sum(v_c)                 # m-ha
        avg_y_y = np.sum(y_) / n_s

        ### Tech (for the pumping cost calculation in Finance module)
        self.update_irr_tech(i_te)  # update tech
        a_te = self.a_te
        b_te = self.b_te
        pumping_rate = a_te * irr_vol + b_te    # (McCarthy et al., 2020)
        self.pumping_rate = pumping_rate        # m-ha/day

        # record
        self.y = y 
        self.yield_rate_per_field = avg_y_y
        self.irr_vol_per_field = irr_vol    # m-ha

        return y, avg_y_y, irr_vol