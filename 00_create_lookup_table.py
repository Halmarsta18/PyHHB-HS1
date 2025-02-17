import xarray as xr
import HHB as PyHHB
import metpy.calc as mpcalc
from metpy.units import units
import numpy as np

# old_or_young = 'young'
old_or_young = 'old'
    
if old_or_young == 'young':
    
    save_str = 'young_woman_indoors'
    
elif old_or_young == 'old':
    
    save_str = 'old_woman_indoors'

huss_min = -0.0055
huss_min_thresh = 0
huss_max = 0.0429

huss_coarse_res = 20
huss_fine_res = 60

ps_min = 49.02556458
ps_min_thresh = 70
ps_max = 105.6

ps_coarse_res = 20
ps_fine_res = 40

tas_min = -67.67245484
tas_min_thresh = 15
# tas_max = 52.9595337
tas_max = 57

tas_coarse_res = 20
tas_fine_res = 60

#######################################################################
# IMPORT PyHHB
#######################################################################

# ----------------- Initial conditions; basal prerequisites -----------------

# ///////////////////////////////////

# Constants that seem pretty immutable to me:

# density
# Density of sweat, in kg / L
# Assumed to be equivalent to that of water. Maybe we can improve upon this assumption later, 
# but it probably wouldn't matter too much.
density = 1

# Lh_vap
# Heat of vaporisation of water at 30⁰C, 2426 J · g ^(-1)
Lh_vap = PyHHB.Lh_vap

# ///////////////////////////////////

# Things the user will want to define on a case-by-case basis:

# Hprod_rest
# Internal heat production at rest, in W / kg

Hprod_rest = 1.8

# ///////////////////////////////////

# Things that were originally defined in a "personal profile" (you can change them if you want, 
# but existing personal profiles may provide you with a good benchmark to start with):

# (I will be taking these example values from Young_adult_livability.txt)

# Tsk_C
# Skin temperature, in degrees C
Tsk_C = 35.0

# Emm_sk
# Area weighted emissivity of the clothed body surface, dimentionless
Emm_sk = 0.98

# W
# External work being done, in W
# In usage of PyHHB thus far, we conservatively assume that the subject is not doing any external 
# work, and is at rest.
W = 0

# Av_ms
# Wind speed (m / s)
Av_ms = 2.68

# ///////////////////////////////////

# Things that are different between survivability and livability

# A_eff
# Effective radiative area of the body, dimentionless
A_eff_surv = 0.70
A_eff_liv = 0.73

# Icl
# Insulation clothing value, in CLO
Icl_surv = 0
Icl_liv = 0.36

# Re_cl
# Evaporative resistance of clothing
Re_cl_surv = 0
Re_cl_liv = 0.01

# ///////////////////////////////////

def SurvLivFull(Ta_C, hu, hu_type, PB_kPa = 101.3, sun = "Night-Indoors", old_or_young = 'young',
                Exp_time = 3, Mmax_only = False):

# This function defines the Python Human Heat Balance Model (PyHHB). See Vanos et al (2023) for
# more information.

# PURPOSE: Given a temperature and humidity, output information about human survivability and
#          livability in those conditions.

# IMPORTANT NOTE: Please read about the following input choices CAREFULLY. Circumstances of heat
#                 exposure (age, exposure time, shade conditions) create large deviations in
#                 results.

# INPUT PARAMETERS:

# Ta_C: float or array, near-surface air temperature (C)
# hu: float or array, humidity in either specific humidity (dimensionless) or RH (dimensionless)
# hu_type: str, describes given humidity metric, 'q' or 'rh'. IMPORTANT NOTE: If using RH, your
#          input should be between 0 and 1! Do NOT use a percentage!
# PB_kPa: barometric pressure in kPa; defaults to 101.3 kPa
# Av_ms: Wind speed (m / s); defaults to 1 m / s
# sun: str, describes sunlight conditions, 'Night-Indoors' or 'Day-Outdoors'; defaults to 
#      'Night-Indoors'.
# old_or_young: str, choose whether you wish to consider a healthy young adult (aged 18-45) or an
#               old adult (65+), 'old' or 'young'. Defaults to 'young'.
# Exp_time: int, exposure time, in hours. Right now, PyHHB is only configured for exposure times
#           of 3 or 6 hours. Don't use input other than 3 or 6. Choose whichever option is 
#           appropriate for the model output you're applying PyHHB to (if model output is your
#           usecase). Defaults to 3.

# Default conditions assume a most optimistic scenario.

# OUTPUT PARAMETERS:

# (will fill this in later if I have more time)

# FIRST: Data preprocessing & defining of constants.
# Some constants will change based on parameters the user has defined. Let's define them now.
# Additionally, we will need to put temperature and humidity into NumPy arrays if they aren't
# already.

    if old_or_young == 'old':

        # person_condition
        # Maximum skin wettedness based on condition
        # The maximum skin wettedness of a person changes depending on heat acclimation status.
        # Here are some values from the PyHHB documentation:

        #     ISO:
        #     Unacclimated = 0.85
        #     Acclimatied = 1.00

        #     Ravanelli et al. MSSE (2018):
        #     Untrained & Unacclimated = 0.72
        #     Trained & Unacclimated = 0.84
        #     Trained & Acclimated = 0.95

        #     Morris 2015
        #     YNG Morris 2015 = 0.65
        #     OLD Morris 2015 = 0.5

        # The relevant PyHHB function, PyHHB.wmax(), recognizes the following strings as valid
        # values of person_condition:

        # 'Unacclimated'
        # 'fully acclimated'
        # 'Untrained & Unacclimated'
        # 'Trained & Unacclimated'
        # 'Trained & Acclimated'
        # 'YNG_Morris_2021'
        # 'OLD_Morris_2021'

        person_condition = "OLD_Morris_2021"

        # Mass, in kg
        Mass = 73.9

        # Height, in m
        # This can be set to -9999 to indicate you'd rather define a specific value for AD in the
        # personal profile.
        Height = -9999

        # AD
        # Corporeal surface area, in m ^ 2; can either be given in a personal profile or estimated
        # from mass and height
        AD = 1.78
        # AD = PyHHB.AD_from_mass_height(Mass, Height)

        # Smax
        # Maximum sweat rate, in L · h ^ (-1).
        Smax = 0.51

    elif old_or_young == 'young':

        # person_condition
        # Maximum skin wettedness based on condition
        # The maximum skin wettedness of a person changes depending on heat acclimation status.
        # Here are some values from the PyHHB documentation:

        #     ISO:
        #     Unacclimated = 0.85
        #     Acclimatied = 1.00

        #     Ravanelli et al. MSSE (2018):
        #     Untrained & Unacclimated = 0.72
        #     Trained & Unacclimated = 0.84
        #     Trained & Acclimated = 0.95

        #     Morris 2015
        #     YNG Morris 2015 = 0.65
        #     OLD Morris 2015 = 0.5

        # The relevant PyHHB function, PyHHB.wmax(), recognizes the following strings as valid
        # values of person_condition:

        # 'Unacclimated'
        # 'fully acclimated'
        # 'Untrained & Unacclimated'
        # 'Trained & Unacclimated'
        # 'Trained & Acclimated'
        # 'YNG_Morris_2021'
        # 'OLD_Morris_2021'

        person_condition = "YNG_Morris_2021"

        # Mass, in kg
        Mass = 56.2

        # Height, in m
        # This can be set to -9999 to indicate you'd rather define a specific value for AD in the
        # following line.
        Height = -9999

        # AD
        # Corporeal surface area, in m ^ 2; can either be given manually below or estimated from
        # mass and height using a PyHHB function
        AD = 1.6
        # AD = PyHHB.AD_from_mass_height(Mass, Height)

        # Smax
        # Maximum sweat rate, in L · h ^ (-1).
        Smax = 0.75
        
    elif old_or_young == 'KENDRA_young':
        
        person_condition = "KENDRA_young"
        
        Mass = 59.41
        
        Height = -9999
        
        AD = 1.63
        
        Smax = 0.61

    elif old_or_young == 'KENDRA_old':
        
        person_condition = "KENDRA_old"
        
        Mass = 56.69
        
        Height = -9999
        
        AD = 1.57
        
        Smax = 0.33
        
    # Now that we know all that, we can calculate a couple more constants.
    
    # ///////////////////////////////////

    # M
    # Metabolic rate
    # We assume the metabolic rate of a person is 1.8 W / kg. Hprod_rest = M - W, and in usage of
    # PyHHB so far, we assume W = 0. As such, M = Hprod_rest most of the time.
    # UPDATE 1/21/24: This is used to be a fixed number defined earlier in the cell, but now it's
    # dependent on Hprod_rest and Mass.
    M = Hprod_rest * Mass

    # M_rest
    # Resting metabolic rate; metabolic energetic expenditure while people are resting, in W.
    M_rest = Hprod_rest * Mass
    
    # Now, see if we need to change the type of Ta_C and hu.

    if type(Ta_C) == int or type(Ta_C) == float or type(Ta_C) == list:
        Ta_C = np.array([Ta_C])
        
    if type(hu) == int or type(hu) == float or type(hu) == list:
        hu = np.array([hu])
        
    # If we were given specific humidity, it will need to be converted to RH. Before we do that,
    # though, we need to calculate the mixing ratio.
    # UPDATE 1/21/24: Mixing ratio is a new necessary component of the model

    if hu_type == 'q':
        
        # mixing_ratio
        mixing_ratio = mpcalc.mixing_ratio_from_specific_humidity(hu)
        
        RH = mpcalc.relative_humidity_from_specific_humidity(PB_kPa * units.kPa, 
                                                             Ta_C * units.degC, 
                                                             hu)
        
    elif hu_type == 'rh':
        
        # mixing ratio
        mixing_ratio = mpcalc.mixing_ratio_from_relative_humidity(PB_kPa * units.kPa, 
                                                                  Ta_C * units.degC,
                                                                  hu)
        
        # print("mixing_ratio: " + str(mixing_ratio))
        
        RH = hu

    # ///////////////////////////////////

    # Things that need to be calculated/require knowledge of prior constants and choices and WILL
    # change as a result of climatic variables, but require no extra input than what we've given
    # already:
    
    # mrt_C
    # Mean radiant temperature, in C
    # mrt_C is Ta_C if indoors and Ta_C + 15 if outdoors. 

    if sun == "Night-Indoors":
        mrt_C = Ta_C
    elif sun == "Day-Outdoors":
        mrt_C = Ta_C + 15
        
    # SECOND: Run the model.

    # The following code is separated into "tiers." All the information from a higher tier is
    # necessary to calculate values in a lower tier.

    # ----------------- Tier 8 -----------------

    # hr_cof_from_radiant_features
    # UPDATE 1/21/24: Needs two branches for survivability and livability
    hr_cof_from_radiant_features_surv = PyHHB.hr_cof_from_radiant_features(mrt_C, Tsk_C, Emm_sk,
                                                                           A_eff_surv)
    hr_cof_from_radiant_features_liv = PyHHB.hr_cof_from_radiant_features(mrt_C, Tsk_C, Emm_sk,
                                                                          A_eff_liv)
    
    # hc_cof_from_Av
    hc_cof_from_Av = PyHHB.hc_cof_from_Av(Av_ms)
    
    # Psa_kPa_from_TaC
    Psa_kPa_from_TaC = PyHHB.Psa_kPa_from_TaC(Tsk_C)
    
    # ----------------- Tier 7 -----------------

    # to_from_hr_tr_hc_ta
    # UPDATE 1/21/24: Needs two branches for survivability and livability
    to_from_hr_tr_hc_ta_surv = PyHHB.to_from_hr_tr_hc_ta(hr_cof_from_radiant_features_surv, 
                                                         mrt_C,
                                                         hc_cof_from_Av, Ta_C)
    to_from_hr_tr_hc_ta_liv = PyHHB.to_from_hr_tr_hc_ta(hr_cof_from_radiant_features_liv, mrt_C,
                                                        hc_cof_from_Av, Ta_C)
        
    # h_coef_from_hc_hr
    # UPDATE 1/21/24: Needs two branches for survivability and livability
    h_coef_from_hc_hr_surv = PyHHB.h_coef_from_hc_hr(hc_cof_from_Av, 
                                                     hr_cof_from_radiant_features_surv)
    h_coef_from_hc_hr_liv = PyHHB.h_coef_from_hc_hr(hc_cof_from_Av, 
                                                    hr_cof_from_radiant_features_liv)
    
    # vapor_pressure
    # UPDATE 1/21/24: Now metpy function rather than PyHHB function
    # Pv_kPa_from_Psa_RH = PyHHB.Pv_kPa_from_Psa_RH(Psa_kPa_from_TaC, RH)
    vapor_pressure = mpcalc.vapor_pressure(PB_kPa * units.kPa, mixing_ratio).magnitude
    
    # ----------------- Tier 6 -----------------

    # Dry_Heat_Loss_c_plus_r
    # UPDATE 1/21/24: Needs two branches for survivability and livability
    Dry_Heat_Loss_c_plus_r_surv = PyHHB.Dry_Heat_Loss_c_plus_r(Tsk_C, to_from_hr_tr_hc_ta_surv,
                                                               Icl_surv, h_coef_from_hc_hr_surv,
                                                               AD)
    Dry_Heat_Loss_c_plus_r_liv = PyHHB.Dry_Heat_Loss_c_plus_r(Tsk_C, to_from_hr_tr_hc_ta_liv,
                                                              Icl_liv,
                                                              h_coef_from_hc_hr_liv, AD)

    # Cres_from_M_Ta
    Cres_from_M_Ta = PyHHB.Cres_from_M_Ta(M, Ta_C, AD)

    # Eres_from_M_Pa
    # UPDATE 1/21/24: Now takes vapor_pressure rather than Pv_kPa_from_Psa_RH
    Eres_from_M_Pa = PyHHB.Eres_from_M_Pa(M, vapor_pressure, AD)
    
    # he_cof
    he_cof = PyHHB.he_cof(hc_cof_from_Av)
    
# ----------------- Tier 5 -----------------

    # Ereq_from_HeatFluxes
    # UPDATE 1/21/24: Needs two branches for survivability and livability
    Ereq_from_HeatFluxes_surv = PyHHB.Ereq_from_HeatFluxes(M, W, Dry_Heat_Loss_c_plus_r_surv,
                                                           Cres_from_M_Ta, Eres_from_M_Pa)
    Ereq_from_HeatFluxes_liv = PyHHB.Ereq_from_HeatFluxes(M, W, Dry_Heat_Loss_c_plus_r_liv,
                                                          Cres_from_M_Ta, Eres_from_M_Pa)
    # Emax_env
    # UPDATE 1/21/24: Now takes vapor_pressure instead of Pv_kPa_from_Psa_RH
    #                 Needs two branches for survivability and livability
    Emax_env_surv = PyHHB.Emax_env(Psa_kPa_from_TaC, vapor_pressure, Re_cl_surv, he_cof,
                                   Icl_surv,
                                   AD)
    Emax_env_liv = PyHHB.Emax_env(Psa_kPa_from_TaC, vapor_pressure, Re_cl_liv, he_cof, Icl_liv,
                                  AD)
    
    Emax_env_surv[Emax_env_surv<0] = 0
    Emax_env_liv[Emax_env_liv<0] = 0
    
# ----------------- Tier 4 -----------------
    # wreq_HSI_skin_wettedness
    # UPDATE 1/21/24: Needs two branches for survivability and livability
    wreq_HSI_skin_wettedness_surv = PyHHB.wreq_HSI_skin_wettedness(Ereq_from_HeatFluxes_surv,
                                                                   Emax_env_surv)
    
    wreq_HSI_skin_wettedness_liv = PyHHB.wreq_HSI_skin_wettedness(Ereq_from_HeatFluxes_liv,
                                                                  Emax_env_liv)
    
# ----------------- Tier 3 -----------------

    # wmax
    wmax = PyHHB.wmax(person_condition)
    
    # 4. Sweating_efficiency_r
    # UPDATE 1/21/24: Needs two branches for survivability and livability
    Sweating_efficiency_r_surv = PyHHB.Sweating_efficiency_r(wreq_HSI_skin_wettedness_surv)
    Sweating_efficiency_r_liv = PyHHB.Sweating_efficiency_r(wreq_HSI_skin_wettedness_liv)
    
# ----------------- Tier 2 -----------------

    # Emax_wettedness
    # UPDATE 1/21/24: Now takes vapor_pressure instead of Pv_kPa_from_Psa_RH
    #                 Needs two branches for survivability and livability
    # Emax_wettedness = PyHHB.Emax_wettedness(wmax, Psa_kPa_from_TaC, vapor_pressure, Re_cl,
    #                                         he_cof, Icl, AD)
    Emax_wettedness_surv = PyHHB.Emax_wettedness(wmax, Psa_kPa_from_TaC, vapor_pressure, 
                                                 Re_cl_surv, 
                                                 he_cof, Icl_surv, AD)
    Emax_wettedness_liv = PyHHB.Emax_wettedness(wmax, Psa_kPa_from_TaC, vapor_pressure, 
                                                Re_cl_liv, 
                                                he_cof, Icl_liv, AD)

    # Emax_sweat_rate
    # UPDATE 1/21/24: Needs two branches for survivability and livability
    Emax_sweat_rate_surv = PyHHB.Emax_sweat_rate(Smax, Lh_vap, density, 
                                                 Sweating_efficiency_r_surv)
    Emax_sweat_rate_liv = PyHHB.Emax_sweat_rate(Smax, Lh_vap, density, Sweating_efficiency_r_liv)

    # Sreq
    # UPDATE 1/21/24: Needs two branches for survivability and livability
    Sreq_surv = PyHHB.Sreq(Ereq_from_HeatFluxes_surv, Sweating_efficiency_r_surv, Lh_vap)

    Sreq_liv = PyHHB.Sreq(Ereq_from_HeatFluxes_liv, Sweating_efficiency_r_liv, Lh_vap)
    
# ----------------- Tier 1 -----------------

    # survivability
    survivability, flag_survivability = PyHHB.Survivability(Exp_time, Ereq_from_HeatFluxes_surv, 
                                                            Emax_wettedness_surv, 
                                                            Emax_sweat_rate_surv,
                                                            Sreq_surv, Smax, Mass)
    
# ----------------- Tier 0 -----------------
# ----------------- ULTIMATE GOAL: OBTAIN LIVABILITY -----------------

    Mmax, mask_non_livable = PyHHB.livability_Mmax(survivability, Ereq_from_HeatFluxes_liv, 
                                                   Emax_wettedness_liv, Emax_sweat_rate_liv, 
                                                   M_rest)
    
    # Extra: Convert Mmax from W to METs

    Mmax = PyHHB.MetabolicRate_W_to_MET_Mass(Mmax, Mass)
    
    if Mmax_only == True:
        
        return Mmax
    
    else:
    
        return survivability, flag_survivability, Mmax, mask_non_livable

#######################################################################
# CREATE LOOKUP TABLE
#######################################################################

n_coarse_huss, n_coarse_ps, n_coarse_tas = (huss_coarse_res, ps_coarse_res, tas_coarse_res)
n_fine_huss, n_fine_ps, n_fine_tas = (huss_fine_res, ps_fine_res, tas_fine_res)

huss_coarse_linspace = np.linspace(huss_min, huss_min_thresh, n_coarse_huss)
huss_fine_linspace = np.linspace(huss_min_thresh, huss_max, n_fine_huss)
huss_linspace = np.concatenate((huss_coarse_linspace[:-1], huss_fine_linspace))

ps_coarse_linspace = np.linspace(ps_min, ps_min_thresh, n_coarse_ps)
ps_fine_linspace = np.linspace(ps_min_thresh, ps_max, n_fine_ps)
ps_linspace = np.concatenate((ps_coarse_linspace[:-1], ps_fine_linspace))

tas_coarse_linspace = np.linspace(tas_min, tas_min_thresh, n_coarse_tas)
tas_fine_linspace = np.linspace(tas_min_thresh, tas_max, n_fine_tas)
tas_linspace = np.concatenate((tas_coarse_linspace[:-1], tas_fine_linspace))

# Units going into PyHHB must be C and kilopascal! Convert if necessary.

# tas_linspace = tas_linspace - 273.15

# ps_linspace = ps_linspace / 1000

huss, ps, tas = np.meshgrid(huss_linspace, ps_linspace, tas_linspace)

survivability, flag_survivability, Mmax, mask_non_livable = SurvLivFull(tas, huss, 'q', 
                                                                        PB_kPa = ps, 
                                                                        old_or_young = old_or_young, 
                                                                        sun = 'Night-Indoors')

pyhhb_output = xr.Dataset(data_vars = dict(survivability = (["ps", "huss", "tas"], survivability),
                                           flag_survivability = (["ps", "huss", "tas"], flag_survivability),
                                           Mmax = (["ps", "huss", "tas"], Mmax),
                                           mask_non_livable = (["ps", "huss", "tas"], mask_non_livable)),
                          coords = dict(ps = ("ps", ps_linspace),
                                        huss = ("huss", huss_linspace),
                                        tas = ("tas", tas_linspace)),
                          attrs = dict(description = "PyHHB lookup table"))

pyhhb_output = pyhhb_output.transpose("huss", "ps", "tas")
pyhhb_output = pyhhb_output[["huss", "ps", "tas", "survivability", "flag_survivability", "Mmax", "mask_non_livable"]]

#######################################################################
# SAVE LOOKUP TABLE
#######################################################################

labspace = '/dfs9/baldwij1-lab/'
workdir = labspace + 'hstaudmy/chapter1/'

path_save = workdir + 'output/lookup_tables/' + save_str + '/'

huss_res_str = str(huss_coarse_res) + '-' + str(huss_fine_res)
ps_res_str = str(ps_coarse_res) + '-' + str(ps_fine_res)
tas_res_str = str(tas_coarse_res) + '-' + str(tas_fine_res)

pyhhb_output.to_netcdf(path_save + 'newLT_' + huss_res_str + '_' + ps_res_str + '_' + tas_res_str + '_' + save_str + '.nc')
