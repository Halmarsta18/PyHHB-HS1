import xarray as xr
import numpy as np
import sys
import os.path

huss_res_str = '20-60' 
ps_res_str = '20-40'
tas_res_str = '20-60'

scenario = 'ssp245'
warming_level = 'two'
fresh = True

# pyhhb_specs = 'young_woman_indoors'
pyhhb_specs = 'old_woman_indoors'

baseline_start = 1980
baseline_end = 2010

model = 'ERA5'

# =============================================================================
# Defining directories
# =============================================================================

labspace = '/dfs9/baldwij1-lab/'
workdir = labspace + 'hstaudmy/chapter1/'

path_deltas = workdir + 'output/' + model + '/04_apply_deltas_and_trim_to_90/' + scenario + '/' + \
              warming_level + '/'
path_save = workdir + 'output/' + model + '/05_run_look_up_table_pyhhb/' + scenario + '/' + warming_level + \
            '/' + pyhhb_specs + '/'
    
path_mask = workdir + 'data/ERA5/'
path_lookup_table = workdir + 'output/lookup_tables/' + pyhhb_specs + '/'


print("Opening lookup table . . . ", flush = True)

# =============================================================================
# Load in existing files
# =============================================================================

# Create lookup table

lt_prime = xr.open_dataset(path_lookup_table + 'newLT_' + huss_res_str + '_' + ps_res_str + '_' + \
                           tas_res_str + '_' + pyhhb_specs + '.nc')
      
print("Lookup table opened. Opening mask . . . ", flush = True)

mask = xr.open_dataset(path_mask + 'e5.oper.invariant.128_172_lsm.ll025sc.1979010100_1979010100.nc')

print("Opened mask. Defining lookup table functions . . .", flush = True)

# =============================================================================
# Define lookup table functions
# =============================================================================

def expand_array_with_fill(original_array, new_shape, fill_value=-1):
    # Create a new array with the desired shape and fill it with the fill value
    expanded_array = np.full(new_shape, fill_value, dtype = original_array.dtype)
    
    # Determine the shape of the original array
    original_shape = original_array.shape
    
    # Copy the original array into the expanded array
    expanded_array[:original_shape[0], :original_shape[1], :original_shape[2]] = original_array
    
    return expanded_array

def find_nearest(array, values):
    """Find the nearest values in an array for a given set of values, handling NaNs."""
    array = np.asarray(array)
    indices = np.full(values.shape, -1, dtype=int)  # Use -1 to indicate invalid indices
    
    valid_mask = ~np.isnan(values)
    
    # Only compute nearest for non-NaN values
    indices[valid_mask] = np.abs(array - values[valid_mask][..., np.newaxis]).argmin(axis=-1)
    
    return indices

def pyhhb_lookup(humidity_da, pressure_da, temperature_da, humidity_coords, pressure_coords, temperature_coords, lt_mmax):
    # Find nearest values
    nearest_h = find_nearest(humidity_coords, humidity_da)
    nearest_p = find_nearest(pressure_coords, pressure_da)
    nearest_t = find_nearest(temperature_coords, temperature_da)
    
    # Initialize Mmax with NaNs
    Mmax = np.full(humidity_da.shape, np.nan, dtype = float)

    # Determine valid indices (where none of the inputs are NaN and indices are valid)
    valid_mask = (nearest_h != -1) & (nearest_p != -1) & (nearest_t != -1)

    # If a place is valid according to the mask, keep it. If not, replace whatever's there with a -1.
    nearest_h = np.where(valid_mask == True, nearest_h, -1).astype('int')
    nearest_p = np.where(valid_mask == True, nearest_p, -1).astype('int')
    nearest_t = np.where(valid_mask == True, nearest_t, -1).astype('int')

    # Now, let's put a np.nan at the end of the lookup table in every dimension.
    # We'll need to create a lookup table of a shape that is one larger in each dimension, so let's make a tuple of that
    # shape now.
    new_shape = (np.shape(lt_mmax)[0] + 1, np.shape(lt_mmax)[1] + 1, np.shape(lt_mmax)[2] + 1)
    
    # Now, we can use a function we defined earlier to put a np.nan at the end of the lookup table in every dimension.
    # Anything that got replaced with a -1 in the np.where() functions above will now grab a np.nan!
    expanded_lt = expand_array_with_fill(lt_mmax, new_shape, fill_value = np.nan)

    # Compute Mmax only for valid indices
    
    Mmax = expanded_lt[nearest_h, nearest_p, nearest_t]
        
    return Mmax
    
# =============================================================================
# Setting the scene
# =============================================================================

# Setting details in the run (THIS ANALYSIS IS SENSIBLE TO WHAT WE CONSIDER WARM/HOT OR 
# NOT)
print("Lookup table functions defined. Naming some constants . . .", flush = True)
# Tair_threshold = 25 # Temperature to say it is warm weather to do the analysis.

# Setting geography boundaries 
lat_min = -60
lat_max = 90
lon_min = 0
lon_max = 360

print("Constants named. Selecting mask ...", flush = True)

mask = mask['LSM'].sel(time = '1979-01-01', latitude = slice(lat_max, lat_min), 
                       longitude = slice(lon_min, lon_max)).drop_vars('time')

print("Mask selected. Deleting folks we don't need anymore . . .", flush = True)

del labspace
del path_lookup_table
del path_mask
del workdir

# Now for each year . . .
      
print("Run data through lookup table. Beginning iteration . . .", flush = True)

months = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

years = range(baseline_start, (baseline_end + 1))

for year in years:
 
    year_str = str(year)
    
    if fresh == False:

        if os.path.isfile(path_save + 'LT_' + year_str + '_mmax.nc'):     

            continue
    
    print("Begin year " + year_str, flush = True)

    mmax_ds = [None] * 12
    i = 0

    for mon in months:
        
        mon_str = str(mon)
        
        if os.path.isfile(path_save + 'LT_' + year_str + '_' + mon_str + '_mmax.nc'):
            
            mmax_lt = xr.open_dataset(path_save + 'LT_' + year_str + '_' + mon_str + '_mmax.nc')
            
            try:
            
                mmax_lt = mmax_lt.rename_vars({'__xarray_dataarray_variable__' : 'mmax'})
                
            except:
                
                print('Try 1 did not work', flush = True)
            
            mmax_ds[i] = mmax_lt

            continue

        era5_plus_delta = xr.open_dataset(path_deltas + year_str + '_era5_plus_delta.nc')


        era5_plus_delta = era5_plus_delta.where(era5_plus_delta['time'].dt.month == mon, drop = True)

        era5_plus_delta = era5_plus_delta.sel(latitude = slice(lat_max, lat_min), 
                                              longitude = slice(lon_min, lon_max))

        era5_plus_delta = era5_plus_delta.where(mask > 0)

        era5_plus_delta['tas'] = era5_plus_delta['tas'] - 273.15
        era5_plus_delta['ps'] = era5_plus_delta['ps'] / 1000

        print("Time to trim the lookup table to only what we need . . .", flush = True)

        lt = lt_prime

        # Make lookup table smaller to fit needs, starting with huss

        huss_max = np.atleast_1d(era5_plus_delta['huss'].max().values)


        huss_min = np.atleast_1d(era5_plus_delta['huss'].min().values)

        lt_min_huss = np.atleast_1d(lt['huss'].min().values)[0]
        lt_max_huss = np.atleast_1d(lt['huss'].max().values)[0]

        if huss_min <= lt_min_huss:

            print("Yikes, you need a bigger lookup table (need smaller huss values)!!", flush = True)
            
            print("Minimum huss from the data:", flush = True)
            print(huss_min, flush = True)
            print("Minimum huss from the lookup table:", flush = True)
            print(lt_min_huss, flush = True)
        
        if huss_max >= lt_max_huss:

            print("Yikes, you need a bigger lookup table (need bigger huss values)!!", flush = True)
            
            print("Maximum huss from the data:", flush = True)
            print(huss_max, flush = True)
            print("Maximum huss from the lookup table:", flush = True)
            print(lt_max_huss, flush = True)
            
        new_min_huss_bound = np.atleast_1d(lt.where(lt['huss'] < huss_min, drop = True)['huss'].max().values)[0]
        new_max_huss_bound = np.atleast_1d(lt.where(lt['huss'] > huss_max, drop = True)['huss'].min().values)[0]

        lt = lt.where(lt['huss'] >= new_min_huss_bound, drop = True)
        lt = lt.where(lt['huss'] <= new_max_huss_bound, drop = True)

        # Now trim lookup table in ps dimension

        ps_max = np.atleast_1d(era5_plus_delta['ps'].max().values)

        ps_min = np.atleast_1d(era5_plus_delta['ps'].min().values)

        lt_min_ps = np.atleast_1d(lt['ps'].min().values)[0]
        lt_max_ps = np.atleast_1d(lt['ps'].max().values)[0]

        if ps_min <= lt_min_ps:

            print("Yikes, you need a bigger lookup table (need smaller ps values)!!", flush = True)
            
            print("Minimum ps from the data:", flush = True)
            print(ps_min, flush = True)
            print("Minimum ps from the lookup table:", flush = True)
            print(lt_min_ps, flush = True)
                        
        if ps_max >= lt_max_ps:

            print("Yikes, you need a bigger lookup table (need bigger ps values)!!", flush = True)
            
            print("Maximum ps from the data:", flush = True)
            print(ps_max, flush = True)
            print("Maximum ps from the lookup table:", flush = True)
            print(lt_max_ps, flush = True)
            
        new_min_ps_bound = np.atleast_1d(lt.where(lt['ps'] < ps_min, drop = True)['ps'].max().values)[0]
        new_max_ps_bound = np.atleast_1d(lt.where(lt['ps'] > ps_max, drop = True)['ps'].min().values)[0]

        lt = lt.where(lt['ps'] >= new_min_ps_bound, drop = True)
        lt = lt.where(lt['ps'] <= new_max_ps_bound, drop = True)

        # Finally, trim lookup table in tas dimension

        tas_max = np.atleast_1d(era5_plus_delta['tas'].max().values)

        tas_min = np.atleast_1d(era5_plus_delta['tas'].min().values)

        lt_min_tas = np.atleast_1d(lt['tas'].min().values)[0]
        lt_max_tas = np.atleast_1d(lt['tas'].max().values)[0]

        if tas_min <= lt_min_tas:

            print("Yikes, you need a bigger lookup table (need smaller tas values)!!", flush = True)
            
            print("Minimum tas from the data:", flush = True)
            print(tas_min, flush = True)
            print("Minimum tas from the lookup table:", flush = True)
            print(lt_min_tas, flush = True)
            
#             raise Exception("Yikes, you need a bigger lookup table (need smaller tas values)!!")

        if tas_max >= lt_max_tas:

            print("Yikes, you need a bigger lookup table (need bigger tas values)!!", flush = True)
            
            print("Maximum tas from the data:", flush = True)
            print(tas_max, flush = True)
            print("Maximum tas from the lookup table:", flush = True)
            print(lt_max_tas, flush = True)
            
#             raise Exception("Yikes, you need a bigger lookup table (need bigger tas values)!!")

        new_min_tas_bound = np.atleast_1d(lt.where(lt['tas'] < tas_min, drop = True)['tas'].max().values)[0]
        new_max_tas_bound = np.atleast_1d(lt.where(lt['tas'] > tas_max, drop = True)['tas'].min().values)[0]

        lt = lt.where(lt['tas'] >= new_min_tas_bound, drop = True)
        lt = lt.where(lt['tas'] <= new_max_tas_bound, drop = True)

        print("Done trimming lookup table. New lookup table dimensions:", flush = True)
        print(lt.sizes, flush = True)

        print("Time to make everything float16 . . .", flush = True)

        print("Converting lookup table to float32 . . .", flush = True)

        lt['huss'] = lt['huss'].astype('float32')
        lt['ps'] = lt['ps'].astype('float32')
        lt['tas'] = lt['tas'].astype('float32')

        print("Converting climate model output to float16 . . .", flush = True)

        era5_plus_delta['huss'] = era5_plus_delta['huss'].astype('float16')
        era5_plus_delta['ps'] = era5_plus_delta['ps'].astype('float16')
        era5_plus_delta['tas'] = era5_plus_delta['tas'].astype('float16')

        print("Done converting everything to float16!", flush = True)

        print("Using lookup table . . .", flush = True)

        # Now, we'll do it live!

        mmax_lt = xr.apply_ufunc(pyhhb_lookup,
                                 era5_plus_delta['huss'],
                                 era5_plus_delta['ps'],
                                 era5_plus_delta['tas'],
                                 lt.coords['huss'],
                                 lt.coords['ps'],
                                 lt.coords['tas'],
                                 lt['Mmax'],
                                 input_core_dims = [["latitude", "longitude", 'time'],
                                                    ["latitude", "longitude", 'time'],
                                                    ["latitude", "longitude", 'time'],
                                                    ['huss'], ['ps'], ['tas'],
                                                    ['huss', 'ps', 'tas']],
                                 output_core_dims = [["latitude", "longitude", 'time']],
                                 vectorize = True,
                                 dask = 'parallelized',  # Use Dask for parallel execution if needed
                                 output_dtypes = [float, float])    

        del era5_plus_delta

        mmax_ds[i] = mmax_lt
        
        mmax_lt.to_netcdf(path_save + 'LT_' + year_str + '_' + mon_str + '_mmax.nc')

        del mmax_lt

        i = i + 1

        print("Done at " + mon_str, flush = True)

    try:
        
        mmax_all = xr.concat([mmax_ds[0], mmax_ds[1], mmax_ds[2], mmax_ds[3], mmax_ds[4], mmax_ds[5], mmax_ds[6], mmax_ds[7], mmax_ds[8], 
                              mmax_ds[9], mmax_ds[10], mmax_ds[11]], dim = "time")
        
    except:
        
        print('Try 2 did not work (concatenation)', flush = True)

    print("Done at " + year_str + '. Saving . . .', flush = True)
    
    try:

        mmax_all.to_netcdf(path_save + 'LT_' + year_str + '_mmax.nc')
        
        i = 0
        
        files = [None] * 12
        
        try:
        
            for a_month in months:

                files[i] = path_save + 'LT_' + year_str + '_' + str(a_month) + '_mmax.nc'

                i = i + 1

            for f in files:

                os.remove(f)
                
        except:

            print("Couldn't delete monthly progress after saving out yearly progress", flush = True)
        
    except:
        
        print('try 3 did not work (to_netcdf)', flush = True)

    print("Saved! Moving on . . .", flush = True)
    
exit()
