import numpy as np
import itertools
import os
import pandas as pd
from sklearn.preprocessing import RobustScaler,MinMaxScaler, StandardScaler
from ImageSampler import ImageSampler

def get_element_map(mineral_image,composition_df,element,noise=False,noise_value=0.1):
    # Convert mineral map to element map
    # Create the NumPy Lookup Array
    # The old values (index) must be contiguous integers starting from 0.
    lookup_table = composition_df[element].to_numpy()

    # Convert to element values
    new_matrix = lookup_table[mineral_image]

    # Add  gaussian noise
    if noise:
        new_matrix = new_matrix + new_matrix*np.random.uniform(low=-noise_value, high=noise_value, size=new_matrix.shape)
        new_matrix = np.where(new_matrix<0,0,new_matrix)
    return new_matrix
    
# def run_random_expanding_window(simulation,args,composition_df,element_interest = 'Fe',max_spot_size=10,noise_value=0):
#     composition_per_window = []
#     np.random.seed(args['main_seed']+simulation) 
#     selected_coords = np.random.randint(low=0, high=args['image_size']-max_spot_size, size=(1,2))
#     for spot_size in np.arange(1,max_spot_size,1):
#         args['spot_size'] = spot_size

#         np.random.seed(args['main_seed']) 
#         sampler = ImageSampler(**args)
#         image = sampler.main_image 


#         seed_set = np.random.randint(0, args['spot_size']*args['spot_size']*(args['spot_size']**2), size=args['spot_size']*args['spot_size']*(args['spot_size']**2))

#         mask,coords = sampler.sampling_mask(seed_set=seed_set,
#                                         n_spots= 1 ,
#                                         coords=selected_coords) #args['n_spots']

#         # masked_element = np.where(mask==1, e_map, -1)
        

#         e_map = get_element_map(mineral_image=image,composition_df=composition_df,
#                 element=element_interest,noise=True,noise_value=noise_value) #0.1
#         croped_composition = e_map[mask.astype(bool)]
#         composition_per_window.append(croped_composition)
#     return composition_per_window,e_map



# def semivariance_from_samples(samples):
#     """
#     Compute experimental semivariance for a 1D array of samples.
#     Missing / sentinel values (<= -1) are removed.
#     Returns (gamma, n_pairs, n_valid_samples).
#     If fewer than 2 valid samples, returns (np.nan, 0, n_valid).
#     """
#     arr = np.asarray(samples).ravel()
#     # remove missing sentinel values (your code uses -1)
#     valid = arr[arr > -1]   # keep only values > -1; change condition if you use different sentinel
#     n_valid = len(valid)
#     if n_valid < 2:
#         return np.nan, 0, n_valid

#     # compute all unique pairwise squared differences efficiently
#     # Use broadcasting to compute upper triangular differences
#     diffs = valid[:, None] - valid[None, :]
#     # take only i < j (upper triangle without diagonal)
#     iu = np.triu_indices(n_valid, k=1)
#     sq_diffs = diffs[iu]**2
#     N_pairs = sq_diffs.size
#     gamma = 0.5 * sq_diffs.mean()   # semivariance estimator
#     return float(gamma), int(N_pairs), int(n_valid)


def run_targeted_sampling(simulation,composition_df,args,targeted_minerals_idx,
                                  max_n_targets_per_mineral,mineral_dict,
                                  element_interest = 'Fe',noise_value=0):

    # Set the seed for the random number generator
    np.random.seed(args['main_seed'])


    # Create an instance of the ImageSampler class
    sampler = ImageSampler(**args)
    args['n_classes']  = args['n_classes']-1

    args['n_spots'] = 1

    # Access the generated image
    image = sampler.main_image 


    croped_composition_all = []
    masked_e_map_all = []

    for n_targets_per_mineral in range(max_n_targets_per_mineral):
        # print(n_targets_per_mineral)

        # Start with empty coords
        selected_coords=[]
        # Loops through selected minerals
        for mineral_id in targeted_minerals_idx: 

            # Get pixel coord where the mineral is present in the image (assumes we have an image class map)
            min_idx = np.where(image==mineral_id)


            ## Call function that activly filters coords within the image
            filtered_coords = sampler.filter_coords(min_idx)
            # get x,y positions
            mineral_coords = (filtered_coords[0],filtered_coords[1])


            #Now we have the coordinates of each mineral in the image
            # We will randomly select #n target spots per mineral
            assert len(mineral_coords[0]) >= n_targets_per_mineral, f"Not enough non-overlapping occurrences of mineral: {mineral_dict[mineral_id]}, to sample {n_targets_per_mineral} targets out of {n_targets_per_mineral} occurrences. Consider reducing n_targets_per_mineral, reduncing void proportion or increasing the image size."
            np.random.seed(args['main_seed']+simulation) 
            random_target_idx = np.random.choice(len(mineral_coords[0]), size=n_targets_per_mineral, replace=False)

            # organize
            random_target_coords_per_mineral = [(mineral_coords[0][i],mineral_coords[1][i]) for i in random_target_idx]
            selected_coords.append(random_target_coords_per_mineral)
        selected_coords = [np.array(item, dtype=int) for sublist in selected_coords for item in sublist]

        seed_set = np.random.randint(0, args['spot_size']*args['spot_size']*(args['spot_size']**2), size=args['spot_size']*args['spot_size']*(args['spot_size']**2))

        mask,coords = sampler.sampling_mask(seed_set=seed_set,
                                        n_spots=args['n_spots'] ,
                                        coords=selected_coords)


        #Same image for every sampling, but the masks change
        e_map = get_element_map(mineral_image=image,composition_df=composition_df,
            element=element_interest,noise=True,noise_value=noise_value) #0.1

        masked_e_map = np.where(mask==1,e_map,0)
        croped_composition = e_map[mask.astype(bool)]

        croped_composition_all.append(croped_composition)
        masked_e_map_all.append(masked_e_map)
    return croped_composition_all, e_map,masked_e_map_all




def get_probabilities(phases_proportion,phases_composition, void_proportion):
    import streamlit as st
    phases_proportion_void = phases_proportion.copy()
    # st.write(phases_proportion.columns)
    p = [phases_proportion[c]/phases_proportion.sum(axis=1).iloc[0] * (100-void_proportion) for c in phases_proportion.columns]
    phases_proportion_void = pd.concat(p,axis=1)
    phases_proportion_void['background'] = void_proportion
    # st.write(phases_proportion_void)



    probs = phases_proportion_void/int(phases_proportion_void.sum(axis=1).iloc[0]) #/sum(phases_proportion_void['phase_proportion'])
    # st.write(probs)
    probs = probs.loc['phase_proportion'].to_list()
    # st.write(probs)

    probs = [round(p,6) for p in probs]
    # st.write(probs)

    phases_composition_void = phases_composition.copy()
    phases_composition_void.loc[:,'phase_proportion'] =probs
    # st.write(probs)
    # st.write('end ')


    return phases_proportion_void,phases_composition_void,probs


def scaler(df,standarization=False,minmax=False):
    if standarization:
        # Create a StandardScaler object
        scaler = StandardScaler()
        # Fit and transform the data
    else:
        # Create a RobustScaler object
        scaler = RobustScaler(quantile_range=(5, 95))
        # Fit and transform the data
    df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
        
    if minmax:
        # Create a MinMaxScaler object
        scaler = MinMaxScaler()
        # Fit and transform the data
    df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

    return df


def pre_process_for_pca(data,targeted_minerals,n_targets_per_mineral,standarization = False ,minmax = True ):
    
    # Normalize data
    norm_data = scaler(data,standarization=standarization,minmax=minmax)
    
    targeted_spots = []

    for mineral in targeted_minerals:
        targeted_spots  = targeted_spots + [mineral]*n_targets_per_mineral

    #make a list of 'random' assign to fill the list to match the number of spots
    if len(targeted_spots) < data.shape[0]: #n_spots:
        non_targeted_spots = ['random'] * (int(norm_data.shape[0]) - len(targeted_spots))
        targeted_spots = targeted_spots + non_targeted_spots

    return norm_data, targeted_spots


def load_standard_composition(parent_folder, filename = 'monte-carlo3_composition.csv'):
    # phases_composition is the composition of each phase in the sample - represents the bulk chemical compisition of each phase
    # SiO2 in %wt, Ti, Ni, and Ce in ppm
    phases_composition = pd.read_csv(os.path.join(parent_folder,filename), sep=';') #get_phase_composition_multi()
    # phases_composition.drop('phase_composition', axis=1, inplace=True)



    # phases_proportion is the amount of each phase in the entire sample - represents the amout detect in a segmentation or manual classification
    phases_proportion = phases_composition[['mineral','phase_proportion']] # {'phase1' : 54, 'phase2' : 25, 'phase3' : 14,   'phase4' : 6.4, 'phase5' : 0.5, 'phase6' : 0.1}
    phases_proportion = phases_proportion.set_index('mineral').T
    # phases_proportion_sum = np.sum(phases_proportion,axis=1)


    # Mineral phases
    # mineral_phases = phases_composition['mineral'].unique()



    # # Lets obtain the mean sample compostion give all the phases and their contributions
    elements_analyzed = phases_composition.columns[2:]
    sample_mean_composition = {}
    for oxide in elements_analyzed:
        # sample_mean_composition[oxide] = np.round(sum([phases_composition.query('mineral == @phase')['phase_proportion'] *phases_composition.query('mineral == @phase')[oxide] for phase in mineral_phases]) / sum(phases_composition['phase_proportion']),2)
        sample_mean_composition[oxide] = np.round(np.sum(phases_composition['phase_proportion']*phases_composition[oxide])/np.sum(phases_composition['phase_proportion']),2)
    sample_mean_composition= pd.DataFrame(sample_mean_composition, index=[0])
    return phases_composition,phases_proportion,elements_analyzed,sample_mean_composition


# # ----------------------------------------------------------------
# # ----------------------------------------------------------------


# def get_phase_composition_single():
#     phases_composition = {'phase1' : {'SiO2': 60, 'Ti': 0.5, 'Ni':0.1, 'Ce':1}, 
#                     'phase2' : {'SiO2': 45, 'Ti': 2.3, 'Ni':43, 'Ce':0.1}, 
#                     'phase3' : {'SiO2': 100, 'Ti': 0.5, 'Ni':0, 'Ce':0.3},  
#                     'phase4' : {'SiO2': 43, 'Ti': 0.8, 'Ni':81, 'Ce':25}, 
#                     'phase5' : {'SiO2': 48, 'Ti': 0.3, 'Ni':12, 'Ce':400}, 
#                     'phase6' : {'SiO2': 0.2, 'Ti': 0.01, 'Ni':560000, 'Ce':0}}
#     return phases_composition


# def get_phase_composition_multi():

#     phases_composition = {
#         'phase1': {'SiO2': 60, 'Ti': 23, 'Ni': 0.1, 'Ce': 12, 'Sr': 1000, 'Ba': 154, 'Y': 3, 'La': 36, 'Cu': 3, 'Cr': 1, 'Sc': 3.1, 'Fe': 0.04, 'Zn': 67, 'Mo': 2.7, 'Pb': 0.5, 'Co': 0.2, 'Li': 34, 'Al': 19},
#         'phase2': {'SiO2': 45, 'Ti': 2.3, 'Ni': 43, 'Ce': 0.1, 'Sr': 234, 'Ba': 21, 'Y': 4500, 'La': 56, 'Cu': 21, 'Cr': 2795, 'Sc': 5386, 'Fe': 33, 'Zn': 38, 'Mo': 679, 'Pb': 284, 'Co': 29, 'Li': 84, 'Al': 0.7},
#         'phase3': {'SiO2': 100, 'Ti': 0.5, 'Ni': 0.01, 'Ce': 0.3, 'Sr': 12, 'Ba': 56, 'Y': 4, 'La': 7, 'Cu': 0.2, 'Cr': 0.1, 'Sc': 0.12, 'Fe': 0.02, 'Zn': 0.3, 'Mo': 2, 'Pb': 0.6, 'Co': 0.01, 'Li': 19, 'Al': 0.002},
#         'phase4': {'SiO2': 43, 'Ti': 0.07, 'Ni': 81, 'Ce': 2, 'Sr': 5, 'Ba': 0.3, 'Y': 0.1, 'La': 12, 'Cu': 567, 'Cr': 245, 'Sc': 23, 'Fe': 53, 'Zn': 23, 'Mo': 5, 'Pb': 1.6, 'Co': 67, 'Li': 11, 'Al': 0.1},
#         'phase5': {'SiO2': 41, 'Ti': 0.3, 'Ni': 12, 'Ce': 1705, 'Sr': 230, 'Ba': 532, 'Y': 2585, 'La': 642, 'Cu': 55, 'Cr': 17960, 'Sc': 275, 'Fe': 29, 'Zn': 532, 'Mo': 78, 'Pb': 85, 'Co': 83, 'Li': 1, 'Al': 21},
#         'phase6': {'SiO2': 0.2, 'Ti': 0.01, 'Ni': 56000, 'Ce': 0, 'Sr': 0.2, 'Ba': 3, 'Y': 5, 'La': 0.2, 'Cu': 6, 'Cr': 45, 'Sc': 2, 'Fe': 45, 'Zn': 2, 'Mo': 6600, 'Pb': 11978, 'Co': 551, 'Li': 0.01, 'Al': 0.02}
#     }
#     return phases_composition

# ----------------------------------------------------------------
# ----------------------------------------------------------------
def image_composition(img,elements_analyzed,composition):
    unique,counts = np.unique(img, return_counts=True)

    raw_df = pd.DataFrame({'mineral':u,'count':c} for u,c in zip(unique,counts))

    raw_df = raw_df[raw_df['mineral'] !=  'background'] #raw_df.index[-1]]

    # composition = composition[composition['mineral'] != 'background']

    raw_df['percentage'] = raw_df['count']/sum(raw_df['count'])*100

    # import streamlit as st
    # st.dataframe(composition)
    # st.dataframe(raw_df)

    raw_df = raw_df.merge(composition, left_on=raw_df.index, right_index=True)
    composition_df = raw_df.loc[:,elements_analyzed]
    composition_df = composition_df * raw_df['percentage'].values.reshape(-1,1) / 100
    return composition_df, raw_df, unique,counts


def get_mineral_count(simulation_number,df_sampled, unique_sampled, n_classes):
    
    minerals_count_in_subset = pd.DataFrame(index=[simulation_number], columns=np.arange(0,n_classes))
    minerals_proportion_in_subset = pd.DataFrame(index=[simulation_number], columns=np.arange(0,n_classes))

    # minerals_count_in_subset.loc[simulation_number, unique_sampled] = df_sampled['count'] #counts/n_spots*100
    minerals_count_in_subset.loc[simulation_number, df_sampled.index] = df_sampled['count']
    minerals_proportion_in_subset.loc[simulation_number, unique_sampled] = df_sampled['percentage'] #counts
    for phase in np.arange(0,n_classes):
        if phase not in unique_sampled:
            # df_sampled.loc[phase,'count'] = 0
            minerals_count_in_subset.loc[simulation_number,phase] = 0
            minerals_proportion_in_subset.loc[simulation_number,phase] = 0

    return minerals_count_in_subset, minerals_proportion_in_subset




# ----------------------------------------------------------------
# Since standardzing or minmax scaling is not adequated for compositional data, let's implement some auxiliary functions, such as Centered Log-Ratio (CLR) Transformation 
# ----------------------------------------------------------------

"""Concept:
Divides each component of the composition by the geometric mean of all components.
Then takes the natural logarithm of the ratios.

Formula:
Let x_i be the i-th component of the composition.
g(x) is the geometric mean: g(x) = (x_1 * x_2 * ... * x_D)^(1/D) where D is the number of components.
CLR(x_i) = log(x_i / g(x))

Key Features:
Symmetric: Treats all components equally.
Interpretable: Relates each component to the overall average of the composition.
"""


def clr_transformation(df,ppm_columns=None):
    """
    Performs the Centered Log-Ratio (CLR) transformation on a compositional dataset.

    Args:

    This approach leverages the property that the mean of the logarithms of a set of numbers 
    is equal to the logarithm of their geometric mean.
    """
    # Check for zero or negative values
    if (df <= 0).any(axis=None):
        df.replace(0, 1e-15, inplace=True)
        import warnings
        warnings.warn("CLR transformation cannot be applied to data with zero or negative values."
                      "Zero was converted to 1e-15 to avoid undefined results.")

    # Convert DataFrame to longdouble for higher precision
    df_clr = df.astype(np.longdouble) 

    # Calculate the geometric mean of each row
    g = np.exp(np.mean(np.log(df), axis=1))
    g_series = pd.Series(g, index=df.index) # Create a series for faster division
    # Divide each component by the geometric mean and take the natural logarithm
    df_clr = df_clr.div(g_series, axis=0) # Divide each component by the geometric mean 

    
    df_clr = np.log(df_clr)

    return df_clr

def ppm_to_wt(df,ppm_columns=None):

    if ppm_columns == None:
        ppm_columns = ['Ni', 'Ce', 'Sr', 'Ba', 'Y', 'La', 'Cu', 'Cr', 'Sc',
       'Zn', 'Mo', 'Pb', 'Co', 'Li']
    df[ppm_columns] = df[ppm_columns] / 10000
    return df
    

import pandas as pd

def closure_data(df, target_sum=100):
  """
  Rescales each row of a DataFrame so that the sum of values in each row equals the target_sum.

  Args:
      df: pandas DataFrame containing the data.
      target_sum: The desired sum for each row (default: 100).

  Returns:
      pandas DataFrame with rescaled values.
  """

  # Calculate the sum of each row
  row_sums = df.sum(axis=1)

  # Rescale each row by dividing by the row sum and multiplying by the target_sum
  df_rescaled = df.div(row_sums, axis=0) * target_sum

  return df_rescaled




import math
def calculate_log_scale_widths(positions, original_position, original_width):
    """
    Calculates widths for boxplots at different positions on a log scale,
    maintaining visual consistency with an original boxplot.

    Args:
        positions: A list of positions for the boxplots.
        original_position: The position of the original boxplot.
        original_width: The width of the original boxplot.

    Returns:
        A list of widths corresponding to the input positions.
    """

    original_high = original_position + original_width
    log_original_position = math.log10(original_position)
    log_original_high = math.log10(original_high)
    log_width = log_original_high - log_original_position

    widths = []
    for new_position in positions:
        log_new_position = math.log10(new_position)
        log_new_high = log_new_position + log_width
        new_high = 10**log_new_high
        new_width = new_high - new_position
        widths.append(new_width)

    return widths


#### Get highest absolute loadings 
def get_top_two_abs_elements_list(df: pd.DataFrame) -> dict:
    """
    Identifies the two elements with the highest absolute values for each
    principal component (PC) and returns them as a dictionary of lists.

    Args:
        df: A pandas DataFrame where rows represent principal components (e.g., PC1, PC2)
            and columns represent elements with numerical values.

    Returns:
        A dictionary where keys are PC names and values are lists containing the
        names of the two elements with the highest absolute values (in descending order).
    """
    top_elements_list = {}
    for index, row in df.iterrows():
        # Calculate the absolute values of the row
        abs_row = row.abs()
        # Sort the absolute values in descending order
        sorted_abs_row = abs_row.sort_values(ascending=False)
        # Get the names of the top two elements based on absolute values
        top1 = sorted_abs_row.index[0]
        top2 = sorted_abs_row.index[1]
        # top3 = sorted_abs_row.index[3]
        top_elements_list[index] = [top1, top2]

    return top_elements_list



#### Get highest absolute loadings
def get_top_three_abs_elements_list(df: pd.DataFrame) -> dict:
    """
    Identifies the two elements with the highest absolute values for each
    principal component (PC) and returns them as a dictionary of lists.

    Args:
        df: A pandas DataFrame where rows represent principal components (e.g., PC1, PC2)
            and columns represent elements with numerical values.

    Returns:
        A dictionary where keys are PC names and values are lists containing the
        names of the two elements with the highest absolute values (in descending order).
    """
    top_elements_list = {}
    for index, row in df.iterrows():
        # Calculate the absolute values of the row
        abs_row = row.abs()
        # Sort the absolute values in descending order
        sorted_abs_row = abs_row.sort_values(ascending=False)
        # Get the names of the top two elements based on absolute values
        top1 = sorted_abs_row.index[0]
        top2 = sorted_abs_row.index[1]
        top3 = sorted_abs_row.index[2]
        top_elements_list[index] = [top1, top2,top3]

    return top_elements_list





# # ----------------------------------------------------------------
# # From here below, the old functions from sampling from table
# # ----------------------------------------------------------------

# def create_samples(phases_proportion, num_samples=10000, seed=42):
#     """
#     Returns a numpy array of samples with provided probabilities.

#     Args:
#     phases_proportion: Dict where keys are phases and values are their 
#                     proportions in percentage.
#     num_samples: The desired number of samples.

#     Returns:
#     A numpy array of samples.
#     """
#     # Set seed for reproducibility
#     np.random.seed(seed)
#     # Convert percentages to probabilities
#     probabilities = [value / 100 for value in phases_proportion.values()]

#     # Create samples using numpy.random.choice - random sampler from np
#     samples = np.random.choice(list(phases_proportion.keys()), 
#                             size=num_samples, 
#                             p=probabilities,
#                             replace=True)

#     return samples

# def create_samples_df(phases_proportion, num_samples=10000, seed=42):
#     """
#     Returns a numpy array of samples with provided probabilities.

#     Args:
#     phases_proportion: Dict where keys are phases and values are their 
#                     proportions in percentage.
#     num_samples: The desired number of samples.

#     Returns:
#     A numpy array of samples.
#     """
#     # Set seed for reproducibility
#     np.random.seed(seed)
#     # Convert percentages to probabilities
#     probabilities = [phases_proportion[phase].item() / 100 for phase in phases_proportion.columns]


#     # Create samples using numpy.random.choice - random sampler from np
#     samples = np.random.choice(list(phases_proportion.keys()), 
#                             size=num_samples, 
#                             p=probabilities,
#                             replace=True)

#     return samples


# def subsample_multiphases(samples,mixture_size=10, n_spots=100, seed=0):
#     """
#     Returns a numpy array of subsamples uniformly sampled from the given samples.

#     Args:
#     samples: A numpy array of samples.
#     mixture_size: The number of phases in each subsample.
#     n_spots: The number of spots in each subsample.
#     seed: The seed for reproducibility.

#     Returns:
#     A numpy array of subsamples: each row is a subsample; each column is a phase.
#     Each of these subsamples is one unity of subset simulated.
#     """

#     # Set seed for reproducibility
#     np.random.seed(seed)
#     subsamples = np.empty((n_spots,mixture_size), dtype="<U10")
#     for mix in range(0,mixture_size):
#         subsamples[:,mix]  = np.random.choice(samples, 
#                             size=n_spots, 
#                             replace=False)
#     return subsamples



# import pandas as pd
# def sample_one_subset_multiphase(samples,phases_composition,elements_analyzed,mixture_size=10, n_spots=100, seed=0):
#     '''
#     This performs the entire subsampling of multiphase samples for one subset.
#     It returns the composition of the spots in the subset and the mean composition of the spots in the subset.

#     The mean here refers to the mean obtained in each spot, not the mean of the subset.
#     '''
#     # This is the df with the phases being sample in each shot
#     subsample_group = pd.DataFrame(subsample_multiphases(samples,mixture_size=mixture_size, n_spots=n_spots, seed=seed))

#     #start empty df to store the composition of the phases and the means
#     subset_spots_composition = pd.DataFrame()
#     mean_spot_composition = pd.DataFrame(columns=elements_analyzed)
#     # mean_spot_composition.index.name = 'spot_group'
#     for spot_group in range(0,subsample_group.shape[0]):
#         #make a temporary df to keep the composition of the phases in the spot group
#         spot_composition = pd.DataFrame(columns=['mineral','spot_group'])
#         #assign the minerals
#         spot_composition['mineral'] = subsample_group.loc[spot_group,:]
#         #merge with the df that has the composition of the phases
#         spot_composition = spot_composition.merge(phases_composition, on='mineral', how='left')
#         spot_composition = spot_composition.drop('phase_proportion',axis=1)
#         #assign the spot group
#         spot_composition['spot_group'] = spot_group

#         #append to the final df with all 'raw' data for every particle in every spot
#         subset_spots_composition = pd.concat([subset_spots_composition,spot_composition],axis=0)
        
#         # store the mean composition of each group per element in the columns and the spot_group as index
#         mean_spot_composition.loc[spot_group] = spot_composition.describe().loc['mean',:]



        
#     #this is the df with all the spots and their composition
#     subset_spots_composition.reset_index(drop=True, inplace=True)

#     return subset_spots_composition,mean_spot_composition



# # --------------------------------------------------------------
# def sample_all_subsets_multiphase(samples,phases_composition,elements_analyzed,phases_proportion,
#                                   mixture_size=10,n_spots=10,n_subsets=10,
#                                   randomized_mixture_size=False,subset_seeds=None):
#     simulated_subsets = {}
#     for i in range(n_subsets):
#         if n_spots>200:
#             print(f"Simulating subset {i+1} of {n_subsets}")
#         if i == 0:
#             mineral_count = pd.DataFrame(columns=phases_proportion.columns)
#             mean_composition_per_subset = pd.DataFrame(columns=elements_analyzed)
#         if randomized_mixture_size:
#             np.random.seed(subset_seeds[i])
#             mixture_size = np.random.randint(5, 20) # this will randomly generate a mixture size for each subset
#         mean_subset_composition = pd.DataFrame(columns=elements_analyzed)
#         subset_spots_composition,mean_spot_composition = sample_one_subset_multiphase(samples,
#                                                                                     phases_composition,
#                                                                                     elements_analyzed,
#                                                                                     mixture_size=mixture_size, 
#                                                                                     n_spots=n_spots, 
#                                                                                       seed=subset_seeds[i])
#         mean_subset_composition = mean_spot_composition.describe().loc['mean',:]
#         simulated_subsets[i] = {'subset_spots_composition':subset_spots_composition,
#                                 'mean_spot_composition':mean_spot_composition,
#                                 'mean_subset_composition':mean_subset_composition}

#         mean_composition_per_subset.loc[i] = mean_subset_composition
#                 # # Gets the %of minerals sampled in each subset



#         unique, counts = np.unique(simulated_subsets[i]['subset_spots_composition']['mineral'],return_counts=True)
#         minerals_count_in_subset = pd.DataFrame(index=[i], columns=phases_proportion.columns)  # Create a row for each subset
#         minerals_count_in_subset.loc[i, unique] = np.round(counts/(n_spots*mixture_size)*100,2)   # Assign counts to corresponding minerals
#         for phase in phases_proportion.columns:
#             if phase not in unique:
#                 minerals_count_in_subset.loc[i,phase] = 0

#         mineral_count = pd.concat([mineral_count,minerals_count_in_subset],axis=0)


#     return simulated_subsets, mineral_count,mean_composition_per_subset





# ----------------------------------------------------------------
# The functions below are the old functions incorporated to the ImageSampler.py
# ----------------------------------------------------------------



# def create_image(m, n_classes, probs, seed=None):
#     """
#     Creates an m x m image with n_classes and specified class probabilities.

#     Args:
#     m: Size of the image (m x m).
#     n_classes: Number of distinct classes.
#     probs: List of probabilities for each class. 
#             Must sum to 1.0.
#     seed: Random seed for reproducibility.

#     Returns:
#     A numpy array representing the image.
#     """
#     if seed is not None:
#         np.random.seed(seed)

#     if len(probs) != n_classes or not np.isclose(sum(probs), 1.0):
#         raise ValueError("Invalid probabilities. Must sum to 1.0 and match n_classes.")

#     # Create a cumulative probability distribution
#     cum_probs = np.cumsum(probs)

#     # Generate random numbers for each pixel
#     image = np.random.rand(m, m)

#     # Assign class labels based on the cumulative probabilities
#     for i in range(n_classes):
#         image[(image >= cum_probs[i -1]) & (image < cum_probs[i])] = i 

#     return image.astype(int)


# def sampling_mask(image_size, spot_size, n_spots,seed_set):
#     # Lets try another approach, by creating a mask of the image and removing the spots
    
#     mask = np.zeros((image_size, image_size))
#     coords = []
#     counter=0
#     s=0
#     assert image_size >= spot_size, "spot_size must be smaller than image_size"
#     assert n_spots*(spot_size**2) <= (image_size - spot_size)**2, "n_spots must be smaller than the number of possible spots"
    
#     while counter < n_spots:
#         np.random.seed(seed_set[s])
#         coord = np.random.randint(0, image_size - spot_size, size=2)
#         if  mask[coord[0]:coord[0]+spot_size, coord[1]:coord[1]+spot_size].sum() == 0:
#             mask[coord[0]:coord[0]+spot_size, coord[1]:coord[1]+spot_size] = 1
#             coords.append(coord)
#             counter += 1
#         s+=1
#     return mask, coords


# def sampling_mask_targeted(image_size, spot_size, n_spots,seed_set,coords):
#     # Lets try another approach, by creating a mask of the image and removing the spots
    
#     mask = np.zeros((image_size, image_size))

#     counter=len(coords)

#     s=0
#     assert image_size >= spot_size, "spot_size must be smaller than image_size"
#     assert n_spots*(spot_size**2) <= (image_size - spot_size)**2, "n_spots must be smaller than the number of possible spots"
    
    
#     #Lets include in the mask the pre-select coordinates
#     for coord in coords:
#         mask[coord[0]:coord[0]+spot_size, coord[1]:coord[1]+spot_size] = 1

#     while counter < n_spots:
#         np.random.seed(seed_set[s])
#         coord = np.random.randint(0, image_size - spot_size, size=2)
#         if  mask[coord[0]:coord[0]+spot_size, coord[1]:coord[1]+spot_size].sum() == 0:
#             mask[coord[0]:coord[0]+spot_size, coord[1]:coord[1]+spot_size] = 1
#             coords.append(coord)
#             counter += 1
#         s+=1
#     return mask, coords



# def filter_coords(min_idx, image_size, spot_size):
#   """
#   Filters coordinates to ensure they are within bounds for a given spot size.

#   Args:
#     min_idx: A tuple containing arrays of row and column indices.
#     image_size: A tuple containing the height and width of the image.
#     spot_size: The size of the spot to be placed at the coordinates.

#   Returns:
#     A tuple containing filtered arrays of row and column indices.
#   """
#   rows, cols = min_idx
#   valid_mask = (rows + spot_size < image_size) & (cols + spot_size < image_size)
#   return rows[valid_mask], cols[valid_mask]

# def pre_selected_coords(image, args,selected_coords=[]):
    
#     for mineral_id in range(args['n_classes']):
#         # Get pixel coord where the mineral is present in the image (assumes we have an image class map)
#         min_idx = np.where(image==mineral_id)

#         filtered_coords = filter_coords(min_idx, args['image_size'], args['spot_size'])
#         mineral_coords = (filtered_coords[0],filtered_coords[1])

#         #Now we have the coordinates of each mineral in the image
#         # We will randomly select #n target spots per mineral
#         random_target_idx = np.random.choice(len(mineral_coords[0]), size=args['n_target_sample_per_mineral'], replace=False)

#         random_target_coords_per_mineral = [(mineral_coords[0][i],mineral_coords[1][i]) for i in random_target_idx]
#         selected_coords.append(random_target_coords_per_mineral)
#     selected_coords = [np.array(item, dtype=int) for sublist in selected_coords for item in sublist]
#     return selected_coords

from sklearn.decomposition import PCA

def get_pca_data(norm_data, n_components, targets):

    mask = ~norm_data.isna().any(axis=1)

    norm_data = norm_data[mask]
    targets   = np.asarray(targets)[mask.values]   

    pca = PCA(n_components=n_components).fit(norm_data)

    comps = pca.transform(norm_data)

    per_var = np.round(pca.explained_variance_ratio_ * 100, 1)
    labels = [f'PC{i}' for i in range(1, len(per_var)+1)]

    components = pd.DataFrame(
        comps,
        index=norm_data.index,
        columns=labels
    )

    components["target"] = targets

    loadings_df = pd.DataFrame(
        pca.components_,
        columns=norm_data.columns,
        index=labels
    )

    return pca, components, loadings_df, per_var







