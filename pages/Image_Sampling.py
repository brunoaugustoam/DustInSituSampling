"""
Author: Bruno Monteiro - Mcgill University
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import sys
import os
from pathlib import Path
root = os.getcwd()
parent_folder = str(Path(root).parent)
sys.path.append(parent_folder)
HERE = Path(__file__).parent
sys.path.append(str(HERE / '..'))
from utils import  *
from plots import  *
from ImageSampler import ImageSampler, ImagePlotter

#Strealit layout config
st.set_page_config(layout='wide')

#With sidebar from Streamlit
main_seed = st.number_input('Select main random seed | this will change the random numbers in all simulations',min_value=0, max_value=100,value =42)

n_simulations  = st.number_input('Select number of simulations ',min_value=0, max_value=100,value =100)
with st.sidebar:
    # '''
    # main_seed : int, seed for the random number generator, will also control the order of other seeds arrays
    # image_size : int, size of the image to be created such as image_size x image_size
    # n_spots : int, number of spots to sample in the image
    # spot_size : int, size of the spot to sample, such as spot_size x spot_size
    # n_simulations : int, number of simulations to run
    # '''
    
    image_size  = st.number_input('Select image size (size x size)',min_value=0, max_value=10000,value =1000)
    n_spots = st.selectbox('Select number of spots ', [10,20,50,100,200,500,1000,10000,20000], index=2)
    
    void_proportion = st.number_input('Select background proportion (%)',min_value=0, max_value=99,value =50)
    spot_size = st.selectbox('Select spot size (will be squared = #particles sampled per spot)',[1,2,3,4,5,7,10,17,33],index=5)
    conversion_spot_libs_diameter=  pd.Series({1:5.3, 3:16, 5:26, 7:37, 10:53, 33:175})
    conversion_spot_libs_diameter = conversion_spot_libs_diameter.to_frame(name='Diameter (um)')
    conversion_spot_libs_diameter.index.name = 'Spot size'

    



# Import standard composition data, stored in a csv file - will be used to compare the results of subsamplig
phases_composition_orig,phases_proportion,elements_analyzed,sample_mean_composition = load_standard_composition(root, filename='monte_carlo_real_compositions.csv')

st.info("Input proportions and compositions")
st.dataframe(phases_composition_orig)

minerals_input = list(phases_composition_orig['mineral'])

phases_composition_standard = phases_composition_orig.copy()
phases_composition_standard.loc[len(phases_composition_orig)] = pd.Series({'mineral':'background','phase_proportion':void_proportion})


phases_proportion_void,phases_composition_void,probs = get_probabilities(phases_proportion,phases_composition_standard,void_proportion)

# Get minerals list and number of classes
minerals = list(phases_composition_void['mineral'])
# st.write( minerals)
n_classes =  len(minerals)
 

with st.sidebar:

    targeted_minerals = st.multiselect('Selected minerals to target',minerals_input,minerals_input)
    n_targets_per_mineral = st.selectbox('Select minimum # of target spots | zero to disable', 
                                            [0,1,2,3,4,5,10], index=0)
    st.info('Conversion of spot size to diameter in micrometers')
    st.dataframe(conversion_spot_libs_diameter)
    
targeted_minerals_idx = [minerals.index(item) for item in targeted_minerals]
# st.write(minerals,targeted_minerals_idx)


mineral_dict = {
    0 : 'Ab',
    1 : 'Qz',
    2 : 'Mc',
    3 : 'Spd',
    4 : 'Brl',
    5 : 'Cb',
    6 : 'background'
    }


args = {
    'image_size': image_size,
    'minerals' : minerals, 
    'n_simulations' : n_simulations,
    'n_classes': n_classes,
    'probs': probs,
    'spot_size': spot_size,
    'main_seed': main_seed,
    'elements_analyzed': list(elements_analyzed),
}

# Set the seed for the random number generator
np.random.seed(args['main_seed'])

# Create an instance of the ImageSampler class
sampler = ImageSampler(**args)
args['n_classes']  = args['n_classes']-1
args['targeted_minerals']  = targeted_minerals
# Access the generated image
image = sampler.main_image 



# Get the unique minerals in the image and their count - thats the real composition and must approximalty match the sample mean from the csv file
composition_df, raw_df,u,c = image_composition(image,args['elements_analyzed'],phases_composition_void)


main_image_composition = composition_df.sum()



sampled_whole_image_composition = pd.DataFrame(columns=args['elements_analyzed'])
sampled_image_composition = pd.DataFrame(columns=args['elements_analyzed'])
sampled_whole_image_mineral_count = {}

for simulation in range(n_simulations):
    if simulation % 20 == 0:
        st.write(f'Simulation {simulation} / {n_simulations}')
    

    # Set the seed for the random number from this specific simulation
    np.random.seed(args['main_seed'] + simulation)
    # Provide a unique series of seeds to input the sampler - this ensures that each simulation is unique and fixing the sets allow reporoducibility in future experiments
    seed_set = np.random.randint(0, n_spots*n_spots*(spot_size**2),size=n_spots*n_spots*(spot_size**2))

    # '''
    # Here, to target a specific mineral, we will provide some pre-defined coordinates
    # The coordinates can be selected with np.where to get the indexes of the mineral we want to target
    # Then we can filter the coordinates to avoid the borders of the image
    # '''
    
    if n_targets_per_mineral>0:
        # Here must filter out the minerals to be analysed, instead of all
        selected_coords = sampler.pre_selected_coords(targeted_minerals_idx,n_targets_per_mineral,mineral_dict)
        
    else:
        selected_coords = []


    # Now we can sample the image randomly but already targeting some minerals
    mask,coords = sampler.sampling_mask(seed_set=seed_set, n_spots=n_spots ,coords=selected_coords)


    # Create masked image overlaying mask over the original image
    # the -1 is because the last class is the void, and the indexing starts at 0
    masked_imge = np.where(mask==1, image, args['n_classes'])

    # x = np.unique(masked_imge, return_counts=True)
    
    # gets unique minerals and counts in the masked image
    unique,counts = np.unique(masked_imge, return_counts=True)

    # Builds a dataframe with the mineral count
    raw_df = pd.DataFrame({'mineral':u,'count':c} for u,c in zip(unique,counts) if u != args['n_classes'])
    if raw_df.empty:
        st.error(f"No minerals were sampled in simulation {simulation}. Please try different parameters.")
        continue
    
    # Remove the background class from the composition - redundancy
    raw_df = raw_df[raw_df.index !=  args['n_classes']]


    composition = phases_composition_void[phases_composition_void['mineral'] != 'background']


    raw_df['mineral_key'] = raw_df['mineral'].astype(int)
    # Use the mineral dictionary to map the mineral names to their indices
    raw_df['mineral'] = raw_df['mineral'].map(mineral_dict)
    # Merge with the composition data
    raw_df = pd.merge(left=raw_df, right=composition, on='mineral', how='inner')


    # Calculate the percentage of each mineral in the masked image
    raw_df['percentage'] = raw_df['count'] / sum(raw_df['count']) * 100
    


    subset_df =  raw_df.set_index('mineral') 
    
    for mineral in minerals_input:
        if mineral not in subset_df.index:
            # If the mineral is not in the subset, set its count and percentage to 0
            subset_df.loc[mineral, 'count'] = 0
            subset_df.loc[mineral, 'percentage'] = 0
    minerals_count_in_subset = subset_df['count'].T
    minerals_proportion_in_subset = subset_df['percentage'].T



    # Auxiliary function to get mineral count and proportion in the sampled subset
    if simulation == 0:

        mineral_count = pd.DataFrame(columns=minerals_input, index=np.arange(0,n_simulations) )

        mineral_proportion =  pd.DataFrame(columns=minerals_input, index=np.arange(0,n_simulations) )
    
    mineral_count.loc[simulation] = minerals_count_in_subset
    mineral_proportion.loc[simulation] = minerals_proportion_in_subset
    

    mean_spot_composition,phases_composition_simulation,stacked_flat_spots = sampler.get_mean_spot_composition(coords,phases_composition_standard)
    sampled_image_composition.loc[simulation] = mean_spot_composition.mean() # this is the same as the sampled_whole_image_composition






st.write('One example of sampled masked, showing last simulation')

# Create an instance of the ImagePlotter class
variaton_to_plot = 0.1
plotter = ImagePlotter(sampler, mask=mask, coords=coords ,masked_imge=masked_imge, main_image_composition=main_image_composition)

# Plot the masked image

fig1,fig2 = plotter.plot_masked_image(minerals, web=True)

col1,col2 = st.columns(2)
col1.pyplot(fig1, use_container_width=True)
col2.pyplot(fig2, use_container_width=True)

#----------------------
# Plotting the percentage of each mineral per simulation
# #----------------------
# Dictionary for renaming columns

new_col_names = {}
for i,phase in enumerate(minerals):
    new_col_names[i] = phase

# mineral_count = mineral_count.drop(6, axis=1)
# mineral_proportion = mineral_proportion.drop(6, axis=1)

mineral_count = mineral_count.rename(columns=new_col_names)
mineral_proportion = mineral_proportion.rename(columns=new_col_names)


fig_count = plotter.mineral_count_per_simulation(mineral_count,mineral_proportion,phases_proportion)
st.pyplot(fig_count, use_container_width=True)


#----------------------
# Plotting the composition provided per spot against the expected mean composition
#----------------------
fig_element = plotter.mean_element_composition_per_simulation(sampled_image_composition,columns=4)
st.pyplot(fig_element, use_container_width=True)
# ----------------------
