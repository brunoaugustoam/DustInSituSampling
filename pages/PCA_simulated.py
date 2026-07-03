

import sys
import os

from pathlib import Path
root = os.getcwd()
parent_folder = str(Path(root).parent)
sys.path.append(parent_folder)
HERE = Path(__file__).parent
sys.path.append(str(HERE / '..'))
from MultivariateAnalysis import PcaPlotter

import streamlit as st

from utils import  *
from plots import  *
from ImageSampler import ImageSampler, ImagePlotter

import matplotlib
matplotlib.rcParams.update(matplotlib.rcParamsDefault)


st.set_page_config(layout="wide")


st.title("PCA from single Simulation")


#############################################################################################
col1, col2 = st.columns(2)
with col1:
    main_seed = st.number_input('Select main random seed | this will change the random numbers',min_value=0, max_value=100,value =42)
with col2:
    n_spots = st.selectbox('Select number of spots ', [20,50,100,200,500,1000,2000,5000], index=2)


#With sidebar from Streamlit
with st.sidebar:
    # '''
    # main_seed : int, seed for the random number generator, will also control the order of other seeds arrays
    # image_size : int, size of the image to be created such as image_size x image_size
    # n_spots : int, number of spots to sample in the image
    # spot_size : int, size of the spot to sample, such as spot_size x spot_size
    # n_simulations : int, number of simulations to run
    # '''
    image_size  = st.number_input('Select image size (size x size)',min_value=0, max_value=10000,value =1000)

    
    
    void_proportion = st.number_input('Select background proportion (%)',min_value=0, max_value=99,value =50)
    spot_size = st.selectbox('Select spot size (will be squared = #particles sampled per spot)',[1,2,3,4,5,7,10,17,33],index=5)
    

    
n_simulations  = 1


# Import standard composition data, stored in a csv file - will be used to compare the results of subsamplig
phases_composition_orig,phases_proportion,elements_analyzed,sample_mean_composition = load_standard_composition(root, filename='monte_carlo_real_compositions.csv')


minerals_input = list(phases_composition_orig['mineral'])

phases_composition_standard = phases_composition_orig.copy()
phases_composition_standard.loc[len(phases_composition_orig)] = pd.Series({'mineral':'background','phase_proportion':void_proportion})


phases_proportion_void,phases_composition_void,probs = get_probabilities(phases_proportion,phases_composition_standard,void_proportion)

# Get minerals list and number of classes
minerals = list(phases_composition_void['mineral'])
n_classes =  len(minerals)
 
with st.sidebar:
    targeted_minerals = minerals_input

    targeted_minerals = st.multiselect('Selected minerals to target',minerals_input,targeted_minerals)
    if n_spots <= 20:
        n_targets_per_mineral = st.selectbox('Select minimum # of target spots | zero to disable', 
                                            [0,1,2], index=0)
    else:
        n_targets_per_mineral = st.selectbox('Select minimum # of target spots | zero to disable', 
                                            [0,1,2,5,10], index=0)
    
    conversion_spot_libs_diameter=  pd.Series({1:5, 3:15, 5:25, 7:37, 10:55, 33:180})
    conversion_spot_libs_diameter = conversion_spot_libs_diameter.to_frame(name='Diameter (um)')
    conversion_spot_libs_diameter.index.name = 'Spot size'
    st.info('Conversion of spot size to diameter in micrometers')
    st.dataframe(conversion_spot_libs_diameter)

    
    
targeted_minerals_idx = [minerals.index(item) for item in targeted_minerals]


mineral_dict = {
    0 : 'Ab',
    1 : 'Qz',
    2 : 'Mc',
    3 : 'Spd',
    4 : 'Brl',
    5 : 'Cb',
    6 : 'background'
    }



mineral_to_color= {
    'Ab' : 'blue',
	'st-Ab' : 'darkblue',

    'Qz': 'orange',
	'st-Qz': 'darkorange',
	
    'Mc' : 'green',
	'st-Mc' : 'forestgreen',

    'Spd' : 'violet',
	'st-Spd' : 'darkviolet',

    'Brl' : 'brown',
	'st-Brl' : 'saddlebrown',

    'Cb': 'cyan',
	'st-Cb': 'darkturquoise',

    'random' : 'black' ,

    'st' : 'magenta'
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

#############################################################################################



# Set the seed for the random number from this specific simulation
np.random.seed(args['main_seed'])
# Provide a unique series of seeds to input the sampler - this ensures that each simulation is unique and fixing the sets allow reporoducibility in future experiments
seed_set = np.random.randint(0, n_spots*n_spots*(spot_size**2), size=n_spots*n_spots*(spot_size**2))

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



# gets unique minerals and counts in the masked image
unique,counts = np.unique(masked_imge, return_counts=True)

# Builds a dataframe with the mineral count
raw_df = pd.DataFrame({'mineral':u,'count':c} for u,c in zip(unique,counts) if u != args['n_classes'])
if raw_df.empty:
    st.error(f"No minerals were sampled. Please try different parameters.")
    

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


spot_composition,phases_composition_simulation,stacked_flat_spots = sampler.get_mean_spot_composition(coords,phases_composition_standard)

mean_spot_composition = spot_composition.mean()



# Create an instance of the ImagePlotter class
variaton_to_plot = 0.1


plotter = ImagePlotter(sampler, mask=mask, coords=coords ,masked_imge=masked_imge, main_image_composition=main_image_composition)

# Plot the masked image
show_masked_image =  st.checkbox('Show masked image', value=False)
if show_masked_image :
    fig1,fig2 = plotter.plot_masked_image(minerals, web=True)

    col1,col2 = st.columns(2)
    col1.pyplot(fig1, use_container_width=True)
    col2.pyplot(fig2, use_container_width=True)

#########################################################################################
mineral_proportion = minerals_proportion_in_subset.to_frame().T
mineral_count = minerals_count_in_subset.to_frame().T


composition_to_display = pd.DataFrame({'Simulation composition': mean_spot_composition,
                                        'Image composition': main_image_composition,})
                                        



#############################################################################################
standarization = False #st.checkbox("Standardize | Default is to median (RobustScaler)", value=False)
minmax = True #st.checkbox("MinMaxScaler (scale data between 0 and 1)", value=True)

elements = [i for i in elements_analyzed if i != 'Al'] 

data = spot_composition.copy()
data = data.dropna().reset_index(drop=True)

data = data.loc[:,elements]

data_combined = pd.concat([data, phases_composition_orig.loc[:,elements]]).reset_index(drop=True)

standards = ['st-'+i for i in phases_composition_orig['mineral']]



data_norm, targets_pre_processed = pre_process_for_pca(data_combined, targeted_minerals,n_targets_per_mineral)
targets_pre_processed[data.shape[0]:] = standards




#-------------------PCA-------------------

legend = False
# Actually call the function and plot the PCA
@st.fragment
def call_pca_plotter(norm_data,targets_pre_process):
    n_components = st.number_input("Number of components",min_value=2, max_value=10,value =3)
    pca, components, loadings_df,explained_variance = get_pca_data(norm_data,n_components, targets_pre_process)
    # Get these two selectbox as two columns side by side
    col1, col2 = st.columns(2)
    with col1:
        x_axis_pca = st.selectbox('Select x-axis PC', list(np.arange(1,n_components+1)), index=0)
    with col2:
        y_axis_pca = st.selectbox('Select y-axis PC', list(np.arange(1,n_components+1)), index=1)
    pc_to_plot = [x_axis_pca-1, y_axis_pca-1]  # Convert to zero-based index
    fig = plot_pca(components, loadings_df, pcs=pc_to_plot, standards=standards,minerals=targeted_minerals,
                mineral_to_color=mineral_to_color, 
                    legend=legend,void_n=void_proportion)
    st.pyplot(fig, use_container_width=False)
    st.info("Visualizing the loadings of the principal components")

    fig = reversed_matrix_plot_pca_loadings(components,loadings_df,num_cols=n_components, 
                                            pca=pca,
                                            mineral_to_color=mineral_to_color)
    st.pyplot(fig, use_container_width=False)

    return pca,components, loadings_df, pc_to_plot,n_components




pca,components, loadings_df, pc_to_plot,n_components = call_pca_plotter(data_norm,targets_pre_processed)


