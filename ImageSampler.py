import numpy as np
import pandas as pd
import sys
import os
from pathlib import Path
import streamlit as st
root = os.getcwd()
parent_folder = str(Path(root).parent)
sys.path.append(parent_folder)
from utils import  *
from plots import  *




class ImageSampler():   
    def __init__(self,image_size, minerals, n_classes, probs, spot_size, n_simulations, elements_analyzed,main_seed=None):
        """
        Initializes an ImageSampler object.
        """
        self.image_size = image_size
        self.minerals = minerals
        self.n_classes = n_classes
        self.probs = probs
        self.spot_size = spot_size
        self.n_simulations = n_simulations
        # self.n_targets_per_mineral = n_targets_per_mineral
        # self.targeted_minerals_idx = targeted_minerals_idx
        self.elements_analyzed = elements_analyzed

        self.main_seed = main_seed

        self.main_image = self.create_image(self.main_seed)
        assert self.main_image is not None, 'No data found'
        assert len(self.main_image.shape) == 2, 'Wrong data shape'
        assert self.main_image.shape[0] == self.main_image.shape[1], 'Image is not squared'
       

    def create_image(self, seed=None):
        # import streamlit as st
        """
        Creates an m x m image with n_classes and specified class probabilities.

        Args:
        m: Size of the image (m x m).
        n_classes: Number of distinct classes.
        probs: List of probabilities for each class. 
                Must sum to 1.0.
        seed: Random seed for reproducibility.

        Returns:
        A numpy array representing the image.
        """
        if seed is not None:
            np.random.seed(seed)

        # Check if the probabilities are valid
        # the +1 is to account for the background class
        if len(self.probs) != self.n_classes or not np.isclose(sum(self.probs), 1.0,atol=0.05):
            raise ValueError("Invalid probabilities. Must sum to 1.0 and match n_classes.")

        # Create a cumulative probability distribution
        cum_probs = np.cumsum(self.probs)
        # st.write(cum_probs)
    
        # Generate random numbers for each pixel
        image = np.random.rand(self.image_size, self.image_size)
        # st.write('img generated')
        # st.write(np.histogram(image))


        # Assign class labels based on the cumulative probabilities
        image[image < cum_probs[0]] = 10
        # image[(image >= cum_probs[0]) & (image < cum_probs[1])] = 1
        for i    in range(11,10+self.n_classes):
            # st.write(i,'/',self.n_classes)
            # st.write(cum_probs[i-10-1])
            # st.write(cum_probs[i-10])
            

            # label_maks = np.where((image >= cum_probs[i -1]) & (image < cum_probs[i]))
            # image[label_maks] = i

            image[(image >= cum_probs[i-10 -1]) & (image < cum_probs[i-10])] = i
            # st.write(np.histogram(image))
        # image[image >= cum_probs[self.n_classes -1] ] = self.n_classes-1
        image = image.astype(int) 
        image = image - 10
        # st.write('img generated')
        # st.write(np.unique(image, return_counts=True))

        return image




    def __len__(self):

        return self.main_image.shape[0]

  

    def pre_selected_coords(self,targeted_minerals_idx,n_targets_per_mineral,
                            mineral_dict = {0 : 'Ab',1 : 'Qz',2 : 'Mc',3 : 'Spd',4 : 'Brl',5 : 'Cb',6 : 'background'},
                            ):


        # Start with empty coords
        selected_coords=[]
        # Loops through selected minerals
        for mineral_id in targeted_minerals_idx: 
            # import streamlit as st
            # st.write(mineral_id, targeted_minerals_idx)

            # Get pixel coord where the mineral is present in the image (assumes we have an image class map)
            min_idx = np.where(self.main_image==mineral_id)
            

            ## Call function that activly filters coords within the image
            filtered_coords = self.filter_coords(min_idx)
            # get x,y positions
            mineral_coords = (filtered_coords[0],filtered_coords[1])
            
            # st.write(filtered_coords)


            #Now we have the coordinates of each mineral in the image
            # We will randomly select #n target spots per mineral
            assert len(mineral_coords[0]) >= n_targets_per_mineral, f"Not enough non-overlapping occurrences of mineral: {mineral_dict[mineral_id]}, to sample {n_targets_per_mineral} targets out of {n_targets_per_mineral} occurrences. Consider reducing n_targets_per_mineral, reduncing void proportion or increasing the image size."

            random_target_idx = np.random.choice(len(mineral_coords[0]), size=n_targets_per_mineral, replace=False)

            # organize
            random_target_coords_per_mineral = [(mineral_coords[0][i],mineral_coords[1][i]) for i in random_target_idx]
            selected_coords.append(random_target_coords_per_mineral)
        selected_coords = [np.array(item, dtype=int) for sublist in selected_coords for item in sublist]

        return selected_coords
    
    def filter_coords(self,min_idx):
        """
        Filters coordinates to ensure they are within bounds for a given spot size.

        Args:
            min_idx: A tuple containing arrays of row and column indices.
            image_size: A tuple containing the height and width of the image.
            spot_size: The size of the spot to be placed at the coordinates.

        Returns:
            A tuple containing filtered arrays of row and column indices.
        """
        rows, cols = min_idx
        valid_mask = (rows + self.spot_size < self.image_size) & (cols + self.spot_size < self.image_size)
        return rows[valid_mask], cols[valid_mask]


    def sampling_mask(self,seed_set,n_spots,coords=[]):
        """
        This creates the mask which represents the subsampling.
        seed_set: an array of controlled randomly generated seeds to always obtain the same random mask when a given main_seed is provided
        n_spots: number of spots in subsampling
        coords: list of pre-selected coordinates to sample in the case of targeted susampling
        """

        mask = np.zeros((self.image_size, self.image_size))
        counter=len(coords)

        s=0
        # These assertions guarantee that we can actually subsamlpe the given number of spots
        assert self.image_size >= self.spot_size, "spot_size must be smaller than image_size"
        assert n_spots*(self.spot_size**2) <= (self.image_size - self.spot_size)**2, "n_spots must be smaller than the number of possible spots contained by image size: increase image size or decrease n_spots"
        
        #Lets include in the mask the pre-select coordinates | if no pre-selected coords, it will be an empty list
        for coord in coords:
            mask[coord[0]:coord[0]+self.spot_size, coord[1]:coord[1]+self.spot_size] = 1

        # This loop randomly collects non-overlapping mask positions until the number of spots is achieved  
        while counter < n_spots:
            # Here the seed_set assures that we get the same masks everytime the simulation is performed.
            # To obtain other set of subsamples, one need to change the main_seed
            np.random.seed(seed_set[s])
            #Randomly try a coord
            coord = np.random.randint(0, self.image_size - self.spot_size, size=2)
            # Check if any pixel within these coords was already taken
            if  mask[coord[0]:coord[0]+self.spot_size, coord[1]:coord[1]+self.spot_size].sum() == 0:
                # If not (sum=0), select these pixels as subsample
                mask[coord[0]:coord[0]+self.spot_size, coord[1]:coord[1]+self.spot_size] = 1
                coords.append(coord)
                counter += 1
            s+=1
        return mask, coords

    def get_mean_spot_composition(self,coords,phases_composition):
        stacked_spots = np.zeros((1,self.spot_size,self.spot_size))
        stacked_flat_spots = np.zeros((1,self.spot_size**2))
        # counts_per_spot = pd.DataFrame(columns=['mineral','counts'])

        phases_composition['samples'] = phases_composition.index
        mean_spot_composition = pd.DataFrame(columns=self.elements_analyzed)
        spot_std = pd.DataFrame(columns=self.elements_analyzed)

        for spot,coord in enumerate(coords):
            sampled_spot = self.main_image[coord[0]:coord[0]+self.spot_size, coord[1]:coord[1]+self.spot_size]
            stacked_spots = np.vstack((stacked_spots,np.expand_dims(sampled_spot,axis=0)))

            flattened_array = sampled_spot.reshape(-1)
            stacked_flat_spots = np.vstack((stacked_flat_spots,flattened_array))

            df_array = pd.DataFrame({'samples': flattened_array})
            df_merged = pd.merge(df_array, phases_composition, on='samples', how='left')
            df_merged = df_merged.drop(columns=['samples','phase_proportion','mineral'])
            # After testing, lets try to stick to np arrays
            mean_spot_composition.loc[spot] = df_merged.mean()


        return mean_spot_composition,phases_composition, stacked_flat_spots[1:]
    



class ImagePlotter(ImageSampler):
    def __init__(self, image_sampler, mask, coords,masked_imge, main_image_composition,variaton_to_plot=0.1,alpha=0.85):
        self.image_sampler = image_sampler
        self.mask = mask
        self.coords = coords
        self.masked_imge = masked_imge
        self.main_image_composition = main_image_composition

        self.variaton_to_plot = variaton_to_plot
        self.alpha = alpha
        self.n_classes = self.image_sampler.n_classes - 1
        self.minerals = self.image_sampler.minerals[:-1]
        self.map_colors = self.get_colors()

    def get_colors(self):
        return {
        'Ab' : 'blue',

        'Qz': 'orange',
        
        'Mc' : 'green',

        'Spd' : 'violet',

        'Brl' : 'brown',

        'Cb': 'cyan',

        'background' : 'grey' ,

        }

    
    # @st.fragment
    def plot_masked_image(self,phases,web=True, clipview=True):
        import matplotlib.colors as mcolors
        fig1, ax = plt.subplots(figsize=(8, 8))
        # cmap = plt.cm.get_cmap('Set2', self.n_classes+1) 
        # sorted_classes = sorted()
        color_list = [ self.map_colors[k] for k in self.map_colors.keys()]
        cmap = mcolors.ListedColormap(color_list)



        ax.imshow(self.image_sampler.main_image, cmap=cmap)
        ax.imshow(self.mask, cmap='bone', alpha=0.4)
        handles = [plt.Rectangle((0, 0), 1, 1, color=cmap(i)) for i in range(self.n_classes+1)]
        ax.legend(handles, phases)


        fig2, ax = plt.subplots(figsize=(8, 8))
        # cmap = plt.cm.get_cmap('Set2', self.n_classes+1) 
        ax.imshow(self.masked_imge, cmap=cmap)
        # handles = [plt.Rectangle((0, 0), 1, 1, color=cmap(i)) for i in range(self.n_classes+1)]

        phases = phases + ['background']
        ax.legend(handles, phases)

        if web:
            return fig1,fig2
        else:
            plt.show()
            return None,None
        


    def mineral_count_per_simulation(self,mineral_count,mineral_proportion,phases_proportion, columns=3):
        lines = self.n_classes //columns if self.n_classes % columns == 0 else self.n_classes //columns + 1
        fig, axs = plt.subplots(nrows=lines,ncols=columns, figsize=(columns*5,lines*4))
        plt.suptitle(f'Mineral count per simulation and the true mean', fontsize=16)   

        for ax,phase in zip(axs.flat,self.image_sampler.minerals):
            array_to_plot_count = mineral_count.loc[:,phase]
            array_to_plot_proportion = mineral_proportion.loc[:,phase]


            
            # mineral_count.loc[:,phase]
            ax.scatter(np.arange(array_to_plot_proportion.shape[0]),array_to_plot_proportion,
                            color='k', marker='o', alpha=self.alpha)
            
            # ax2 = ax.twinx()

            
            # ax2.scatter(np.arange(array_to_plot_count.shape[0]), array_to_plot_count,
                        # color='b', marker='+', alpha=self.alpha, s=100)
            # ax2.set_ylabel('Absolute count', color='b')

            ax.set_title(phase)

                
            step=5 if self.image_sampler.n_simulations<20 else self.image_sampler.n_simulations//5
            ax.set_xticks(range(0,self.image_sampler.n_simulations,step))
            ax.set_xticklabels(range(0,self.image_sampler.n_simulations,step))
            elem_bulk_mean = phases_proportion[phase].values[0]
            ax.axhline(elem_bulk_mean, color='r', linestyle='dashed', label='Bulk composition',alpha=self.alpha)
            ax.axhline(elem_bulk_mean+elem_bulk_mean*self.variaton_to_plot, color='r', linestyle='dashdot',alpha=self.alpha-0.2)
            ax.axhline(elem_bulk_mean-elem_bulk_mean*self.variaton_to_plot, color='r', linestyle='dashdot',alpha=self.alpha-0.2)

        # Remove empty subplots
        for i in range(self.image_sampler.n_classes, lines * columns):
            row = i // columns
            col = i % columns
            fig.delaxes(axs[row, col])

        # ax2.set_ylabel('Absolute count', color='b')
        fig.supylabel('Mineral proportion')
        fig.supxlabel(f'Independent simulations')
        return fig
    

    def mean_element_composition_per_simulation(self,sampled_per_spot_image_composition,columns=4):
        # lines = self.image_sampler.n_classes //columns if self.image_sampler.n_classes % columns == 0 else self.image_sampler.n_classes //columns + 1

        lines = len(self.image_sampler.elements_analyzed)//columns if len(self.image_sampler.elements_analyzed) % columns == 0 else len(self.image_sampler.elements_analyzed) //columns + 1

        fig, axs = plt.subplots(nrows=lines,ncols=columns, figsize=(columns*5,lines*4)) #width,height #, dpi=150)
        # plt.suptitle(f'Oxide mean per simulation and the true mean', fontsize=14)   

        for ax,elem in zip(axs.flat,sampled_per_spot_image_composition.columns):    

            array_to_plot = sampled_per_spot_image_composition.loc[:,elem]
            
            ax.scatter(np.arange(array_to_plot.shape[0]),array_to_plot,
                            color='g', marker='^', alpha=self.alpha)
            ax.set_title(elem)

                
            step=5 if self.image_sampler.n_simulations<20 else self.image_sampler.n_simulations//5
            ax.set_xticks(range(0,self.image_sampler.n_simulations,step))
            ax.set_xticklabels(range(0,self.image_sampler.n_simulations,step))
            elem_bulk_mean = self.main_image_composition.loc[elem]#.values[0]
            ax.axhline(elem_bulk_mean, color='r', linestyle='dashed', label='Bulk composition',alpha=self.alpha)
            ax.axhline(elem_bulk_mean+elem_bulk_mean*self.variaton_to_plot, color='r', linestyle='dashdot',alpha=self.alpha-0.2)
            ax.axhline(elem_bulk_mean-elem_bulk_mean*self.variaton_to_plot, color='r', linestyle='dashdot',alpha=self.alpha-0.2)
            #make y-axis log
            ax.set_yscale('log')



        # Remove empty subplots
        for i in range(len(self.image_sampler.elements_analyzed), lines * columns):
            row = i // columns
            col = i % columns
            fig.delaxes(axs[row, col])


        fig.supylabel('Oxide concentration')
        fig.supxlabel(f'Independent simulations')
        return fig
    


