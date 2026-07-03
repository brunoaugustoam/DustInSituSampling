

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import streamlit as st
import matplotlib

from utils import *

# from ImageSampler import  ImagePlotter

def plot_phase_distribuition(phases_proportion, sampled_subsets, n_subsets, n_spots, variaton_to_plot,phases_per_column = 3):



    fig, ax = plt.subplots(2,phases_per_column, figsize=(15, 8))
    plt.suptitle(f'Phase distribution in the sample and in the sampled subsets with {n_spots} spots', fontsize=16)   
    for f in range(phases_per_column):
        phase = f'phase{f+1}'
        simulated_phase_distribuition = [sampled_subsets[i][phase] for i in range(n_subsets)]
        ax[0,f].set_title(f'{phase}')
        ax[0,f].scatter([i for i in range(n_subsets)], simulated_phase_distribuition,
                    color='k',marker='o', linewidth=0.5, alpha=0.9)
        # AQUI TEM QUE PASSAR O STD PRA CADA PONTO E NAO PRA TUDO - SINCERAMENTE NAO SEI SE FAZ SENTIDO ESSE STD
        # for i in range(n_subsets):
            # ax[0,f].errorbar(i, simulated_phase_distribuition[i], yerr=np.std(simulated_phase_distribuition[i]), fmt='none', ecolor='k',capsize=3, alpha=0.7)
    
        ax[0,f].hlines(phases_proportion[phase], 0, n_subsets, colors='r', linestyles='dashed', alpha=0.9)
        # ax[0,f].hlines(mean_distribuition[phase], 0, n_subsets, colors='g', linestyles='dashdot', alpha=0.7) #mean_distribuition std_distribuition
        ax[0,f].hlines(phases_proportion[phase]+phases_proportion[phase]*variaton_to_plot, 0, n_subsets, colors='r', linestyles='dotted', alpha=0.7)
        ax[0,f].hlines(phases_proportion[phase]-phases_proportion[phase]*variaton_to_plot, 0, n_subsets, colors='r', linestyles='dotted', alpha=0.7)

        phase = f'phase{f+4}'
        simulated_phase_distribuition = [sampled_subsets[i][phase] for i in range(n_subsets)]
        ax[1,f].set_title(f'{phase}')
        ax[1,f].scatter([i for i in range(n_subsets)], simulated_phase_distribuition,
                    color='k',marker='o', linewidth=0.5, alpha=0.7)

        ax[1,f].hlines(phases_proportion[phase], 0, n_subsets, colors='r', linestyles='dashed', alpha=0.7, label='Real mean')
        # ax[1,f].hlines(mean_distribuition[phase], 0, n_subsets, colors='g', linestyles='dashdot', alpha=0.7, label='Sampled mean') #mean_distribuition std_distribuition
        ax[1,f].hlines(phases_proportion[phase]+phases_proportion[phase]*variaton_to_plot, 0, n_subsets, colors='r', linestyles='dotted', alpha=0.7)
        ax[1,f].hlines(phases_proportion[phase]-phases_proportion[phase]*variaton_to_plot, 0, n_subsets, colors='r', linestyles='dotted', alpha=0.7)
    plt.legend(loc='upper right')
    fig.supylabel('Phase proportion (%)')
    fig.supxlabel(f'Independent simulations with {n_spots} spots')
    return fig


def plot_element_distribuition(elements_analyzed, element_per_simulation, sample_mean_composition, variaton_to_plot, n_subsets, n_spots):

    fig, ax = plt.subplots(1,len(elements_analyzed), figsize=(15, 8))
    plt.suptitle(f'Oxide distribution in the sample and in the sampled subsets with {n_spots} spots', fontsize=16)   
    # for f in range(phases_per_column):
    for (k,element) in enumerate(elements_analyzed):

        ax[k].set_title(f'{element}')
        ax[k].scatter([i for i in range(n_subsets)], [element_per_simulation[j][element] for j in range(n_subsets)] ,
                    color='g',marker='^', alpha=0.7)#, label='Simulated composition')

        

        ax[k].hlines(sample_mean_composition[element], 0, n_subsets, colors='r', linestyles='dashed', alpha=0.9, label='Real mean')
        # ax[k].hlines(element_distribuition_mean[element], 0, n_subsets, colors='g', linestyles='dashdot', alpha=0.9, label='Sampled mean')
        ax[k].hlines(sample_mean_composition[element]+sample_mean_composition[element]*variaton_to_plot, 0,
                    n_subsets, colors='r', linestyles='dotted', alpha=0.7)
        ax[k].hlines(sample_mean_composition[element]-sample_mean_composition[element]*variaton_to_plot, 0, 
                    n_subsets, colors='r', linestyles='dotted', alpha=0.7)

    plt.legend(loc='upper right')
    fig.supylabel('Oxide concentration')
    fig.supxlabel(f'Independent simulations with {n_spots} spots')
    return fig


def phase_distribuition_n_spots(phases_proportion, sampled_subsets_varying_n, n_subsets_varying, varying_n_spots, phases_per_column = 3,variaton_to_plot=0.1):
    fig_varying_spots, ax = plt.subplots(2,phases_per_column, figsize=(15, 8))
    plt.suptitle(f'Phase distribution variation given the number of spots analyzed in the sample, analyzing {n_subsets_varying} subsets', fontsize=16) 
    l, c = 0, 0
    # for f in range(phases_per_column):
    for phase in sorted(list(phases_proportion.keys())):
        for n_spots in varying_n_spots:
            
            sim_phase_distribuition = [sampled_subsets_varying_n[n_spots][i][phase] for i in range(n_subsets_varying)]
            ax[l,c].set_title(f'{phase}')
            ax[l,c].scatter([n_spots]*len(sim_phase_distribuition), sim_phase_distribuition, label=f'Proportion sampled',
                    color='k',marker='o', linewidth=0.5, alpha=0.7)
            ax[l,c].hlines(phases_proportion[phase], 0, max(varying_n_spots), colors='r', linestyles='solid', label='Real mean', alpha=0.7)
            ax[l,c].hlines(phases_proportion[phase]+phases_proportion[phase]*variaton_to_plot, 0, max(varying_n_spots),
                            colors='r', linestyles='dashdot', alpha=0.7, label=f'{variaton_to_plot*100}% Variation')
            ax[l,c].hlines(phases_proportion[phase]-phases_proportion[phase]*variaton_to_plot, 0, max(varying_n_spots),
                            colors='r', linestyles='dashdot', alpha=0.7)
            ax[l,c].set_xscale('log')
            ax[l,c].set_xticks(varying_n_spots)
            ax[l,c].get_xaxis().set_major_formatter(plt.ScalarFormatter())
        if c == phases_per_column-1:
            c=0
            l+=1
        else:
            c+=1
        


    fig_varying_spots.supylabel('Phase proportion (%)')
    fig_varying_spots.supxlabel(f'Number of spots analyzed (with {n_subsets_varying} subsets)')
        # Adjust layout for better spacing
        # plt.tight_layout()
    return fig_varying_spots


def phase_composition_distribuition_n_spots(element_per_simulation_varying_n, varying_n_spots,n_subsets_varying,elements_analyzed,sample_mean_composition,variaton_to_plot=0.1):
    fig, ax = plt.subplots(1,len(elements_analyzed), figsize=(15, 8))
    plt.suptitle(f'Oxide distribution varying number of spots, analyzing {n_subsets_varying} subsets', fontsize=16)   
    for n_spots in varying_n_spots:
        
        for (k,element) in enumerate(elements_analyzed):
            sim_phase_distribuition = [element_per_simulation_varying_n[n_spots][i][element] for i in range(n_subsets_varying)]
            ax[k].set_title(f'{element}')
            ax[k].scatter([n_spots]*len(sim_phase_distribuition), sim_phase_distribuition,
                        color='g',marker='^', alpha=0.7)

            ax[k].hlines(sample_mean_composition[element], 0, max(varying_n_spots), colors='r', linestyles='dashed', alpha=0.9)
            ax[k].hlines(sample_mean_composition[element]+sample_mean_composition[element]*variaton_to_plot, 0, max(varying_n_spots), 
            colors='r', linestyles='dotted', alpha=0.7)
            ax[k].hlines(sample_mean_composition[element]-sample_mean_composition[element]*variaton_to_plot, 0, max(varying_n_spots), 
                        colors='r', linestyles='dotted', alpha=0.7)

    fig.supylabel('Oxide concentration')
    fig.supxlabel(f'Number of spots analyzed (with {n_subsets_varying} subsets)')
    return fig


def plot_element_scatter(element_per_simulation, n_subsets,sample_mean_composition,elements_to_plot=['SiO2','Ni','Ti','Ce']):
    num_combinations = 6 #count_combinations(elements_to_plot)
    columns = 3
    lines = num_combinations//columns
    e1,e2 = 0,1
    fig, axs = plt.subplots(nrows=lines,ncols=columns, figsize=(15, 8))
    plt.suptitle(f'Oxide mean per simulation and the true mean', fontsize=16)   

    for ax in axs.flat:
        # for phase in sample_mean_composition.keys():
            # ax.scatter(phases_composition[phase][elements_to_plot[e1]], phases_composition[phase][elements_to_plot[e2]],
                            # color='r', marker='o', alpha=0.7, label='Real data')

        ax.scatter(sample_mean_composition[elements_to_plot[e1]], sample_mean_composition[elements_to_plot[e2]],
                            color='r', marker='o', alpha=1, label='Real data mean')
            
        for k in range(n_subsets):           
            ax.scatter(element_per_simulation[k][elements_to_plot[e1]], element_per_simulation[k][elements_to_plot[e2]],
                            color='b', marker='^', alpha=0.7)
            ax.set_xlabel(f'{elements_to_plot[e1]}')
            ax.set_ylabel(f'{elements_to_plot[e2]}')
        if e2 < len(elements_to_plot)-1:
            e2 = e2+1
        else:
            e1 = e1+1
            e2 = e1+1
        if e1 == len(elements_to_plot)-1:
            break
    fig.supylabel('Oxide concentration')
    plt.legend(loc='upper left')
    return fig



def matrix_plot(df,sample_mean=None):
    # Create a matrix of scatter plots
    num_cols = len(df.columns)
    fig, axs = plt.subplots(num_cols, num_cols, figsize=(25,25))

    # Later would be good to add a threshold keep the aspect ratio when the values get closer to the sample mean
    for i in range(num_cols):
        for j in range(num_cols):
            if i == j:
                axs[i, j].text(0.5, 0.5, df.columns[i], ha='center', va='center', transform=axs[i, j].transAxes)
            else:
                axs[i, j].scatter(df[df.columns[j]], df[df.columns[i]])
                axs[i, j].set_xlabel(df.columns[j])
                axs[i, j].set_ylabel(df.columns[i])
                if sample_mean is not None:
                    #here add not only the sample mean, but the element composition of each phase
                    axs[i, j].scatter(sample_mean[df.columns[j]], sample_mean[df.columns[i]], color='red', marker='+', s=100)

    return fig


def plot_elements_per_subset(simulated_subsets,n_subsets,sample_mean_composition,elements_analyzed,columns=5,variaton_to_plot=0.1):
    
    lines = len(elements_analyzed)//columns+1

    fig, axs = plt.subplots(nrows=lines,ncols=columns, figsize=(15, 15), dpi=150)
    plt.suptitle(f'Oxide mean per simulation and the true mean', fontsize=16)   
    alpha=0.85
    for k in range(n_subsets):    
        array_to_plot = simulated_subsets[k]['mean_subset_composition']
        for element,ax in zip(elements_analyzed,axs.flat):
            ax.scatter(k,array_to_plot[element],
                            color='g', marker='^', alpha=alpha)
            ax.set_title(element)
            if k ==0:
                
                step=5 if n_subsets<20 else n_subsets//5
                ax.set_xticks(range(0,n_subsets,step))
                ax.set_xticklabels(range(0,n_subsets,step))
                elem_bulk_mean = sample_mean_composition[element].values[0]
                ax.axhline(elem_bulk_mean, color='r', linestyle='dashed', label='Bulk composition',alpha=alpha)
                ax.axhline(elem_bulk_mean+elem_bulk_mean*variaton_to_plot, color='r', linestyle='dashdot',alpha=alpha-0.2)
                ax.axhline(elem_bulk_mean-elem_bulk_mean*variaton_to_plot, color='r', linestyle='dashdot',alpha=alpha-0.2)
            else:
                pass
            # Remove empty subplots
    for i in range(len(elements_analyzed), lines * columns):
        row = i // columns
        col = i % columns
        fig.delaxes(axs[row, col])

    # plt.legend(loc='upper right')
    fig.supylabel('Oxide concentration')
    fig.supxlabel(f'Independent simulations')
    return fig


def plot_phases_per_subset(mineral_count,n_subsets,phases_proportion,columns=3,variation_to_plot=0.1):
    
    lines = len(phases_proportion.columns)//columns

    fig, axs = plt.subplots(nrows=lines,ncols=columns, figsize=(15, 8), dpi=150)
    plt.suptitle(f'Mineral proportion per simulation and the true count', fontsize=16)   

    for k in range(n_subsets):    
        array_to_plot = mineral_count.loc[k]
        for mineral,ax in zip(phases_proportion.columns,axs.flat):
            ax.scatter(k,array_to_plot[mineral],
                            color='k', marker='o', alpha=0.7)
            ax.set_title(mineral)
            if k ==0:
                alpha=0.7
                step=5 if n_subsets<20 else n_subsets//5
                ax.set_xticks(range(0,n_subsets,step))
                ax.set_xticklabels(range(0,n_subsets,step))
                phase_prop = phases_proportion[mineral].values[0]
                
                ax.axhline(phase_prop, color='r', linestyle='dashed', label='Bulk composition',alpha=alpha)
                ax.axhline(phase_prop+phase_prop*variation_to_plot, color='r', linestyle='dashdot',alpha=alpha-0.2)

                ax.axhline(phase_prop-phase_prop*variation_to_plot, color='r', linestyle='dashdot',alpha=alpha-0.2)

            else:
                pass
    for i in range(len(phases_proportion.columns), lines * columns):
        row = i // columns
        col = i % columns
        fig.delaxes(axs[row, col])

    # plt.legend(loc='upper right')
    fig.supylabel('Mineral sampled proportion')
    fig.supxlabel(f'Independent simulations')
    return fig



def plot_multiphase_composition_distribuition_n_spots(simulated_subsets_varying_nspots, varying_n_spots,n_subsets,elements_analyzed,
                                                sample_mean_composition,columns=5,variaton_to_plot=0.1):
    lines = len(elements_analyzed)//columns+1

    fig, axs = plt.subplots(nrows=lines,ncols=columns, figsize=(15, 15), dpi=150)

    plt.suptitle(f'Oxide distribution varying number of spots, analyzing {n_subsets} subsets', fontsize=16)   
    for n_spots in varying_n_spots:
        
        for k in range(n_subsets):    
            array_to_plot = simulated_subsets_varying_nspots[n_spots][k]['mean_subset_composition']
            for element,ax in zip(elements_analyzed,axs.flat):  
                
                ax.scatter(n_spots,array_to_plot[element],
                            color='g', marker='^', alpha=0.7)
                ax.set_title(f'{element}')
                #insert the better xticks
                ax.hlines(sample_mean_composition[element], 0, max(varying_n_spots), colors='r', linestyles='dashed', alpha=0.9)
                ax.hlines(sample_mean_composition[element]+sample_mean_composition[element]*variaton_to_plot, 0, max(varying_n_spots), 
                colors='r', linestyles='dotted', alpha=0.7)
                ax.hlines(sample_mean_composition[element]-sample_mean_composition[element]*variaton_to_plot, 0, max(varying_n_spots), 
                            colors='r', linestyles='dotted', alpha=0.7)
                

                step=10 if max(varying_n_spots)<100 else max(varying_n_spots)//5
                ax.set_xticks(range(0,max(varying_n_spots)+1,step))
                ax.set_xticklabels(range(0,max(varying_n_spots)+1,step))

    for i in range(len(elements_analyzed), lines * columns):
        row = i // columns
        col = i % columns
        fig.delaxes(axs[row, col])
    fig.supylabel('Oxide concentration')
    fig.supxlabel(f'Number of spots analyzed (with {n_subsets} subsets)')
    return fig




def plot_multiphase_mineral_count_n_spots(mineral_count_all_spots, varying_n_spots,phases_proportion,
                                        n_subsets=10,columns=3,variaton_to_plot=0.1):

    lines = len(phases_proportion.columns)//columns

    fig, axs = plt.subplots(nrows=lines,ncols=columns, figsize=(15, 12), dpi=150)

    plt.suptitle(f'Mineral proportion per simulation and the true count', fontsize=16)   

    for n_spots in varying_n_spots:
        
        for k in range(n_subsets):    
            array_to_plot = mineral_count_all_spots[n_spots].loc[k]
            for mineral,ax in zip(phases_proportion.columns,axs.flat):
                
                ax.scatter(n_spots,array_to_plot[mineral],
                            color='k', marker='o', alpha=0.7)
                ax.set_title(f'{mineral}')

                if k ==0:
                    alpha=0.7
                    phase_prop = phases_proportion[mineral].values[0]
                    
                    ax.axhline(phase_prop, color='r', linestyle='dashed', label='Bulk composition',alpha=alpha)
                    ax.axhline(phase_prop+phase_prop*variaton_to_plot, color='r', linestyle='dashed', label='Bulk composition',alpha=alpha)

                    ax.axhline(phase_prop-phase_prop*variaton_to_plot, color='r', linestyle='dashed', label='Bulk composition',alpha=alpha)

                step=10 if max(varying_n_spots)<100 else max(varying_n_spots)//5
                ax.set_xticks(range(0,max(varying_n_spots)+1,step))
                ax.set_xticklabels(range(0,max(varying_n_spots)+1,step))
    fig.supylabel('Mineral sampled concentration')
    fig.supxlabel(f'Number of spots analyzed')
    return fig


def plot_element_scatter_matrix(mineral_count,phases_proportion,n_simulations,variaton_to_plot=0.1,web=True):

    columns = 3
    lines = len(phases_proportion.columns)//columns if len(phases_proportion.columns)%columns==0 else len(phases_proportion.columns)//columns+1

    fig, axs = plt.subplots(nrows=lines,ncols=columns, figsize=(15, 15), dpi=150)
    plt.suptitle(f'Mineral count per simulation and the true mean', fontsize=16)   
    alpha=0.85
    for ax,phase in zip(axs.flat,phases_proportion.columns):
        array_to_plot = mineral_count.loc[:,phase]
        ax.scatter(np.arange(array_to_plot.shape[0]),array_to_plot,
                        color='k', marker='o', alpha=alpha)
        ax.set_title(phase)

            
        step=5 if n_simulations<20 else n_simulations//5
        ax.set_xticks(range(0,n_simulations,step))
        ax.set_xticklabels(range(0,n_simulations,step))
        elem_bulk_mean = phases_proportion[phase].values[0]
        ax.axhline(elem_bulk_mean, color='r', linestyle='dashed', label='Bulk composition',alpha=alpha)
        ax.axhline(elem_bulk_mean+elem_bulk_mean*variaton_to_plot, color='r', linestyle='dashdot',alpha=alpha-0.2)
        ax.axhline(elem_bulk_mean-elem_bulk_mean*variaton_to_plot, color='r', linestyle='dashdot',alpha=alpha-0.2)

        # Remove empty subplots
    for i in range(len(phases_proportion.columns), lines * columns):
        row = i // columns
        col = i % columns
        fig.delaxes(axs[row, col])

    # plt.legend(loc='upper right')
    fig.supylabel('Mineral count')
    fig.supxlabel(f'Independent simulations')
    if web:
        return fig
    else:
        plt.show()
        return None
    


def plot_element_composition_per_simulation(sampled_per_spot_image_composition, every_image_composition,n_simulations,simulation_to_plot,elements_analyzed,variaton_to_plot):

    columns = 5
    lines = len(elements_analyzed)//columns+1

    fig, axs = plt.subplots(nrows=lines,ncols=columns, figsize=(15, 15), dpi=150)
    plt.suptitle(f'Oxide mean per simulation and the true mean', fontsize=16)   
    alpha=0.85
    for ax,elem in zip(axs.flat,sampled_per_spot_image_composition.columns):    

        array_to_plot = sampled_per_spot_image_composition.loc[:,elem]
        
        ax.scatter(np.arange(array_to_plot.shape[0]),array_to_plot,
                        color='g', marker='^', alpha=alpha)
        ax.set_title(elem)

            
        step=5 if n_simulations<20 else n_simulations//5
        ax.set_xticks(range(0,n_simulations,step))
        ax.set_xticklabels(range(0,n_simulations,step))
        elem_bulk_mean = every_image_composition.loc[simulation_to_plot,elem]#.values[0]
        ax.axhline(elem_bulk_mean, color='r', linestyle='dashed', label='Bulk composition',alpha=alpha)
        ax.axhline(elem_bulk_mean+elem_bulk_mean*variaton_to_plot, color='r', linestyle='dashdot',alpha=alpha-0.2)
        ax.axhline(elem_bulk_mean-elem_bulk_mean*variaton_to_plot, color='r', linestyle='dashdot',alpha=alpha-0.2)

        # Remove empty subplots
    for i in range(len(elements_analyzed), lines * columns):
        row = i // columns
        col = i % columns
        fig.delaxes(axs[row, col])

    # plt.legend(loc='upper right')
    fig.supylabel('Oxide concentration')
    fig.supxlabel(f'Independent simulations')



def mineral_count_multiple_simulation(data,phases_proportion, n_classes, minerals,n_simulations, variaton_to_plot=0.1, columns=3):
    lines = n_classes //columns if n_classes % columns == 0 else n_classes //columns + 1
    fig, axs = plt.subplots(nrows=lines,ncols=columns, figsize=(columns*5,lines*4), dpi=150)
    plt.suptitle(f'Mineral count per simulation and the true mean', fontsize=16)   
    alpha=0.85
    symbols = ['o','^','s','p','*','+','x','D','h','v','<','>','d']
    colors = ['b','g','r','c','m','y','k','w']
    for idx, n_targets_data in enumerate(data):
        import streamlit as st
        mineral_proportion = data[n_targets_data]
        st.dataframe(mineral_proportion)
        for ax,phase in zip(axs.flat,minerals):
            # array_to_plot_count = mineral_count.loc[:,phase]
            array_to_plot_proportion = mineral_proportion.loc[:,phase]


            
            # mineral_count.loc[:,phase]
            ax.scatter(np.arange(array_to_plot_proportion.shape[0]),array_to_plot_proportion,
                            color=colors[idx], marker=symbols[idx], alpha=alpha)
            
            # ax2 = ax.twinx()

            
            # ax2.scatter(np.arange(array_to_plot_count.shape[0]), array_to_plot_count,
            #             color='b', marker='+', alpha=alpha, s=50)
            # ax2.set_ylabel('Absolute count', color='b')

            ax.set_title(phase)

                
            step=5 if n_simulations<20 else n_simulations//5
            ax.set_xticks(range(0,n_simulations,step))
            ax.set_xticklabels(range(0,n_simulations,step))
            elem_bulk_mean = phases_proportion[phase].values[0]
            ax.axhline(elem_bulk_mean, color='r', linestyle='dashed', label='Bulk composition',alpha=alpha)
            ax.axhline(elem_bulk_mean+elem_bulk_mean*variaton_to_plot, color='r', linestyle='dashdot',alpha=alpha-0.2)
            ax.axhline(elem_bulk_mean-elem_bulk_mean*variaton_to_plot, color='r', linestyle='dashdot',alpha=alpha-0.2)

    # Remove empty subplots
    for i in range(n_classes, lines * columns):
        row = i // columns
        col = i % columns
        fig.delaxes(axs[row, col])

    # ax2.set_ylabel('Absolute count', color='b')
    fig.supylabel('Mineral proportion')
    fig.supxlabel(f'Independent simulations')
    return fig


def mean_element_composition_multiple_simulation(data,main_image_composition,
                                                 elements_analyzed,n_simulations,
                                                 variaton_to_plot=0.1,columns=5):


    lines = len(elements_analyzed)//columns if len(elements_analyzed) % columns == 0 else len(elements_analyzed) //columns + 1

    fig, axs = plt.subplots(nrows=lines,ncols=columns, figsize=(lines*4,columns*4), dpi=150)
    plt.suptitle(f'Oxide mean per simulation and the true mean', fontsize=16)   
    alpha = 0.85
    symbols = ['^','s','p','*','+','x','D','h','v','<','>','d']
    colors = ['g','r','c','m','y','k','w']


    for idx, (key,sampled_per_spot_image_composition) in enumerate(data.items()):


        sampled_per_spot_image_composition = sampled_per_spot_image_composition.drop(columns=['Unnamed: 0'])
        for ax,elem in zip(axs.flat,elements_analyzed):  

        

            array_to_plot = sampled_per_spot_image_composition.loc[:,elem]
           
            # st.dataframe(array_to_plot)
            ax.scatter(np.arange(array_to_plot.shape[0]),array_to_plot,
                            color=colors[idx], marker=symbols[idx], alpha=alpha)
            ax.set_title(elem)

                
            step=5 if n_simulations<20 else n_simulations//5
            ax.set_xticks(range(0,n_simulations,step))
            ax.set_xticklabels(range(0,n_simulations,step))
            elem_bulk_mean = main_image_composition.loc[0,elem] #.values[0]
            ax.axhline(elem_bulk_mean, color='r', linestyle='dashed', label='Bulk composition',alpha=alpha)
            ax.axhline(elem_bulk_mean+elem_bulk_mean*variaton_to_plot, color='r', linestyle='dashdot',alpha=alpha-0.2)
            ax.axhline(elem_bulk_mean-elem_bulk_mean*variaton_to_plot, color='r', linestyle='dashdot',alpha=alpha-0.2)

            # Remove empty subplots
        for i in range(len(elements_analyzed), lines * columns):
            row = i // columns
            col = i % columns
            fig.delaxes(axs[row, col])

        # plt.legend(loc='upper right')
        fig.supylabel('Oxide concentration')
        fig.supxlabel(f'Independent simulations')
        return fig
    

# from matplotlib.ticker import FixedLocator, ScalarFormatter
from matplotlib.ticker import ScalarFormatter
def plot_boxplots(data, phases_proportion, minerals,x_axis_variable,abundance=None,columns=3,x_label='# analyse spots',web=True):
    variaton_to_plot=0.1
    

 
    lines = len(minerals)//columns if len(minerals) % columns == 0 else len(minerals) //columns + 1
    # lines = 2
    fig, axs = plt.subplots(nrows=lines,ncols=columns, figsize=(columns*5,lines*8), dpi=200)
    alpha = 0.9
    fontsize = 24

    base_value = 10  # Reference value
    base_width = 3  # Reference width

    from utils import calculate_log_scale_widths
    widths = calculate_log_scale_widths(x_axis_variable, base_value, base_width)
    # abundance = {'pl':'High', 'ol':'Medium', 'gt':'Low', 'py':'Trace'}
    for ax,mineral in zip(axs.flat,minerals):  

        orig_prop =  phases_proportion[mineral].item()
        ax.axhline(orig_prop, color='red', linestyle='dashed',alpha=alpha, linewidth=4)
        top = orig_prop+orig_prop*variaton_to_plot
        ax.axhline(top, color='red', linestyle='dashdot',alpha=alpha-0.2, linewidth=2)
        bottom = orig_prop-orig_prop*variaton_to_plot
        ax.axhline(bottom, color='red', linestyle='dashdot',alpha=alpha-0.2, linewidth=2)


        ymin = bottom
        ymax = top
        box_data = []
        box_positions = []
        for i, (key, value) in enumerate(data.items()):
            # value.columns = new_columns
            array_to_plot = value.loc[:,mineral]

            box_data.append(array_to_plot)
            box_positions.append(key)
            # ax.scatter([key]*len(array_to_plot),array_to_plot,
            #             color='k', marker='o', alpha=alpha-0.15,s=75)
            ymin = min(ymin,min(array_to_plot))
            ymax = max(ymax,max(array_to_plot))
            # if key == 10 and mineral == 'Plagioclase':   
            #     # ax.set_ylabel('Mineral Proportion', fontsize=20)
            #     ax.set_ylabel('%', fontsize=fontsize)
            # instead of scatter, plot a boxplot

        ax.boxplot(box_data, positions=box_positions, patch_artist=True, boxprops=dict(facecolor='lightcyan', 
                                                                                    color='darkcyan',
                                                                                    alpha=0.75),
                    medianprops=dict(color='darkblue', linewidth=4), 
                    flierprops=dict(marker='+', markerfacecolor='darkcyan', markersize=12, linestyle='none'),
                    widths= widths ) #[2,10,30,70,120,300])

        if abundance is not None:
            ax.set_title(abundance[mineral], fontsize=fontsize)
        # ax.set_xlabel(f'# analyse spots', fontsize=fontsize)
        
            
        # step=10
        

        ax.set_xscale('log')
        ax.set_xticks(x_axis_variable)
        ax.minorticks_off()
        ax.xaxis.set_major_formatter(ScalarFormatter())

        ax.tick_params(axis='x', which='both', labelsize=fontsize, labelrotation=90)
        ymin = min(ymin,bottom*0.8)
        ymax = max(ymax,top*1.1)
        if ymax < 10:
            ax.set_yscale('symlog')
        
        y_ticks = np.linspace(ymin,ymax,6)
        
        ax.set_yticks(y_ticks)
        ax.set_yticklabels([f'{i:.2f}' for i in y_ticks],fontsize=fontsize)


        #set one single y axis label for every axis, like y sup labe;
    # fig.supylabel('Mineral Proportion (%)', fontsize=fontsize)
    fig.supxlabel(f"{x_label}", fontsize=fontsize)




    for i in range(len(minerals), lines * columns):
        row = i // columns
        col = i % columns
        fig.delaxes(axs[row, col])

    if web:
        return fig
    else:
        plt.show()






def plot_multiple_boxplots(data1,data2, phases_proportion, minerals,x_axis_variable,columns=3,abundance=None,x_label='# analyse spots',web=True):
    variaton_to_plot=0.1


 
    lines = len(minerals)//columns if len(minerals) % columns == 0 else len(minerals) //columns + 1
    # lines = 2
    fig, axs = plt.subplots(nrows=lines,ncols=columns, figsize=(columns*8,lines*9), dpi=150)
    alpha = 0.9
    fontsize = 20

    base_value = 15  # Reference value
    base_width = 4  # Reference width

    from utils import calculate_log_scale_widths
    widths = calculate_log_scale_widths(x_axis_variable, base_value, base_width)
    # abundance = {'pl':'High', 'ol':'Medium', 'gt':'Low', 'py':'Trace'}
    for ax,mineral in zip(axs.flat,minerals):  

        orig_prop =  phases_proportion[mineral].item()
        ax.axhline(orig_prop, color='red', linestyle='dashed',alpha=alpha, linewidth=4)
        top = orig_prop+orig_prop*variaton_to_plot
        ax.axhline(top, color='red', linestyle='dashdot',alpha=alpha-0.2, linewidth=2)
        bottom = orig_prop-orig_prop*variaton_to_plot
        ax.axhline(bottom, color='red', linestyle='dashdot',alpha=alpha-0.2, linewidth=2)


        ymin = bottom
        ymax = top
        box_data = []
        box_positions = []
        for i, (key, value) in enumerate(data1.items()):

            array_to_plot = value.loc[:,mineral]

            box_data.append(array_to_plot)
            box_positions.append(key+0.25)

            ymin = min(ymin,min(array_to_plot))
            ymax = max(ymax,max(array_to_plot))

        ax.boxplot(box_data, positions=box_positions, patch_artist=True, boxprops=dict(facecolor='chocolate', 
                                                                                    color='chocolate',
                                                                                    alpha=0.60),
                    medianprops=dict(color='darkblue', linewidth=4), 
                    flierprops=dict(marker='+', markerfacecolor='chocolate', markersize=12, linestyle='none'),
                    widths= widths ) 
        

        box_data2 = []
        box_positions2 = []
        for i, (key, value) in enumerate(data2.items()):

            array_to_plot = value.loc[:,mineral]

            box_data2.append(array_to_plot)
            box_positions2.append(key-0.25)

            ymin = min(ymin,min(array_to_plot))
            ymax = max(ymax,max(array_to_plot))
        

        
        ax.boxplot(box_data2, positions=box_positions2, patch_artist=True, boxprops=dict(facecolor='paleturquoise', 
                                                                                    color='paleturquoise',
                                                                                    alpha=0.60),
                    medianprops=dict(color='orchid', linewidth=4), 
                    flierprops=dict(marker='+', markerfacecolor='paleturquoise', markersize=12, linestyle='none'),
                    widths= widths) 

        if abundance is not None:
            ax.set_title(abundance[mineral], fontsize=fontsize)
        # ax.set_xlabel(f'# analyse spots', fontsize=fontsize)
        
            
        step=10
        ax.set_xscale('log')
        ax.set_xticks(x_axis_variable)
        ax.set_xticklabels(x_axis_variable,rotation=90,fontsize=fontsize)
        #Transform x axis to log scale
        

        ymin = min(ymin,bottom*0.9)
        ymax = max(ymax,top*1.1)
        y_ticks = np.linspace(ymin,ymax,6)
        ax.set_yticks(y_ticks)
        ax.set_yticklabels([f'{i:.2f}' for i in y_ticks],fontsize=fontsize)

        #set one single y axis label for every axis, like y sup labe;
    fig.supylabel('Mineral Proportion (%)', fontsize=fontsize)
    fig.supxlabel(f"{x_label}", fontsize=fontsize)




    for i in range(len(minerals), lines * columns):
        row = i // columns
        col = i % columns
        fig.delaxes(axs[row, col])

    if web:
        return fig
    else:
        plt.show()



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




def plot_pca(components_PCA, loadings_df, pcs=[0, 1],standards=[],minerals=[] , mineral_to_color=None, legend=False,void_n=None): #targeted_minerals=None,
    alpha = 0.6
    fontsize = 16

    # if targeted_minerals is not None:
    #     components_PCA['target'] = targeted_minerals

    # Define which loadings to plot (top 2 for each PC)
    # top_elements = get_top_two_abs_elements_list(loadings_df)
    from utils import get_top_three_abs_elements_list
    top_elements = get_top_three_abs_elements_list(loadings_df)

    arrow_plot = top_elements[f'PC{pcs[0]+1}'] + top_elements[f'PC{pcs[1]+1}']

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    # Add colors to dataframe to simplify plotting
    components = components_PCA.copy()
    # Compute colors for all points
    colors = [mineral_to_color.get(sample, 'gray') for sample in components['target']]


    components['color'] = colors

    # Split into random vs non-random
    df_random = components[components['target'] == 'random']
    df_st = components.query("target in @standards") #components[components['target'] == 'st']
    # print(df_st)


    df_not_random = components.query("target in @minerals") #components[components['target'] != 'random']
    # df_not_random = df_not_random[df_not_random['target'] != 'st']

    # Plot normal standards (x)
    ax.scatter(df_st.iloc[:, pcs[0]], df_st.iloc[:, pcs[1]],
                c=df_st['color'], marker='^', alpha=1, s=150) 
    
    # Plot normal minerals (circles)
    ax.scatter(df_not_random.iloc[:, pcs[0]], df_not_random.iloc[:, pcs[1]],
                c=df_not_random['color'], marker='o', alpha=alpha, s=75)



    # Plot random minerals (smaller diamonds)
    ax.scatter(df_random.iloc[:, pcs[0]], df_random.iloc[:, pcs[1]],
                c=df_random['color'], marker='D', alpha=alpha - 0.30, s=15)


    # Legend for minerals (non-random)
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', label=mineral,
                    markerfacecolor=color, markersize=10)
        for mineral, color in mineral_to_color.items() if (mineral != 'random' and mineral not in standards)
    ]

    # Add separate legend entry for random points
    legend_elements.append(
        plt.Line2D([0], [0], marker='D', color='w', label='random',
                    markerfacecolor='gray', markersize=6)
    )
    legend_elements.append(
        plt.Line2D([0], [0], marker='^', color='w', label='st',
                    markerfacecolor='magenta', markersize=10)
    )
    if legend:
        ax.legend(handles=legend_elements, title='Targeted on',
                    bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=fontsize - 4)



    # # Draw loading arrows
    for column in arrow_plot:
        ax.arrow(0, 0,
                    loadings_df.loc[f'PC{pcs[0]+1}', column],
                    loadings_df.loc[f'PC{pcs[1]+1}', column],
                    color='gray', alpha=alpha)
        ax.text(loadings_df.loc[f'PC{pcs[0]+1}', column] * 1.01,
                loadings_df.loc[f'PC{pcs[1]+1}', column] * 1.01,
                s=column, fontsize=fontsize - 6, color='k', alpha=alpha)

    xcol = f"PC{pcs[0]+1}"
    ycol = f"PC{pcs[1]+1}"

    for x, y, lab,color_st in zip(df_st[xcol], df_st[ycol], df_st["target"],df_st['color']):
        ax.text(x * 1.05, y * 1.01, lab, fontsize=fontsize-4,color=color_st)

    ax.set_title(f'Background proportion {void_n}%', fontsize=fontsize-8)		

    # # Draw loading arrow
    ax.set_xlabel(f'PC{pcs[0]+1}', fontsize=fontsize-2)
    ax.set_ylabel(f'PC{pcs[1]+1}', fontsize=fontsize-2)    	

    ax.set_xscale('symlog', linthresh=0.1)
    ax.set_yscale('symlog', linthresh=0.1)


    # ax.set_axis_off()
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    return fig




def reversed_matrix_plot_pca_loadings(components,loadings_df,num_cols=3, pca=None,
                                      mineral_to_color=None,targeted_minerals=None):
    if targeted_minerals is not None:
        components['target'] = targeted_minerals

    alpha= 0.7
    fig, axs = plt.subplots(num_cols, num_cols, figsize=(7*num_cols,7*num_cols))
    k=0
    alpha = 0.5
    fontsize = 22

    # Later would be good to add a threshold keep the aspect ratio when the values get closer to the sample mean
    for i in range(num_cols):
        for j in range(num_cols):
            if i == j:
                
                colormap = matplotlib.colormaps['terrain'] # 'viridis', 'plasma', 'inferno', 'magma', 'cividis'
                norm = plt.Normalize(loadings_df.loc[f'PC{i+1}'].min(), loadings_df.loc[f'PC{i+1}'].max())
                colors = colormap(norm(loadings_df.loc[f'PC{i+1}']))

                axs[i, j].bar(loadings_df.columns,loadings_df.loc[f'PC{i+1}'], color=colors)
                axs[i, j].grid( linestyle='--', linewidth=0.4,which='major', axis='both')
                axs[i, j].set_title(f'PC{i+1} loadings', fontsize=fontsize)
            elif i < j:
                colors = [mineral_to_color.get(sample, 'gray') for sample in components['target']] 

                scatter = axs[i, j].scatter(components[components.columns[j]], components[components.columns[i]],
                                             c=colors, marker='o', alpha=alpha, s=40)


                axs[i, j].set_xlabel(f'PC{j+1}', fontsize=fontsize)
                axs[i, j].set_ylabel(f'PC{i+1}', fontsize=fontsize)
                # Set equal aspect ratio for consistent scales
                # axs[i, j].set_aspect('equal') 
                axs[i, j].grid( linestyle='--', linewidth=0.4,which='major', axis='both')
                
            else:
                for column in range(len(loadings_df.columns)):
                    axs[i,j].arrow(0, 0, pca.components_[j, column], pca.components_[i, column], 
                        color='gray', label=loadings_df.columns[column], alpha=alpha)
                    axs[i,j].text(pca.components_[j, column]*1.01, pca.components_[i, column]*1.01, 
                            s=loadings_df.columns[column], fontsize=fontsize,color='k', alpha=alpha)
                    axs[i,j].set_xlabel(f'PC{j+1}', fontsize=fontsize)
                    axs[i,j].set_ylabel(f'PC{i+1}', fontsize=fontsize)
                    # axs[i, j].set_aspect('equal')
                
            
    
    return fig


