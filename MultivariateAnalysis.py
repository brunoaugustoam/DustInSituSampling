from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt





class PcaPlotter():
    def __init__(self, variaton_to_plot=0.1,alpha=0.5):
        self.variaton_to_plot = variaton_to_plot
        self.alpha = alpha



    @staticmethod
    def plot_pca_component_analysis(plot_data,max_components=10):
        
        # Create a scree plot
        
        pca = PCA(n_components=max_components).fit(plot_data)
        per_var = np.round(pca.explained_variance_ratio_ * 100 , 1)
        labels = ['PC' + str(x) for x in range(1,len(per_var) +1 )]
        fig, ax = plt.subplots(1,2, figsize=(15,6))
        ax[0].bar(x=range(1,len(per_var)+1 ), height = per_var, tick_label= labels )
        ax[0].set_ylabel(f'% explaine variance')
        ax[0].set_title(f"Explained variance: {np.sum(per_var):.0f}")

        y = np.cumsum(pca.explained_variance_ratio_)
        ax[1].plot(np.arange(1, max_components+1), y, marker='o', linestyle='-', color='black')

        ax[1].set_xlabel('Number of Components')
        ax[1].set_xticks(np.arange(1, max_components, step=1)) 
        ax[1].set_ylabel('Cumulative variance (%)')
        ax[1].set_title('The number of components needed to explain variance')
        ax[1].hlines(y=0.95, xmin=0,xmax=max_components, color='grey', linestyle='--')
        ax[1].text(0.1, 0.96, '95% cut-off threshold', color = 'black', fontsize=12)

        return fig
    
        
    def plot_PCA_loading_vectors(self, pca, loadings_df,n_components=3):
        # Plot the loadings
        fig, ax = plt.subplots(figsize=(15,6))
        columns = 3
        lines = n_components//columns if n_components%columns==0 else n_components//columns+1
        fig, axs = plt.subplots(nrows=lines,ncols=columns, figsize=(15,6))
        pc = 0
        pc_other = pc+1
        for j,ax in enumerate(axs.flat):
            # st.write(f"Principal component {pc} vs Principal component {pc_other}")


            for i in range(len(loadings_df.columns)):
                ax.arrow(0, 0, pca.components_[pc_other, i], pca.components_[pc, i], 
                        color='r', label=loadings_df.columns[i], alpha=self.alpha)
                ax.text(pca.components_[pc_other, i], pca.components_[pc, i], 
                        s=loadings_df.columns[i], fontsize=12,color='r', alpha=self.alpha)
                ax.set_xlabel(f'PC{pc_other+1}')
                ax.set_ylabel(f'PC{pc+1}')
            if pc_other == n_components-1:
                pc+=1
                pc_other=pc+1
            else:
                pc_other+=1

        return fig


    def matrix_plot_pca(self,components,bulk_to_color):
        ## LATER ON MAKE A DEFAULT OF COLOR AND MARKER WITH A JSON FILE
        import pandas as pd
        # Create a matrix of scatter plots
        num_cols = len(components.columns)-2
        # Create a dictionary to map bulk values to colors


        
        fig, axs = plt.subplots(num_cols, num_cols, figsize=(15,15))
        k=0
        # Later would be good to add a threshold keep the aspect ratio when the values get closer to the sample mean
        for i in range(num_cols):
            for j in range(num_cols):
                if i == j:
                    axs[i, j].text(0.5, 0.5, components.columns[i], ha='center', va='center', transform=axs[i, j].transAxes)
                    
                else:
                    components_standard = components[components['bulk'] == 'standard']
                    components_sampled = components[components['bulk'] != 'standard']
                    colors = [bulk_to_color.get(bulk, 'gray') for bulk in components_standard['bulk']] 
                    scatter = axs[i, j].scatter(components_standard[components_standard.columns[j]], components_standard[components_standard.columns[i]], 
                                c=colors, label=components_standard['bulk'], alpha=1,marker='D', s=250)
                    
                    colors = [bulk_to_color.get(bulk, 'gray') for bulk in components_sampled['bulk']] 
                    # size = [120 if bulk == 'sampled' else 50 for bulk in components['bulk']]
                    # alpha = [0.4 if bulk == 'sampled' else 1 for bulk in components['bulk']]
                    # mark = ['o' if bulk == 'sampled' else 's' for bulk in components['bulk']]

                    # axs[i, j].scatter(df[df.columns[j]], df[df.columns[i]], c=colors,label=df['bulk'])
                    scatter = axs[i, j].scatter(components_sampled[components_sampled.columns[j]], components_sampled[components_sampled.columns[i]], 
                                c=colors, label=components_sampled['bulk'], alpha=1,marker='o', s=75)

                    
                    # Add text annotations for minerals
                    for k, row in components.iterrows():
                        if not pd.isna(row['minerals']):
                            axs[i, j].text(row[components.columns[j]], row[components.columns[i]], 
                                           row['minerals'], fontsize=14, c='k', alpha=0.8,
                                           ha='center', va='top', rotation=45)

                    axs[i, j].set_xlabel(f'PC{components.columns[j]+1}')
                    axs[i, j].set_ylabel(f'PC{components.columns[i]+1}')
                    # Set equal aspect ratio for consistent scales
                    # axs[i, j].set_aspect('equal') 
                    axs[i, j].grid( linestyle='--', linewidth=0.5,which='major', axis='both')

                    if i==num_cols and j==0:
                        handles, labels = scatter.legend_elements()
                        axs[i, j].legend(handles, labels)

        return fig