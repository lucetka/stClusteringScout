### Thank you for studying my code. You will not learn anything new here.
### --Lucie

### This is my learning project. I am still experimenting here. It's my 2nd Streamlit app and my first deployed app.
### This is also the first time I am using GitHub to do more than just as a manual backup. 


#### irrlevant chatter with myself:
## starting from stOPTICRAP (which had started from stHAPPYCRAPPER, a descendant of stMULTICRAPPER, which in turn had started from simple CRAP)

### to do:

### - 1. split summary plot and recommendations
### - do something about the printed dict
### - tooltips and description
### - clusterplot
### - "icicle tree"
### - covariates
### - BT?
###

### *** i should alos put number_dimensions to session state but then I have to change it when using it later to st.session....

# if number_dimensions = "No dimensionality reduction" .... then what?
#######################
import streamlit as st

from st_aggrid import AgGrid
from st_aggrid.grid_options_builder import GridOptionsBuilder

import pandas as pd

import umap

import umap.plot


import numpy as np
import plotly.express as px
from hdbscan import HDBSCAN
import colorcet as cc

#from io import StringIO, BytesIO
#import io

import seaborn as sns
import matplotlib.pyplot as plt

# Set np random state
random_state = np.random.RandomState(42)              ###   CHANGE THIS IN THE FINAL VERSION, FOR NOW I'M LEAVING IT HERE BECAUSE I'M LAZY
np.random.RandomState(42)


def main():

    #***************************************
    #@st.cache(persist=True) 
    @st.cache_data    
    def load_data(uploaded_file, has_index_col):
        
        if has_index_col == 'Yes':
            data = pd.read_csv(uploaded_file, index_col=0)
            
        else:
            data = pd.read_csv(uploaded_file)

        if data.isnull().sum().sum() > 0:
            data = None
            st.write ("Your dataset has missing values and thus can't be used. Please fix it first (for example, you may want to fill the missing values with 0).")
            
        
        return data  #, covariates

    #@st.cache(persist=True)
    @st.cache_data  
    def load_covariates():
        #covariates = pd.read_csv('.\DATA_and_EMBEDDINGS\omics_2016-4Jul22_Qlik_ISI_only_Articles_Letters_Reviews_with_abstract_index_reference.csv')
        covariates = pd.read_csv(r'C:\Users\lkalvodova\OneDrive - Wiley\DESKTOP\WORK\DATA_SCIENCE\PYTHON\omics_2016-4Jul22_Qlik&ISI\omics_2016-4Jul22_Qlik_ISI_only_Articles_Letters_Reviews_with_abstract_index_reference.csv')
        return covariates


    def cluster_with_all_models_incl_eps(data,nested_dict):   
        st.write("nested dict:", nested_dict)
        st.write(data)
        clustered = {}
        for i in nested_dict.keys():     # i - these are the mcs values
            st.write(i)
            clustered[i] = {} 
            for j in nested_dict[i].keys():   # j - the eps values
                st.write("eps value: ", j)
                st.write(nested_dict[i][j])
                st.write(data)
                #clustered[i][j] = HDBSCAN().fit_predict(data)    #this step is working
                clustered[i][j] = nested_dict[i][j].fit_predict(data)     #this step is failing but for the life of me I can't figure out why
                st.write("after fit_predict: ", data)
        return clustered

  


    #******************************************

    st.set_page_config(layout="wide")
                    
    st.title("Compare clusterings with multiple UMAP and HDBSCAN models")
    st.sidebar.title("CLUSTERING SCOUT")

    "This app will help you choose parameters for dimension reduction and clustering based on your requirements for granularity and clustering completeness (i.e. you can input a threshold for maximum and minimum number of clusters and a threshold for maximum % of unclustered points). These thresholds will not influence getting the clustering results - they only influence the recommendations presented. This current version is very basic and does not provide any guidance for the selection of cluster_selection_epsilon ranges (this is planned to be implemented next) nor any cluster diagnostics."

    st.markdown("First, please your load your data.")

    has_index_col = st.radio("Use the 1st column as index? (say 'Yes' if the 1st column is NOT a variable, i.e. if the 1st column is for e.g. a unique identifier such as the title of the article or if it is a row number)", ('Yes', 'No ( ALL columns will be considered features (variables)! ). If your 2D plot looks very odd, check this again.'), key="has_index_col")
    
    uploaded_file = st.file_uploader("Select your csv file with multidimensional dataset that you wish to cluster:")
    
    if uploaded_file is not None:
        data = load_data(uploaded_file, has_index_col)

   
  
    st.sidebar.subheader("Choose Dimensionality Reduction Method and Hyperparameters")
    st.sidebar.write("(Currently only UMAP)")
    
    st.sidebar.markdown("UMAP")
    
    dim_red_options=['No dimensionality reduction' if i == 51 else i for i in range(2,51)]    # in principle dim=1 is possible with some methods too but it's only useful in very specific cases so I'll leave it out here
   # I am using 50 as a hardcoded max becuase HDBSCAN doesn't work for hi-dim datasets

    number_dimensions = st.sidebar.select_slider("Choose number of dimensions for clustering" , dim_red_options, value = 5, key='number_dimensions')
    #dimred_method = st.sidebar.selectbox("Method", ("UMAP", "TSNE", "PCA"), key = "dimred_method")
    dimred_method = "UMAP"
    
    
    if dimred_method == 'UMAP':
        #st.sidebar.subheader("Model Hyperparameters")
        try:
            if len(data)>0:
                max_n_neighbors = round(len(data)/(1/0.8))    # not sure what is a reasonable max....80% of dataset is crazy high
            else:
                max_n_neighbors = 500
        except:
                max_n_neighbors = 500

        #n_neighbors = st.sidebar.slider('Select the range for n_neighbors values to test:', 3,  max_n_neighbors, (6, 36), key='n_neighbors')
        #st.sidebar.write("len = ", len(data))
        #st.sidebar.write("max_n_neighbors = ", max_n_neighbors)
        try:
            st.sidebar.write("Dataset length  = ", len(data))
            st.sidebar.write("As max n_neighbors we will allow ", max_n_neighbors)
        except:
            st.sidebar.write("No dataset loaded!")

        n_neighbors_help = 'If you are interested in the finer structure of your dataset (i.e., local structure), you will want to choose relatively small values of n_neighbors, however typically I would not recommend using values <5 (if using small values of n_neighbors, I would strongly recommend looking at connectivty and other diagnostic plots). Typically you will want the value of n_neighbors to be "similar" -- definitely at least on the same order -- as your cluster size.'    
        n_neighbors = [10,30]
        n_neighbors[0] = st.sidebar.number_input("Minimal n_neighbors to try:", help=n_neighbors_help, min_value = 1, max_value = max_n_neighbors, step = 1, value = 10, key='n_neighbors_0')
        n_neighbors[1] = st.sidebar.number_input("Maximum n_neighbors to try:", help=n_neighbors_help, min_value = 1, max_value=max_n_neighbors, step = 1, value = 30, key='n_neighbors_1')
        n_neighbors = tuple(n_neighbors)

        nn_step = st.sidebar.number_input("Select the step for n_neighbors: ", min_value = 1, max_value = max_n_neighbors, step = 1, value = n_neighbors[0], key='nn_step' )

        umap_metric = st.sidebar.selectbox("Metric (single value only)", ("euclidean", "manhattan", "chebyshev", "minkowski", "canberra", "braycurtis",
            "haversine", "mahalanobis", "wminkowski", "seuclidean", "cosine", "correlation", "hamming", "jaccard", "dice",
             "russellrao", "kulsinski", "rogerstanimoto", "sokalmichener", "sokalsneath", "yule"), index = 10, key = "umap_metric")
        min_dist = st.sidebar.number_input("min_dist (single value only)", min_value = 0.0, max_value = None,step = 0.01, key = "min_dist")
        random_state = st.sidebar.number_input("random_state", min_value = 0, max_value = None, step = 1, value = 42, key = "random_state")
       


       
        st.sidebar.subheader("HDBSCAN hyperparameters")

        min_samples = st.sidebar.number_input("min_samples  (currently single value only)",1, max_n_neighbors, value = 3, key='min_samples')

        
        #min_cluster_size = st.sidebar.slider('Select the range of min_cluster_size values to test:', 5,  max_n_neighbors, (8, 48), key='min_cluster_size')

        min_cluster_size = [3,3]
        min_cluster_size[0] = st.sidebar.number_input("Smallest minimal cluster size to try:", min_value = 3, max_value = max_n_neighbors, step = 1, value = 5, key='min_cluster_size_0')    #len(data) setting max_value to max_n_neighbors is probably a bit illogical, this is jsut before I figure out how to best do the file loading
        min_cluster_size[1] = st.sidebar.number_input("Largest minimal cluster size to try:", min_value = 3, max_value = max_n_neighbors, step = 1,  value = max_n_neighbors,  key='min_cluster_size_1')   #len(data)
        min_cluster_size = tuple(min_cluster_size)


        mcs_step = st.sidebar.number_input("Select the step for min_cluster_size: ", min_value = 1, max_value = min_cluster_size[1], step = 1, value = min_cluster_size[0], key='mcs_step' )


        hdbscan_metric =  st.sidebar.selectbox("Metric (currently single value only)", ("euclidean", "manhattan"), index = 0, key = "hdbscan_metric")

        cluster_selection_method = st.sidebar.radio('Cluster selection method (currently single choice only):', ('eom', 'leaf'), key = 'cluster_selection_method')
        

        cluster_selection_epsilon = st.sidebar.number_input('Maximum cluster selection epsilon:', min_value = 0.000, max_value = None, value =0.030 , key = 'cluster_selection_epsilon' )   #, format="%f"
        eps_step = st.sidebar.number_input("Select the step for epsilon values: ", min_value =0.000, max_value = cluster_selection_epsilon, value = 0.000, key='eps_step' )     #, format="%f" 
 

        max_acceptable_n_clusters = st.sidebar.number_input('Maximum acceptable number of clusters in your clustering:', min_value = 1, value = 350, key ='max_n_clusters')
        min_acceptable_n_clusters = st.sidebar.number_input('Mminimum acceptable number of clusters in your clustering:', min_value = 1, value = 100, key ='min_n_clusters')

        # find minimum unclustered within eps 0 while keeping # of clusters within user-defined range
        plot_connectivity = st.sidebar.radio('Do you wish to plot the connectivity plot for diagnostic purposes? This can take VERY LONG, however, it will be the last thing calculated/plotted and you can already explore the results while the connectivity plot is being made.',('No','Yes'), key='plot_connectivity')

        umap_models = {}
        configurations = {}


        maxeps = cluster_selection_epsilon
        epsstep = eps_step

        if (st.sidebar.button("Create models with the selected hyperparameter ranges to compare")):   #if (st.sidebar.button("Create models with the selected hyperparameter ranges to compare")) & (epsstep <= maxeps):
            
            

            if ((maxeps>0) and (epsstep == 0)):    # if user enters a nonzero value for cluster_slection_epsilon but doesn't enter step, assume that he/she wants to use just this one value which means step = maxeps
                epsstep = maxeps
            
            ### determine the multiplier
          
                
            
            # if both maxeps and epsstep are integers, multiplier will be 1
            if ((maxeps == int(maxeps)) and (epsstep == int(epsstep))):
                multiplier = int(1)
            else:
                if (epsstep <= 1):
                    multiplier = int(np.power(10,(np.ceil(-np.log10(epsstep)))))
                else:
                    multiplier = int(np.power(10,(np.ceil(np.log10(maxeps)))))
          
            # caveat - now if I say for eg maxeps is 1.5 and step is 1, it will go for eps 0 and 1, ie step overrides maxeps. Maybe I should have asked the user to define how many steps...
            
            st.write('multiplier type: ', type(multiplier))
            st.write("multiplier = ", multiplier)

            n_neighbors_min = n_neighbors[0]
            n_neighbors_max = n_neighbors[1]
            n_neighbors_step = nn_step
            
                  
                 # I just want to print the models for the user to check:
            st.write('UMAP models:')
            umap_models = {nn:umap.UMAP(n_neighbors = nn, n_components = number_dimensions, min_dist=min_dist, metric = umap_metric, random_state=random_state) for nn in range(n_neighbors_min , n_neighbors_max+n_neighbors_step, n_neighbors_step )}
            
                #   let's plot for the extreme cases of nn

            if number_dimensions > 2:
                viz_models = {nn:umap.UMAP(n_neighbors = nn, n_components = 2, min_dist=min_dist, metric = umap_metric, random_state=random_state) for nn in [n_neighbors[0] , n_neighbors[1]] }
            else:
                viz_models = umap_models

            st.write(umap_models)

            min_cluster_size_min = min_cluster_size[0]
            min_cluster_size_max = min_cluster_size[1]
            cluster_step = mcs_step
          

            st.write('HDBSCAN models:')
            
            st.write("epsstep: ", epsstep)
         


            for i in range(min_cluster_size_min,min_cluster_size_max+cluster_step, cluster_step):        
                configurations[i] = {eps/multiplier: HDBSCAN(min_cluster_size = i,
                        min_samples = min_samples,
                        cluster_selection_method =cluster_selection_method,
                        cluster_selection_epsilon=eps/multiplier,
                        gen_min_span_tree=True,
                        memory=r'./tmp_hdbscan_cache/',
                        prediction_data=True) for eps in range(0,int(multiplier*maxeps)+int(multiplier*epsstep),int(multiplier*(epsstep)))}   #(0, maxeps+epsstep, epsstep)   } #np.linspace(0,maxeps,int(maxeps/epsstep))}
            


                #configurations.append({'n_neighbors':n_neighbors, 'n_components':number_dimensions, 'metric':umap_metric, 'min_dist':min_dist, 'random_state':random_state})   
            st.write(configurations)
                       
            st.session_state.umap_models = []
            st.session_state.umap_models.append(umap_models)
                #####start added after demo####
            st.session_state.viz_models = []
            st.session_state.viz_models.append(viz_models) 

                ###st.session_state.number_dimensions = []
                ###st.session_state.number_dimensions = number_dimensions
                #####end added after demo####
            st.session_state.configurations = []
            st.session_state.configurations.append(configurations)
            st.session_state.n_neighbors_min = []
            st.session_state.n_neighbors_min.append(n_neighbors_min)   
            st.session_state.n_neighbors_max = []
            st.session_state.n_neighbors_max.append(n_neighbors_max)
            st.session_state.n_neighbors_step = []
            st.session_state.n_neighbors_step.append(nn_step)
            st.session_state.min_cluster_size_min = []
            st.session_state.min_cluster_size_min.append(min_cluster_size_min)
            st.session_state.min_cluster_size_max = []
            st.session_state.min_cluster_size_max.append(min_cluster_size_max)
            st.session_state.cluster_step = []
            st.session_state.cluster_step.append(cluster_step)
            st.session_state.maxeps = []
            st.session_state.maxeps.append(maxeps)  
            st.session_state.epsstep = []
            st.session_state.epsstep.append(epsstep)
            
            if 'umap_metric' not in st.session_state:
                st.session_state.umap_metric = []
                st.session_state.umap_metric.append(umap_metric)

            if 'random_state' not in st.session_state:
                st.session_state.random_state = []
                st.session_state.random_state.append(random_state)
                 #st.markdown('Configurations to compare:')
                #st.write(st.session_state.n_neighbors_min[0])
            
           

     
        if st.sidebar.button("Calculate and plot results", key = 'calculate_and_plot'):
            
            try:
                if ((uploaded_file is not None) & (  (len(st.session_state.umap_models[0])) > 0)):
                        #st.subheader(f"UMAP n_neigbors={n_neighbors}, metric={umap_metric}, min_dist={min_dist}; HDBSCAN min_samples = {min_samples}, \
                        #             min_cluster_size = {min_cluster_size}, metric = {hdbscan_metric}, cluster_selection_epsilon = {cluster_selection_epsilon},\
                        #             cluster_selection_method = {cluster_selection_method}")

         
                    st.write("now going to run dimred")
                    dimred = {nn:umap.UMAP(n_neighbors = nn, n_components = number_dimensions, min_dist=min_dist, metric = umap_metric, random_state=random_state).fit_transform(data) for nn in range(st.session_state.n_neighbors_min[0] , st.session_state.n_neighbors_max[0] + st.session_state.n_neighbors_step[0], st.session_state.n_neighbors_step[0] )}
                    st.write("dimred has been run, here are the results:")
                    st.write(dimred)
                    st.write(dimred.keys())
                    st.write(dimred[st.session_state.n_neighbors_min[0]])
                    st.write(data)
                    st.write("XXX")
                #####start added after demo####
                ### added after OPTI
                    if number_dimensions > 2:
                        vizred = {nn:umap.UMAP(n_neighbors = nn, n_components = 2, min_dist=min_dist, metric = umap_metric, random_state=random_state).fit_transform(data) for  nn in [n_neighbors[0] , n_neighbors[1]] }   #nn in range(st.session_state.n_neighbors_min[0] , st.session_state.n_neighbors_max[0] + st.session_state.n_neighbors_step[0], st.session_state.n_neighbors_step[0] )
                    else:
                        vizred = dimred
                #####end  added after demo####
            
                    results_incl_eps = {nn:cluster_with_all_models_incl_eps(dimred[nn], st.session_state.configurations[0]) for nn in range(st.session_state.n_neighbors_min[0] , st.session_state.n_neighbors_max[0] + st.session_state.n_neighbors_step[0], st.session_state.n_neighbors_step[0] )}
                    st.write(results_incl_eps)

                    results = results_incl_eps

                # for each UMAP model I will have 1 dataframe with mcs and eps
                    dataframes = {}
                    for i in range(st.session_state.n_neighbors_min[0] , st.session_state.n_neighbors_max[0] + st.session_state.n_neighbors_step[0], st.session_state.n_neighbors_step[0] ):
                        dataframes[i] = pd.DataFrame(results[i])
                        dataframes[i] = dataframes[i].T

                    for i in dataframes:
                        dataframes[i] = dataframes[i].stack()
            
                    for i in dataframes:
                        dataframes[i] = pd.DataFrame(dataframes[i])

            # now I am going to pull them together by concatenating them all

                    df = pd.concat(dataframes.values(), axis = 1)     # the resulting df has a multiindex consisting of mcs - eps nd the columns correspond to the different UMAP
                                                                # models defined by nn values, but they will all be called  0

                    #df.to_csv('halleluja.csv')
                #let's give the columns correct names
                    df.columns = [i for i in range(st.session_state.n_neighbors_min[0] , st.session_state.n_neighbors_max[0] + st.session_state.n_neighbors_step[0], st.session_state.n_neighbors_step[0] )]

                    df = pd.DataFrame(df.unstack(level=[-2]).T.stack())

                    df['n_clusters'] = df[0].apply(lambda x: (len(np.unique(x)))-1)   
                    df['percent_unclustered'] = df[0].apply(lambda x: 100*np.count_nonzero(x==-1)/len(x))

                        #df.to_csv('hallelujaX.csv')
                    df= df.reset_index()
           
                    df = df.rename(columns = {'level_0':'(UMAP) n_neighbors', 'level_1':'min_cluster_size', 'level_2':'eps'})   


                    #df.to_csv('hallelujaY.csv')


                    low_color = df['percent_unclustered'].min()
                    if round(low_color)>low_color:
                        low_color = round(low_color)-1

                    high_color = df['percent_unclustered'].max()
                    high_color = round(high_color)

                    hyperparameters_plot = px.scatter(df, x = 'min_cluster_size', y = 'eps', 
                                size = 'n_clusters', size_max = 100, color = 'percent_unclustered',     
                                color_continuous_scale='Rainbow', animation_frame='(UMAP) n_neighbors', height=1000, range_color = [low_color,high_color],
                                title = f'Number of clusters (bubble size) and % unclustered points (color) for different combinations of UMAP and HDBSCAN models<br>with varying n_neighbors (slider), minimal_cluster_size (x axis) and cluster_slection_epsilon (y axis)<br><sup>min_samples = {min_samples}, cluster_selection_method = {cluster_selection_method}, clustering in {number_dimensions}D')

                    hyperparameters_plot["layout"].pop("updatemenus") # remove animation buttons
            #st.session_state.curr_plot = []
            #st.session_state.curr_plot.append(hyperparameters_plot)

            
                     
            
                    hyperparameters_plot.write_html(f"{number_dimensions}D_{umap_metric}_md{min_dist}_min_samples{min_samples}_{cluster_selection_method}.html")
                    #hyperparameters_plot.show('browser')



                    compliant_results_eps0 = df.loc[(
                        (df['n_clusters'] >= min_acceptable_n_clusters) & (df['n_clusters'] <= max_acceptable_n_clusters)
                                )
                                &
                                (
                        (df['eps'] == 0.00)

                                )
                                    ]
                    compliant_results_eps0 = compliant_results_eps0.sort_values(by=['percent_unclustered'])

                    compliant_results_all = df.loc[(
                            (df['n_clusters'] >= min_acceptable_n_clusters) & (df['n_clusters'] <= max_acceptable_n_clusters)
                            )
                        #&
                        #(
                        # (df['eps'] == 0.00)

                        #)
                                                ]
            
                    compliant_results_eps0 = compliant_results_eps0.sort_values(by=['percent_unclustered'])
            
                    compliant_results_all = compliant_results_all.sort_values(by=['percent_unclustered'])

                
                  
            # 
            #    
                    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([ "2D plots", "Summary plot", "Recommendations", "Clusterings vs hyperparameters", "Diagnostic plots", "Models in use"])

                    


                    with tab1:
                        st.header("2D plots using the min and max values of n_neighbors")

                        i=1 
                        for vizmodel in [  st.session_state.n_neighbors_min[0], st.session_state.n_neighbors_max[0]  ]:
                            curr_fig = 'fig_'+str(i)   

                            curr_fig =  px.scatter(data, x = vizred[vizmodel][:, 0],
                                                     y = vizred[vizmodel][:, 1],
                                                     hover_data={data.index.name: (data.index)},
                                                     title = f'n_neighbors {vizmodel}')                                #color = [str(int(i)) for i in results[vizmodel][mcs][0]]
                        
                            st.plotly_chart(curr_fig, use_container_width=True)
                            curr_fig.write_html(f"fig_{str(i)}.html")
                            #curr_fig.show('browser')
                            i=i+1
                    
              

                    with tab2:
                        st.header("Summary plot")
                        st.plotly_chart(hyperparameters_plot, use_container_width=True)

                    
                    with tab3:
                        st.header("Recommendations")
                        

                        st.write('Clusterings compliant with the requirements at the highest granularity level:')
                        st.dataframe(compliant_results_eps0[['(UMAP) n_neighbors', 'min_cluster_size', 'eps', 'n_clusters', 'percent_unclustered']])
                    
                        st.write('All clusterings compliant with the requirements:')
                        st.dataframe(compliant_results_all[['(UMAP) n_neighbors', 'min_cluster_size', 'eps', 'n_clusters', 'percent_unclustered']])


            
                    with tab4:
                        st.header("Clusterings vs hyperparameters plots")
                        unclustered_vs_mcs_line = px.line(df, x = 'min_cluster_size', y = 'percent_unclustered', 
                                    animation_frame='(UMAP) n_neighbors', height=500, color = 'eps' , range_y = [0,1.05*(df['percent_unclustered'].max())] )

            
                        st.plotly_chart(unclustered_vs_mcs_line)   #use_container_width=True

                               
                        unclustered_vs_mcs = px.scatter(df, x = 'min_cluster_size', y = 'percent_unclustered', 
                                     size = 'n_clusters', size_max = 50, animation_frame='(UMAP) n_neighbors', height=1000, color = 'eps', color_continuous_scale='Turbo', range_y = [-5,1.05*(df['percent_unclustered'].max())])
                        st.plotly_chart(unclustered_vs_mcs, use_container_width=True) 

               
                    with tab5:  
                        st.subheader('Connectivity plots')
                        if plot_connectivity == 'Yes':
                            plt.figure(figsize=(7,5))
                            diagnostic_results_min_nn = umap.UMAP(n_neighbors = st.session_state.n_neighbors_min[0], n_components = 2, min_dist=0.0, 
                                                         metric = umap_metric, random_state=random_state).fit(data)
                    
                            umap.plot.connectivity(diagnostic_results_min_nn, show_points=True, theme="viridis", width = 1800, edge_bundling='hammer')

                            st.pyplot(plt)
                        else:
                            st.write('You did not choose a connectivity plot to be made.')
                    
                    with tab6:
                        st.header("UMAP and HDBSCAN models in use & results")
                            #st.write(results)
                        st.write("UMAP models:")
                        st.write(st.session_state.umap_models[0])
                        st.write("HDBSCAN models:")
                        st.write(st.session_state.configurations[0])
                        st.write("Results for all configurations:")
                        st.write(results_incl_eps)

                      #for mcs in range(st.session_state.min_cluster_size_min[0],st.session_state.min_cluster_size_max[0] + st.session_state.cluster_step[0], st.session_state.cluster_step[0]):
                        #i=1
                                  
                        #curr_fig =  px.scatter(data, x = vizred[vizmodel][:, 0], y = vizred[vizmodel][:, 1], hover_data={data.index.name: (data.index)}, color = [str(int(i)) for i in results[vizmodel][mcs][0]], title = f'n_neighbors {vizmodel}, minimal_cluster_size {mcs}, cluster_lection_epsilon 0, min_samples = {min_samples}, cluster_selection_method = {cluster_selection_method}, clustering in {number_dimensions}D')
                        
                #### end added after demo

                #histogram_plot = px.histogram(results[6][16][0.15])     # no I don't want this, this shows how many points are in clusters -1 + 0, 2 + 3, etc; 
                #st.plotly_chart(histogram_plot, use_container_width=True)
                #tmp = pd.DataFrame(results[6][16][0.15])
                #tmp.to_csv('tmp.csv')
                else:
                    st.write(f'Please make sure to load valid data (currently no missing values are allowed) and create models first. Uploaded file: {uploaded_file}, len models: {len(st.session_state.umap_models[0])}')
            except BaseException as error:
                    st.write('An exception occurred: {}'.format(error))
                    st.write(f'(Try except)Please make sure to load valid data (currently no missing values are allowed) and create models first. Uploaded file: {uploaded_file}, len models: {len(st.session_state.umap_models[0])}')

if __name__ == '__main__':
    main()
