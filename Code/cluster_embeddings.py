from datetime import datetime 
import pathlib
import math
import random
import os 

import numpy as np
import polars as pl
import pandas as pd
import plotnine as p9

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import umap
import hdbscan


def filter_frames(data):
    """
    Filters the input data by removing rows with zeros above a certain threshold. The threshold is defined as the upper fence of the zero distribution.

    Parameters:
    -----------
    data (polars.DataFrame): A polars DataFrame.

    Returns:
    -----------
    A polars DataFrame with rows that have a high number of zeros filtered.
    """
    df = data.to_pandas()
    df_zeros_per_row = (df == 0).astype(int).sum(axis = 1)
    df_summary = pd.DataFrame(df_zeros_per_row.describe())
    df_summary.index.name = "count"
    df_summary.columns = ["counts"]
    lower_fence = df_summary.loc["25%"] - 1.5*(df_summary.loc["75%"] - df_summary.loc["25%"]) 
    upper_fence = df_summary.loc["75%"] + 1.5*(df_summary.loc["75%"] - df_summary.loc["25%"])
    tmp = df_zeros_per_row[df_zeros_per_row <= math.ceil(upper_fence.iloc[0])].index.to_list()
    df = df.filter(tmp, axis = 0)
    return pl.from_pandas(df)

def filter_features(data, df_feature_names, filter_features):
    """
    Filters the features in the given data based on the specified filter features.

    Parameters:
    -----------
    data (pandas.DataFrame): Data to filter.
    df_feature_names (pandas.DataFrame): The feature names.
    filter_features (list, optional): The list of filter features. Defaults to ["TAIL", "corner"].

    Returns:
    ------------
    df (pandas.DataFrame): Data containing filtered features. 
    """
    #meta_cols = ["animal_idx", "start", "duration", "is_behavior", "time", "exp_prefix", "video_name", "longterm_idx", "bout_number"]
    meta_cols = list(set(data.columns) - set(df_feature_names.feature_index))
    remove_features = df_feature_names[df_feature_names.feature.apply(lambda x: any([name in x for name in filter_features]))].feature #filter features
    keep_features = df_feature_names[~df_feature_names.feature.isin(remove_features)].feature #keep features
    df = data.select(meta_cols + keep_features)
    return df
    
def filter_bouts(data, lb=45):
    """
    Filters bouts based on the number of counts in the data.

    Parameters:
    -----------
    data (polars.DataFrame): A dataframe containing the data to be filtered.
    lb (int): A lower bound for the number of counts. Defaults to 45.

    Returns:
    -----------
    dfpl_filtered (polars.DataFrame): A filtered dataframe containing bouts with counts between lb and upper_fence.
    """
    df_bout_number = data.group_by("bout_number").agg(pl.count("bout_number").alias("counts"))
    summary_array = df_bout_number.describe().select(pl.col("describe", "counts")).to_pandas().set_index("describe")
    upper_fence = summary_array.loc["75%"] + 1.5 * (summary_array.loc["75%"] - summary_array.loc["25%"])
    dfpl_filtered_bout_number = data.group_by("bout_number").agg(pl.count("bout_number").alias("counts")).filter(pl.col("counts").is_between(lb, upper_fence.iloc[0])).select(pl.col("bout_number"))
    dfpl_filtered = data.filter(pl.col("bout_number").is_in(dfpl_filtered_bout_number["bout_number"]))
    return dfpl_filtered

def downsample_bouts(data, seed, thresh=50):
    """
    Downsamples bouts in the input data to a maximum length of `thresh`.

    Parameters:
    -----------
    data (pandas.DataFrame): Input data containing bouts to be downsampled.
    seed (int): Seed for the random number generator.
    thresh (int): Maximum length of bouts to keep. Default is 50.

    Returns:
    --------
    df (pandas.DataFrame): Downsampled data containing bouts with length <= `thresh`.
    """
    random.seed(seed)
    df = data.filter(pl.int_range(0, pl.count()).shuffle().over("bout_number") < thresh)
    return df

def compute_pca(data, method = "pca", scale = True, thres = 0.85):
    """
    Compute PCA on the input data and return the transformed data.

    Parameters:
    -----------
    data (pandas.DataFrame): Input data to be transformed.
    method (str): Method to be used for PCA. Default is "pca".
    scale (bool): Whether to scale the input data or not. Default is True.
    thres (float): Threshold for variance explained by the principal components. Default is 0.85.

    Returns:
    -----------
    Z[:, :(n_pc+1)] (numpy.ndarray): Transformed data after applying PCA where n_pc is the number of principal components that explain `thres` variance.
    """
    X = data.select(pl.col(pl.Float64)).to_numpy()
    if scale == True:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        #X = (X - X.mean(axis = 0))/X.std(axis = 0)

    if method == "pca":
        embedding = PCA()
        Z = embedding.fit_transform(X)
        n_pc = np.argmax(np.cumsum(embedding.explained_variance_ratio_) > thres)
    else:
        Z = X
        n_pc = Z.shape[1]

    return Z, n_pc, embedding, scaler

def compute_embeddings(X, method = "umap", n_neighbors = 30, min_dist = 0.0, metric = "euclidean"):
    """
    Computes embeddings for a given dataset using the specified method.

    Parameters:
    -----------
    X (numpy.ndarray): The dataset to compute embeddings for.
    method (str): The method to use for computing embeddings. Defaults to "umap".
    n_neighbors (int): The number of neighbors to use for computing embeddings. Defaults to 30. Balances local versus global structure in the UMAP space. Lower values will force the algorithm to concentrate on very local structure (potentially to the detriment of the big picture), while larger values will push it into paying attention to the broader global structure
    min_dist (float): The minimum distance between points in the embedding. Defaults to 0.0. Minimum distance between points in the lower dimensional space. Smaller values will result in a more clustered/clumped embedding where nearby points on the manifold are drawn closer together, while larger values will result in preserving the broad structure.
    metric (str): The distance metric to use for computing embeddings. Defaults to "euclidean".

    Returns:
    -----------
    Z (numpy.ndarray): The computed embeddings.
    """
    if method == "umap":
        embedding = umap.UMAP(n_neighbors = n_neighbors, min_dist = min_dist, metric = metric)
        Z = embedding.fit_transform(X)        
    return Z, embedding
    
def plot_embeddings(embeddings):
    """
    Plots the embeddings in a 2D scatter plot using ggplot2.

    Parameters:
    -----------
    embeddings (pandas.DataFrame): A pandas DataFrame containing the embeddings to be plotted.

    Rreturns:
    -----------
    p (ggplot2 object): A ggplot2 object representing the scatter plot.
    """
    df = pd.DataFrame(embeddings[:,:2], columns = ["Dim1", "Dim2"])
    p = p9.ggplot(df, p9.aes(x = "Dim1", y = "Dim2")) + p9.geom_point() + p9.theme_bw() + p9.theme(legend_position = "none")
    return p

def cluster_embeddings(data, embeddings, method = "hdbscan", kwargs = {"min_cluster_size": 500, "min_samples": 500, "cluster_selection_epsilon": 0.1}):
    """
    Cluster the embeddings using the specified method.

    Parameters:
    -----------
    data (polars.DataFrame): The data containing the original features and bout information.
    embeddings (numpy.ndarray): The embeddings to be clustered.
    method (str): The clustering method to be used. Default is "hdbscan".
    kwargs (dict): The keyword arguments to be passed to the clustering method. Default is {"min_cluster_size": 500, "min_samples": 500, "cluster_selection_epsilon": 0.1}. 
    min_cluster_size (int) : Smallest size grouping that can be considered a cluster.
    min_samples (int) : Number of samples in a neighborhood for a point to be considered a core point. The larger the value, the more conservative the clustering and more points will be declared as noise.
    cluster_selection_epsilon (float) : Ensures that clusters are always within this distance of each other. This parameter controls the notion of “cohesiveness” in the clustering. Larger values mean that subclusters will be encouraged to merge into larger clusters, while smaller values mean that smaller clusters will be maintained.


    Returns:
    -----------
    df_cluster (pandas.DataFrame): A dataframe containing the clustered embeddings.
    """
    if method == "hdbscan":
        cluster = hdbscan.HDBSCAN(min_cluster_size=kwargs["min_cluster_size"], min_samples=kwargs["min_samples"], cluster_selection_epsilon=kwargs["cluster_selection_epsilon"], prediction_data=True)
        cluster.fit(embeddings)
        df_cluster = pd.DataFrame(embeddings)
        df_cluster.columns = ["Dim1", "Dim2"]
        df_cluster = df_cluster.assign(bout = pd.Series(data['bout_number']).astype("O"))
        df_cluster = df_cluster.assign(Cluster = pd.Series(cluster.labels_).astype("O"))
        df_cluster = df_cluster.assign(prob = pd.Series(cluster.probabilities_).astype("O"))       
    elif method == "kmeans":
        pass #TODO (Doesn't seem necessary for now)
    return df_cluster, cluster

def plot_clusters(data, prob = 0.7, annotate = False):
    """
    Plots the clusters of the given data containing the embeddings and cluster metrics.

    Parameters:
    -----------
    data (pandas.DataFrame): A pandas DataFrame containing the data to be plotted.
    prob (float): The probability threshold for the clusters. Defaults to 0.7.
    annotate (bool): Whether to annotate the plot with the annotations or not. Defaults to False.

    Returns:
    -----------
    p (ggplot2 object): A ggplot object representing the plot of the clusters.
    """
    data = data[data.prob > prob]

    if annotate == True:
        p = p9.ggplot(data, p9.aes(x = "Dim1", y = "Dim2", color = "Cluster")) + p9.geom_point() + p9.theme_minimal() + p9.scale_color_discrete(guide=False) + p9.labs(x = "UMAP 1", y = "UMAP 2", subtitle = "Clusters with probability > 0.7 + Annotated Data") + p9.theme(legend_position = None) + p9.geom_point(data[data.Type == "Annotated"], p9.aes(x = "Dim1", y = "Dim2"), color = "black", size = 2)
    else:
        p = p9.ggplot(data, p9.aes(x = "Dim1", y = "Dim2", color = "Cluster")) + p9.geom_point() + p9.theme_minimal() + p9.scale_color_discrete(guide=False) + p9.labs(x = "UMAP 1", y = "UMAP 2", subtitle = "Clusters with probability > 0.7") + p9.theme(legend_position = None)
    
    return p

def embed_annotated_frames(data, pca_embedding, umap_embedding, feature_scaler, filter_features = ["TAIL", "corner"]):
    """
    Embeds the annotated frames using the embeddings computed on the unannotated frames.

    Parameters:
    -----------
    data (pandas.DataFrame): The data containing the annotated frames.
    pca_embedding (sklearn.decomposition.pca.PCA): The PCA embedding computed on the unannotated frames.
    umap_embedding (umap.umap_.UMAP): The UMAP embedding computed on the unannotated frames.
    feature_scaler (sklearn.preprocessing._data.StandardScaler): The feature scaler computed on the unannotated frames.
    filter_features (list, optional): The list of filter features. Defaults to ["TAIL", "corner"]. Note: This should be the same as the filter_features used for the unannotated frames.

    Returns:
    -----------
    df_embeddings (pandas.DataFrame): A dataframe containing the embeddings of the annotated frames along with bout information.
    """
    if len(filter_features) != 0:
        meta_cols = list(set(data.columns) - set(df_feature_names.feature))
        remove_features = df_feature_names[df_feature_names.feature.apply(lambda x: any([name in x for name in filter_features]))].feature #filter features
        keep_features = df_feature_names[~df_feature_names.feature.isin(remove_features)].feature #keep features
        data = data.select(meta_cols + list(keep_features))
    else:
        data = data

    X = data.select(pl.col(pl.Float64)).to_numpy()
    X = feature_scaler.transform(X)
    X_pc = pca_embedding.transform(X)[:, :(pca_embedding.n_components_+1)]
    Z = umap_embedding.transform(X_pc)

    df_embeddings = pd.DataFrame(Z)
    df_embeddings.columns = ["Dim1", "Dim2"]
    df_embeddings["bout"] = data["bout"].to_numpy()
    df_embeddings["Type"] = "Annotated"
    
    return df_embeddings
    


#Some helpful variables that don't need to be changed
np.random.seed(123)
date = datetime.today().strftime("%Y_%m_%d")
save_path = pathlib.Path('clustering_results/')

if not save_path.exists():
    save_path.mkdir()
    
# Load the data
df_features = pl.read_csv("Data/behavior_features.csv")
df_feature_names = pd.read_csv("Data/feature_names.csv")

# Filter frames
df = filter_frames(df_features)

#Filter features
filter_features = False
if filter_features == True:
    df = filter_features(data = df, df_feature_names = df_feature_names, filter_features = ["TAIL", "corner"])

# Filter bouts
df = filter_bouts(df, lb=45)

# Downsample bouts
df = downsample_bouts(df, seed=123, thresh=50)

#Compute PCA
X_pc, n_pc, pc_embedding, scaler = compute_pca(df) 

#Compute embeddings
embedding_method = "umap"
n_neighbors = 200 
min_dist = 0.5 
Z, umap_embedding = compute_embeddings(X_pc[:,:(n_pc + 1)], method = embedding_method, n_neighbors=n_neighbors, min_dist=min_dist)

#Plot the embeddings
#plot_embeddings(Z) #not necessary to save this plot

# Cluster the embeddings
clustering_method = "hdbscan"
min_cluster_size = 1000
min_samples = 500
cluster_selection_epsilon = 0.1
df_cluster, clusterer = cluster_embeddings(data = df, method = clustering_method, embeddings = Z, kwargs = {"min_cluster_size": min_cluster_size, "min_samples": min_samples, "cluster_selection_epsilon": cluster_selection_epsilon})

#Save the dataframe containing the embeddings and other cluster variables for further post-processing
df_cluster.to_csv(f"clustering_results/df_cluster_embedding_method_{embedding_method}_n_neighbors_{n_neighbors}_min_dist_{min_dist}_clustering_method_{clustering_method}_min_cluster_size_{min_cluster_size}_min_samples_{min_samples}_cluster_selection_epsilon_{cluster_selection_epsilon}_{date}.csv", index=False)

# Plot the clusters
p = plot_clusters(df_cluster, prob = 0.7)
p.save(f"clustering_results/plot_clusters_embedding_method_{embedding_method}_n_neighbors_{n_neighbors}_min_dist_{min_dist}_clustering_method_{clustering_method}_min_cluster_size_{min_cluster_size}_min_samples_{min_samples}_cluster_selection_epsilon_{cluster_selection_epsilon}_{date}.pdf", width=10, height=10, dpi=300)

# Embed annotations
if os.path.isfile("Data/annotated_bouts.csv"):
    df_annotations = pl.read_csv("Data/annotated_bouts.csv")
    df_anno_embeddings = embed_annotated_frames(df_annotations, pc_embedding, umap_embedding, scaler, filter_features = [])
    df_anno_embeddings["Cluster"], df_anno_embeddings["prob"] = hdbscan.approximate_predict(clusterer, df_anno_embeddings[["Dim1", "Dim2"]])
    df_cluster["Type"] = "Unannotated"
    df_both = pd.concat([df_cluster, df_anno_embeddings], axis = 0)
    df_both.to_csv(f"clustering_results/df_both_embedding_method_{embedding_method}_n_neighbors_{n_neighbors}_min_dist_{min_dist}_clustering_method_{clustering_method}_min_cluster_size_{min_cluster_size}_min_samples_{min_samples}_cluster_selection_epsilon_{cluster_selection_epsilon}_{date}.csv", index=False)
    p = plot_clusters(df_both, prob = 0.7, annotate = False)
    p.save(f"clustering_results/plot_clusters_with_annotated_data_embedding_method_{embedding_method}_n_neighbors_{n_neighbors}_min_dist_{min_dist}_clustering_method_{clustering_method}_min_cluster_size_{min_cluster_size}_min_samples_{min_samples}_cluster_selection_epsilon_{cluster_selection_epsilon}_{date}.pdf", width=10, height=10, dpi=300)





