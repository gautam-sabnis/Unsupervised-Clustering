## Unsupervised Clustering
Identify subtypes of behavior using JABS features as input to unsupervised clustering algorithms.

### Requirements 
Run ```Code/cluster_embeddings.py``` on the cluster using the slurm script ```Slurm/cluster_embeddings.sh```. The slurm script uses a conda environment that can be created using the ```environment.yml``` file.
```bash
conda env create -f environment.yml
```

### Steps

0. Some helpful variables that don't need to be changed
```python
date = datetime.today().strftime("%Y_%m_%d")
save_path = pathlib.Path('clustering_results/')

if not save_path.exists():
    save_path.mkdir()
```

1. Load the data
```python
df_features = pl.read_csv("Data/behavior_features.csv")
df_feature_names = pd.read_csv("Data/feature_names.csv")
```

2. Filter frames
```python
df = filter_frames(df_features)
```

3. Filter features
```python
filter_features = False
if filter_features == True:
    df = filter_features(data = df, df_feature_names = df_feature_names, filter_features = ["TAIL", "corner"]) #filters/removes features that contain the strings "TAIL" or "corner"
```

4. Filter bouts
```python
df = filter_bouts(df, lb=45)
```

5. Downsample bouts
```python
df = downsample_bouts(df, seed=123, thresh=50)
```

6. Compute the principal components 
```python
X_pc = compute_pca(df)
```

7. Compute the UMAP embedding
```python
embedding_method = "umap"
n_neighbors = 200
min_dist = 0.5 
Z = compute_embeddings(X_pc, method = embedding_method, n_neighbors=n_neighbors, min_dist=min_dist)
```

8. Cluster the data using the HDBSCAN algorithm using the embeddings as input
```python
clustering_method = "hdbscan"
min_cluster_size = 1000
min_samples = 500 
cluster_selection_epsilon = 0.1  
df_cluster = cluster_embeddings(data = df, method = clustering_method, embeddings = Z, kwargs = {"min_cluster_size": min_cluster_size, "min_samples": min_samples, "cluster_selection_epsilon": cluster_selection_epsilon})
```

9. Save the dataframe containing the embeddings and other cluster variables for further post-processing
```python
df_cluster.to_csv(f"clustering_results/df_cluster_embedding_method_{embedding_method}_n_neighbors_{n_neighbors}_min_dist_{min_dist}_clustering_method_{clustering_method}_min_cluster_size_{min_cluster_size}_min_samples_{min_samples}_cluster_selection_epsilon_{cluster_selection_epsilon}_{date}.csv", index=False)
```

10. Plot the clusters
```python
p = plot_clusters(, prob = 0.7)
p.save(f"clustering_results/plot_clusters_embedding_method_{embedding_method}_n_neighbors_{n_neighbors}_min_dist_{min_dist}_clustering_method_{clustering_method}_min_cluster_size_{min_cluster_size}_min_samples_{min_samples}_cluster_selection_epsilon_{cluster_selection_epsilon}_{date}.pdf", width=10, height=10, dpi=300)
```
 
11. To embed previously annotated data into the UMAP space, make sure the file containing annotations is named `annotated_bouts.csv`. The annotated frames will appear in the plot as black dots. 
