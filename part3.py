# %% read dataframe
import pandas as pd

data = pd.read_pickle("data.pkl")


print(data.columns)
#%%
import seaborn as sns

Force_used = [
    col for col in data.columns if col.startswith("pf_") 
]
(data[Force_used] == "YES").sum(axis=1)

data["number_of_reasons"] = (data[Force_used] == "YES").sum(axis=1)
sns.countplot(data=data, y="forceuse", hue="number_of_reasons")
#%%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load your data into the 'data' DataFrame

# Create the "number_of_reasons" column
Force_used = [col for col in data.columns if col.startswith("pf_") ]
data["number_of_reasons"] = (data[Force_used] == "YES").sum(axis=1)

# Create a grouped bar plot
#sns.set(style="whitegrid")
#plt.figure(figsize=(10, 6))
#sns.barplot(data=data, x="forceuse", y="number_of_reasons", ci=None)
#plt.xlabel("Force Used")
#plt.ylabel("Count")
#plt.xticks(rotation=45)
#plt.title("Distribution force used ")
#plt.show()

#%%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from kmodes.kmodes import KModes

# Load your data into the 'data' DataFrame

# Create the "number_of_reasons" column
Force_used = [col for col in data.columns if col.startswith("pf_")]
data["number_of_reasons"] = (data[Force_used] == "YES").sum(axis=1)

# Select columns for clustering
X = data[Force_used]

# Initialize the KModes clustering algorithm
n_clusters = 3  # You can adjust the number of clusters as needed
km = KModes(n_clusters=n_clusters, init="Huang", n_init=5, verbose=1)

# Fit the model to your data
clusters = km.fit_predict(X)

# Add the cluster labels to your DataFrame
data["cluster_label"] = clusters


#%%

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from kmodes.kmodes import KModes
import folium  

map_center = [data["lat"].mean(), data["lon"].mean()]
m = folium.Map(location=map_center, zoom_start=10)

# Add markers for each data point with cluster information
for index, row in data.iterrows():
    folium.Marker(
        location=[row["lat"], row["lon"]],
        popup=f"Cluster: {row['cluster_label']}",
        icon=folium.Icon(color="blue")
    ).add_to(m)

# Display the map
m.save("cluster_map.html")  # Save the map as an HTML file
#%%
m

#%%
# Visualize the clusters
import seaborn as sns
import matplotlib.pyplot as plt
from kmodes.kmodes import KModes

sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
scatterplot = sns.scatterplot(data=data, x="forceuse", y="number_of_reasons", hue="cluster_label")

# Rotate x-axis labels
scatterplot.set_xticklabels(scatterplot.get_xticklabels(), rotation=45)

plt.xlabel("Force Use")
plt.ylabel("Clustor")
plt.title("Clustered Scatterplot with Rotated X-Axis Labels")
plt.legend(title="Cluster Label", loc="upper right")
plt.show()




#%%
import seaborn as sns

reason_used = [
    col for col in data.columns if col.startswith("cs_") or col.startswith("rf_")
]
(data[reason_used] == "YES").sum(axis=1)

data["number_of_reasons"] = (data[reason_used] == "YES").sum(axis=1)
sns.countplot(data=data, y="forceuse", hue="number_of_reasons")

#%%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load your data into the 'data' DataFrame

# Create the "number_of_reasons" column
reason_used = [col for col in data.columns if col.startswith("cs_") or col.startswith("rf_")]
data["number_of_reasons"] = (data[reason_used] == "YES").sum(axis=1)

# Create a grouped bar plot
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
sns.barplot(data=data, x="forceuse", y="number_of_reasons", ci=None)
plt.xlabel("No of Reasons")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.title("Distribution stopped people by number of reasons")
plt.show()


#%%
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# Assuming you have already loaded your data into the 'data' DataFrame

# Encode categorical variables using one-hot encoding
categorical_columns = ["forceuse"]
numeric_columns = ["number_of_reasons"]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(), categorical_columns)
    ],
    remainder="passthrough"
)

X = preprocessor.fit_transform(data)

# Initialize the KMeans clustering algorithm
kmeans = KMeans(n_clusters=3)  # You can adjust the number of clusters as needed

# Fit the model to your data
kmeans.fit(X)

# Add the cluster labels to your DataFrame
data["cluster_label"] = kmeans.labels_

# Visualize the clusters
sns.scatterplot(data=data, x="forceuse_YES", y="number_of_reasons", hue="cluster_label")
plt.show()



#%%
import pandas as pd
import matplotlib.pyplot as plt
from kmodes.kmodes import KModes

# Load your dataset into a pandas DataFrame
data = pd.read_pickle("data.pkl")  

# Select the relevant categorical columns for clustering
categorical_columns = ["forceuse"]

selected_data = data[categorical_columns]

# Initialize the KModes clustering algorithm
num_clusters = 5 
km = KModes(n_clusters=num_clusters, init='Huang', n_init=5, verbose=1)

# Perform clustering
cluster_ids = km.fit_predict(selected_data)

# Add the cluster IDs to the DataFrame
data['cluster'] = cluster_ids

# Visualize cluster-wise distribution of categorical columns
for column in categorical_columns:
    plt.figure(figsize=(10, 6))
    for cluster_id in range(num_clusters):
        cluster_data = data[data['cluster'] == cluster_id]
        value_counts = cluster_data[column].value_counts(normalize=True)  # Use normalize=True for relative frequencies
        value_counts.sort_index().plot(kind='bar', label=f'Cluster {cluster_id}', alpha=0.7)
    
    plt.xlabel(column)
    plt.ylabel('Relative Frequency')
    plt.title(f'Cluster-wise Distribution of {column}')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


#%%%clustor stopped people by reason for stop

import pandas as pd
import matplotlib.pyplot as plt
from kmodes.kmodes import KModes

# Load your dataset into a pandas DataFrame
data = pd.read_pickle("data.pkl")  

# Select the relevant categorical columns for clustering
categorical_columns = ['cs_objcs', 'cs_descr', 'cs_casng', 'cs_lkout', 'cs_cloth',
                    'cs_drgtr', 'cs_furtv', 'cs_vcrim', 'cs_bulge', 'cs_other']

selected_data = data[categorical_columns]

# Initialize the KModes clustering algorithm
num_clusters = 5  # You can adjust this
km = KModes(n_clusters=num_clusters, init='Huang', n_init=5, verbose=1)

# Perform clustering
cluster_ids = km.fit_predict(selected_data)

# Add the cluster IDs to the DataFrame
data['cluster'] = cluster_ids

# Visualize cluster-wise distribution of categorical columns
for column in categorical_columns:
    plt.figure(figsize=(10, 6))
    for cluster_id in range(num_clusters):
        cluster_data = data[data['cluster'] == cluster_id]
        value_counts = cluster_data[column].value_counts(normalize=True)  # Use normalize=True for relative frequencies
        value_counts.sort_index().plot(kind='bar', label=f'Cluster {cluster_id}', alpha=0.7)
    
    plt.xlabel(column)
    plt.ylabel('Relative Frequency')
    plt.title(f'Cluster-wise Distribution of {column}')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

#%%luster-and silnouse score
import pandas as pd
import matplotlib.pyplot as plt
from kmodes.kmodes import KModes
from sklearn.metrics import silhouette_score

# Load your dataset into a pandas DataFrame
data = pd.read_pickle("data.pkl")  # Replace with your file path

# Select the relevant categorical columns for clustering
categorical_columns = ['cs_objcs', 'cs_descr', 'cs_casng', 'cs_lkout', 'cs_cloth',
                    'cs_drgtr', 'cs_furtv', 'cs_vcrim', 'cs_bulge', 'cs_other']

selected_data = data[categorical_columns]

# Initialize the KModes clustering algorithm
num_clusters = 5  # You can adjust this
km = KModes(n_clusters=num_clusters, init='Huang', n_init=5, verbose=1)

# Perform clustering
cluster_ids = km.fit_predict(selected_data)

# Add the cluster IDs to the DataFrame
data['cluster'] = cluster_ids

# Calculate silhouette score
silhouette_avg = silhouette_score(selected_data, cluster_ids)
print(f"Silhouette Score: {silhouette_avg}")

# Define a list of colors for the bars
colors = [ 'orange', 'green', 'red', 'purple']

# Visualize cluster-wise distribution of categorical columns
for column in categorical_columns:
    plt.figure(figsize=(10, 6))
    for cluster_id in range(num_clusters):
        cluster_data = data[data['cluster'] == cluster_id]
        value_counts = cluster_data[column].value_counts(normalize=True)  # Use normalize=True for relative frequencies
        value_counts.sort_index().plot(kind='bar', label=f'Cluster {cluster_id}', alpha=0.7, color=colors[cluster_id])
    
    plt.xlabel(column)
    plt.ylabel('Relative Frequency')
    plt.title(f'Cluster-wise Distribution of {column}')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()




#%%clustor stopped people by reason for stop

import pandas as pd
from kmodes.kmodes import KModes

# Load your dataset into a pandas DataFrame
data = pd.read_pickle("data.pkl")  # Replace with your file path

# Select the relevant categorical columns for clustering
categorical_columns = ['cs_objcs', 'cs_descr', 'cs_casng', 'cs_lkout', 'cs_cloth',
                    'cs_drgtr', 'cs_furtv', 'cs_vcrim', 'cs_bulge', 'cs_other']

selected_data = data[categorical_columns]

# Initialize the KModes clustering algorithm
num_clusters = 5  # You can adjust this
km = KModes(n_clusters=num_clusters, init='Huang', n_init=5, verbose=1)

# Perform clustering
cluster_ids = km.fit_predict(selected_data)

# Add the cluster IDs to the DataFrame
data['cluster'] = cluster_ids

# Print the counts of data points in each cluster
cluster_counts = data['cluster'].value_counts()
print("Cluster Counts:")
print(cluster_counts)
#%%
import pandas as pd
import matplotlib.pyplot as plt


# Select the relevant categorical columns for clustering
categorical_columns = ['cs_objcs', 'cs_descr', 'cs_casng', 'cs_lkout', 'cs_cloth',
                    'cs_drgtr', 'cs_furtv', 'cs_vcrim', 'cs_bulge', 'cs_other']

# Load cluster assignments from your data (replace 'cluster' with the actual column name)
cluster_assignments = data['cluster']

# Iterate through each categorical column and plot frequency tables or bar plots
for column in categorical_columns:
    plt.figure(figsize=(10, 6))
    for cluster_id in cluster_assignments.unique():
        cluster_data = data[data['cluster'] == cluster_id]
        value_counts = cluster_data[column].value_counts()
        value_counts = value_counts / value_counts.sum()  # Normalize to get relative frequencies
        plt.bar(value_counts.index, value_counts.values, label=f'Cluster {cluster_id}', alpha=0.7)
    
    plt.xlabel(column)
    plt.ylabel('Relative Frequency')
    plt.title(f'Cluster-wise Distribution of {column}')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
#%%#%%
# You can analyze the clusters further if needed
for cluster_id in range(num_clusters):
    cluster_data = data[data['cluster'] == cluster_id]
    print(f"Cluster {cluster_id}:")
    print(cluster_data)
    print("\n")


# %% pick a crime
import pandas as pd

#%%


df = pd.read_pickle("data.pkl")

df_assault = df[df["detailcm"] == "CRIMINAL TRESPASS"]

#%%
#categorical_cols = df.select_dtypes(include=["object"]).columns
#df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

#%%
#from sklearn.decomposition import PCA
#from sklearn.preprocessing import StandardScaler

# Assuming df contains the data
#X = df.drop(columns=["detailcm"])  # Assuming "detailcm" is the target column
#target = df["detailcm"]

# Standardize the features
#scaler = StandardScaler()
#X_scaled = scaler.fit_transform(X)

# Perform PCA
#num_components = 10  # Specify the number of components you want to keep
#pca = PCA(n_components=num_components)
#X_pca = pca.fit_transform(X_scaled)

# Create a DataFrame with the PCA-transformed features
#columns_pca = [f"PC{i+1}" for i in range(num_components)]
#X_pca_df = pd.DataFrame(data=X_pca, columns=columns_pca)

# Concatenate the PCA-transformed features with the target variable
#final_df = pd.concat([X_pca_df, target], axis=1)

#%%
#from sklearn.cluster import AgglomerativeClustering
#from sklearn.metrics import silhouette_score
#from tqdm import tqdm

#scores, labels = {}, {}

#num_reasons = df["detailcm"].nunique()

# %%
#or k in tqdm(range(2, num_reasons + 1)):
#   c = AgglomerativeClustering(n_clusters=k)
#    y = c.fit_predict(df[["perobs", "perstop"]])

#   scores[k] = silhouette_score(df[["perobs", "perstop"]], y)
#    labels[k] = y

#%%

#best_k = max(scores, key=lambda k: scores[k])


# %% apply hierarchical clustering on a range of arbitrary values
# record the silhouette_score and find the best number of clusters

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from tqdm import tqdm

df_assault = df[df["detailcm"] == "CRIMINAL TRESPASS"]
scores, labels = {}, {}
num_city = df["city"].nunique()
num_pct = df["pct"].nunique()

# %%
for k in tqdm(range(num_city, num_pct, 10)):
    c = AgglomerativeClustering(n_clusters=k)
    y = c.fit_predict(df_assault[["lat", "lon"]])
    scores[k] = silhouette_score(df_assault[["lat", "lon"]], y)
    labels[k] = y

# %% find the best k visually
import seaborn as sns

sns.lineplot(x=scores.keys(), y=scores.values(), color = 'green')


# %% find the best k by code
best_k = max(scores, key=lambda k: scores[k])
print(f"The best value of k is: {best_k}")

# %% visualize the hierarchcal clustering result
import folium

m = folium.Map((40.7128, -74.0060))
colors = sns.color_palette("hls", best_k).as_hex()
df_assault["label"] = labels[best_k]
for r in df_assault.to_dict("records"):
    folium.CircleMarker(
        location=(r["lat"], r["lon"]), radius=1, color=colors[r["label"]]
    ).add_to(m)

m


# %% find reason for stop columns
# and apply dbscan
from sklearn.cluster import DBSCAN

css = [col for col in df.columns if col.startswith("cs_")]
c = DBSCAN()
x = df_assault[css] == "YES"
y = c.fit_predict(x)
print(silhouette_score(x, y))


# %% visualize the result on map
import numpy as np

m = folium.Map((40.7128, -74.0060))
k = len(np.unique(y))
colors = sns.color_palette("hls", k).as_hex()
df_assault["label"] = y
for r in df_assault.to_dict("records"):
    folium.CircleMarker(
        location=(r["lat"], r["lon"]), radius=1, color=colors[r["label"]]
    ).add_to(m)

m

# %%
df_assault["label"].value_counts()


# %% pick a label and visualize the datapoints on map
biggest_cluster = 12
#biggest_cluster = df_assault["label"].value_counts().index[0]
m = folium.Map((40.7128, -74.0060))
for r in df_assault[df_assault["label"] == 5].to_dict("records"):
    folium.CircleMarker(
        location=(r["lat"], r["lon"]), radius=1, color=colors[r["label"]]
    ).add_to(m)

m

# %%
