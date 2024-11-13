# """
# Calculate inertia of cluster hierarchy
# """
import dask.array as da

# calculate the inertia based on a list of [centroid, cluster_items] both of which are dask arrays
def inertia(cluster):
    # calculate inertia of the cluster
    inertia = da.mean(da.sum(cluster[0]-cluster[1],axis=1)**2).compute()
    return inertia