import sys
import numpy as np
import random
import matplotlib.pyplot as plt
import math as m
import importlib
from mpl_toolkits.mplot3d import Axes3D
import h5py

def cluster(distances, k):

    m = distances.shape[0] # number of points

    # Pick k random medoids.
    curr_medoids = np.array([-1]*k)
    while not len(np.unique(curr_medoids)) == k:
        curr_medoids = np.array([random.randint(0, m - 1) for _ in range(k)])
    old_medoids = np.array([-1]*k) 
    new_medoids = np.array([-1]*k)

    # To be repeated until mediods stop updating
    while not ((old_medoids == curr_medoids).all()):
        # Assign each point to cluster with closest medoid.
        clusters = assign_points_to_clusters(curr_medoids, distances)
        # Update cluster medoids to be lowest cost point. 
        for curr_medoid in curr_medoids:
            cluster = np.where(clusters == curr_medoid)[0]
            new_medoids[curr_medoids == curr_medoid] = compute_new_medoid(cluster, distances)

        old_medoids[:] = curr_medoids[:]
        curr_medoids[:] = new_medoids[:]
        print 'Mediods still not equal'

    return clusters, curr_medoids

def assign_points_to_clusters(medoids, distances):
    distances_to_medoids = distances[:,medoids]
    clusters = medoids[np.argmin(distances_to_medoids, axis=1)]
    clusters[medoids] = medoids
    return clusters

def compute_new_medoid(cluster, distances):
    mask = np.ones(distances.shape)
    mask[np.ix_(cluster,cluster)] = 0.
    cluster_distances = np.ma.masked_array(data=distances, mask=mask, fill_value=10e9)
    costs = cluster_distances.sum(axis=1)
    return costs.argmin(axis=0, fill_value=10e9)

def DTWDistance(s1,s2,w):
    DTW={}
    w = max(w, abs(len(s1)-len(s2)))

    for i in range(-1,len(s1)):
        for j in range(-1,len(s2)):
            DTW[(i, j)] = float('inf')
    DTW[(-1, -1)] = 0

    for i in range(len(s1)):
        for j in range(max(0, i-w), min(len(s2), i+w)):
            dist= (s1[i]-s2[j])**2
            DTW[(i, j)] = dist + min(DTW[(i-1, j)],DTW[(i, j-1)], DTW[(i-1, j-1)])

    return np.sqrt(DTW[len(s1)-1, len(s2)-1])
 
def LB_Keogh(s1,s2,r):
    '''
    Calculates LB_Keough lower bound to dynamic time warping. Linear
    complexity compared to quadratic complexity of dtw.
    '''
    LB_sum=0
    for ind,i in enumerate(s1):
        
        lower_bound=min(s2[(ind-r if ind-r>=0 else 0):(ind+r)])
        upper_bound=max(s2[(ind-r if ind-r>=0 else 0):(ind+r)])
        
        if i>upper_bound:
            LB_sum=LB_sum+(i-upper_bound)**2
        elif i<lower_bound:
            LB_sum=LB_sum+(i-lower_bound)**2
    
    return np.sqrt(LB_sum)



#Read data from the file
if __name__ == "__main__":
    h5f = h5py.File("Trajectories_training.hdf5",  "r")
    Train_data = h5f["Path_training"][:]
    h5f.close()
    h5f = h5py.File("Labels_training.hdf5",  "r")
    Train_labels = h5f["Labels_training"][:]
    h5f.close()
    

    #Train_data = np.concatenate((Train_data[0:200,:,:],Train_data[500:700,:,:],Train_data[1000:1200,:,:]),axis=0)
    
    #Calculate distances using DTW 
    distances = np.zeros((np.shape(Train_data)[0],np.shape(Train_data)[0]))
    #window size
    w = 10
    for ind,i in enumerate(Train_data):
        for c_ind,j in enumerate(Train_data):
            cur_dist = 0.0
            #Find sum of distances along each dimension
            for z in range(np.shape(Train_data)[2]):
                cur_dist += DTWDistance(i[:,z],j[:,z],w)
            distances[ind,c_ind] = cur_dist
        print('First row completed',ind,c_ind)
    print 'Distances calculated'
    print distances
    clusters, curr_medoids = cluster(distances, 3)
    print 'Mediods are :'
    print curr_medoids
    print 'Cluster assigments : '
    print clusters
    f1 = h5py.File("Clusters.hdf5", "w")
    dset1 = f1.create_dataset("Cluster assignments", data = clusters)
    f1.close()
    f2 = h5py.File("Mediods.hdf5", "w")
    dset2 = f2.create_dataset("Medios", data = curr_medoids)
    f2.close()

