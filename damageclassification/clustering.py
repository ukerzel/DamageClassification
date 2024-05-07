"""
    Before we can identify damage sites, we need to look for suitable regions in the image.
    Typically, damage sites appear as dark regions in the image. Instead of simple thresholding, we use 
    a clustering approach to identify regions that belong together and form damage site candidates.
"""

import numpy as np


import scipy.ndimage as ndi
from scipy.spatial import KDTree

from sklearn.cluster import DBSCAN


def get_centroids(image : np.ndarray, image_threshold = 20,
                  eps=1, min_samples=5, metric='euclidean',
                  min_size = 20, fill_holes = False,
                  filter_close_centroids = False, filter_radius = 50) -> list:
    """    Determine centroids of clusters corresponding to potential damage sites.
    In a first step, a threshold is applied to the input image to identify areas of potential damage sites.
    Using DBSCAN, these agglomerations of pixels are fitted into clusters. Then, the mean x/y values are determined
    from pixels belonging to one cluster. If the number of pixels in a given cluster excees the threshold given by min_size, this cluster is added
    to the list of (x,y) coordinates that is returned as the final list potential damage sites.

    Sometimes, clusters may be found in very close proximity to each other, we can reject those to avoid 
    classifying the same event multiple times (which may distort our statistics).

    DBScan documentation: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html

    Args:
        image (np.ndarray): Input SEM image 
        image_threshold (int, optional): Threshold to be applied to the image to identify candidates for damage sites. Defaults to 20.
        eps (int, optional): parameter eps of DBSCAN: The maximum distance between two samples for one to be considered as in the neighborhood of the other. Defaults to 1.
        min_samples (int, optional): parameter min_samples of DBSCAN: The number of samples (or total weight) in a neighborhood for a point to be considered as a core point. Defaults to 5.
        metric (str, optional): parameter metric of DBSCAN. Defaults to 'euclidean'.
        min_size (int, optional): Minimum number of pixels in a cluster for the damage site candidate to be considered in the final list. Defaults to 20.
        fill_holes (bool, optional): Fill small holes in damage sites clusters using binary_fill_holes. Defaults to False.
        filter_close_centroids (book optional): Filter cluster centroids within a given radius. Defaults to False
        filter_radius (float, optional): Radius within which centroids are considered to be the same. Defaults to 50

    Returns:
        list: list of (x,y) coordinates of the centroids of the clusters of accepted damage site candidates.
    """


    centroids = []

    # apply the threshold to identify regions of "dark" pixels
    # the result is a binary mask (true/false) whether a given pixel is above or below the threshold
    cluster_candidates = image < image_threshold

    # sometimes the clusters have small holes in them, for example, individual pixels
    # inside a region below the threshold. This may confuse the clustering algorith later on
    # and we can use the following to fill these holes
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.binary_fill_holes.html
    # N.B. the algorith only works on binay data
    if fill_holes:
        cluster_candidates = ndi.binary_fill_holes(cluster_candidates)

    # apply the treshold to the image to identify regions of "dark" pixels
    #cluster_candidates = np.asarray(image < image_threshold).nonzero()

    # transform image format into a numpy array to pass on to DBSCAN clustering
    cluster_candidates = np.asarray(cluster_candidates).nonzero()
    cluster_candidates = np.transpose(cluster_candidates)


    # run the DBSCAN clustering algorithm, candidate sites that are not attributed to a cluster are labelled as "-1", i.e. "noise"
    # (e.g. they are too small, etc)
    # For the remaining pixels, a label is assigned to each pixel, indicating to which cluster (or noise) they belong to.
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')

    dbscan.fit(cluster_candidates)

    labels = dbscan.labels_
    # Number of clusters in labels, ignoring noise if present.
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    print('# clusters {}, #noise {}'.format(n_clusters, n_noise))


    # now loop over all labels found by DBSCAN, i.e. all identified clusters and the noise
    # we use "set" here, as the labels are attributed to individual pixels, i.e. they appear as often as we have pixels
    # in the cluster candidates
    for i in set(labels):
        if i>-1:
            # all points belonging to a given cluster
            cluster_points = cluster_candidates[labels==i, :]
            if len(cluster_points) > min_size:
                x_mean=np.mean(cluster_points, axis=0)[0]
                y_mean=np.mean(cluster_points, axis=0)[1]
                centroids.append([x_mean,y_mean])

    if filter_close_centroids:
        proximity_tree = KDTree(centroids)
        pairs = proximity_tree.query_pairs(filter_radius)
        for p in pairs:
            item = centroids.pop(p[0])

    return centroids