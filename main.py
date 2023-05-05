# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# k means from machine learning
def kmeans(X, k):

    m, d = X.shape
    curr_centroids = np.zeros(m)

    # initialize centers randomly
    random_indices = np.random.permutation(m)
    centers = X[random_indices[:k], :]

    # To know when converged
    converged = False
    prev_centroids = None

    distances_list = np.full(k, 100000.0)

    while not converged:
        # calculates cluster according to centers
        for i in range(len(X)):
            x = X[i]
            # calculate the distance of example xi from each center
            for j in range(k):
                distances_list[j] = np.linalg.norm(x - centers[j])

            curr_centroids[i] = np.argmin(distances_list)

        # checks if centroids didn't change
        if np.array_equal(prev_centroids, curr_centroids):
            converged = True
        else:
            prev_centroids = curr_centroids

        # calculates centers according to clusters
        for j in range(k):
            # the index of the examples that are in cluster j
            cluster_indices = np.where(curr_centroids == j)[0]
            # the examples in cluster j
            values = X[cluster_indices, :]
            centers[j] = np.mean(values, axis=0)

    return curr_centroids


def dc_dp_means(X, l):

    m, d = X.shape
    k = 1

    centroids = np.full((1, d), np.mean(X, axis=0), dtype=float)  # check if it's what we want
    labels = np.ones((m, 1), dtype=int)

    converged = False
    prev_centroids = None

    ite = 0

    while not converged and ite < 15:
        ite += 1
        print("Iteration ", ite)
        j_max = -1
        d_max = -1

        distances_list = np.full((len(centroids), 1), 10000, dtype=float)

        # calculates cluster according to centers
        for i in range(m):
            x = X[i]

            # calculate the distance of example xi from each center
            for j in range(len(centroids)):
                distances_list[j] = np.linalg.norm(x - centroids[j])
                # print("distances_list is: ", distances_list[j])

            # choose min distance centroid for example xi
            labels[i] = np.argmin(distances_list) + 1  # We start numbering clusters from 1 and not 0
            # print("labels is: ", labels[i])

            closest_centroid_distance = np.linalg.norm(x - centroids[labels[i] - 1])
            if closest_centroid_distance > d_max:
                j_max = i
                d_max = closest_centroid_distance

        if d_max > l:
            k = k + 1
            centroids = np.append(centroids, np.array([X[j_max]]), axis=0)
            labels[j_max] = k
            # print("ADDED CLUSTER, K = ", k)

        # checks if centroids didn't change
        if np.array_equal(prev_centroids, centroids):
            converged = True
        else:
            prev_centroids = centroids

        # calculates centers according to clusters
        for j in range(k):
            # the index of the examples that are in cluster j
            cluster_labels_j = np.where(labels == j + 1)[0]
            # the examples in cluster j
            values = X[cluster_labels_j, :]
            centroids[j] = np.mean(values, axis=0)

    # print("k = ", k)
    # print("lambda = ", l)
    return labels


def test_q2():

    # Number of points per Gaussian distribution
    n = 200

    # Generate random means for each distribution
    means = np.random.rand(5, 2) * 10

    # Generate random covariance matrices for each distribution
    covs = np.tile(np.identity(2), (5, 1, 1))

    # Sample points from each distribution
    data = np.vstack([np.random.multivariate_normal(means[i], covs[i], n) for i in range(5)])

    # Shuffle the data
    np.random.shuffle(data)

    # choosing optimal k for kmeans
    k_params = [1, 3, 5, 7, 31, 1000]
    for k in k_params:
        plt.scatter(data[:, 0], data[:, 1], c=kmeans(data, k), s=10)
        plt.title("K Means with k = " + str(k))
        plt.show()

    # choosing optimal lambda for dc-dp means
    l_params = [1, 3, 5, 7, 31, 1000]
    for l in l_params:
        plt.scatter(data[:, 0], data[:, 1], c=dc_dp_means(data, l), s=10)
        plt.title("DC-DP Means With lambda = " + str(l))
        plt.show()


def cluster_image_kmeans(k):
    # load image
    img = Image.open('mandrill_.jpg')
    img_data = np.array(img)

    # reshape image data to a 2D array of RGB values
    m, n, _ = img_data.shape
    X = img_data.reshape(m * n, 3)

    # apply K-means clustering
    labels = kmeans(X, k)

    # replace each pixel's color with the value of the centroid of the cluster that pixel was assigned to
    clustered_data = np.zeros_like(X)
    centers = np.zeros((k, 3))
    for i in range(k):
        cluster_indices = np.where(labels == i)[0]
        cluster_data = X[cluster_indices]
        centers[i] = np.mean(cluster_data, axis=0)
        clustered_data[cluster_indices] = centers[i]

    # reshape clustered data back to the original image shape
    clustered_data = clustered_data.reshape(m, n, 3)

    # create and save the clustered image
    clustered_img = Image.fromarray(np.uint8(clustered_data))

    # display original image and clustered image side by side
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(img)
    ax[0].set_title('Original Image')
    ax[1].imshow(clustered_img)
    ax[1].set_title(f'K-Means Clustered Image (k={k})')
    plt.show()

    return clustered_img


def cluster_image_dcdp_means(l):
    # load image
    img = Image.open('mandrill_.jpg')
    img_data = np.array(img)
    # reshape image data to a 2D array of RGB values
    m, n, _ = img_data.shape
    X = img_data.reshape(m * n, 3)
    print("Before labels")
    # apply K-means clustering
    labels = dc_dp_means(X, l)
    print("After labels")
    # replace each pixel's color with the value of the centroid of the cluster that pixel was assigned to
    clustered_data = np.zeros_like(X)
    centers = np.zeros((l, 3))
    for i in range(l):
        cluster_indices = np.where(labels == i)[0]
        cluster_data = X[cluster_indices]
        centers[i] = np.mean(cluster_data, axis=0)
        clustered_data[cluster_indices] = centers[i]

    # reshape clustered data back to the original image shape
    clustered_data = clustered_data.reshape(m, n, 3)

    # create and save the clustered image
    clustered_img = Image.fromarray(np.uint8(clustered_data))

    # display original image and clustered image side by side
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(img)
    ax[0].set_title('Original Image')
    ax[1].imshow(clustered_img)
    ax[1].set_title(f'DCDP-Means Clustered Image (lambda={l})')
    plt.show()

    return clustered_img


def test_q3():
    for k in [3, 5, 7, 13, 31, 100]:
        cluster_image_kmeans(k)

    for l in [3, 5, 7, 13, 31, 100]:
        print(l)
        cluster_image_dcdp_means(l)


if __name__ == '__main__':
    test_q2()
    test_q3()