import numpy as np

#################################
# DO NOT IMPORT OHTER LIBRARIES
#################################


def get_k_means_plus_plus_center_indices(n, n_cluster, x, generator=np.random):
    '''

    :param n: number of samples in the data
    :param n_cluster: the number of cluster centers required
    :param x: data - numpy array of points
    :param generator: random number generator. Use it in the same way as np.random.
            In grading, to obtain deterministic results, we will be using our own random number generator.


    :return: the center points array of length n_clusters with each entry being the *index* of a sample
             chosen as centroid.
    '''
    ###############################################
    # TODO: implement the Kmeans++ initialization
    ###############################################
    def random_distr(l):
        r = generator.rand()
        #print("r :", r)
        s = 0
        index = 0
        for prob in l:
            s += prob
            if s >= r:
                return index
            index = index + 1
        return index

    centers = []
    c1 = generator.randint(0, n)
    centers.append(c1)
    # print("index : ", c1)

    for k in range(1, n_cluster):
        # L2 distance square
        # distances of all points from center c1
        dist = np.linalg.norm(x-x[centers[0]], axis=1) ** 2
        centers_recorded = k
        while k > 1 and centers_recorded > 1:
            dist2 = np.linalg.norm(
                x-x[centers[centers_recorded - 1]], axis=1) ** 2
            dist = np.minimum(dist, dist2)
            centers_recorded = centers_recorded - 1

        # Normalized for probabilty distribution
        normalised_dist = dist/np.sum(dist)
        index = random_distr(normalised_dist)
        # print("index : ", index)
        centers.append(index)

    # DO NOT CHANGE CODE BELOW THIS LINE
    return centers

# Vanilla initialization method for KMeans


def get_lloyd_k_means(n, n_cluster, x, generator):
    return generator.choice(n, size=n_cluster)


class KMeans():

    '''
        Class KMeans:
        Attr:
            n_cluster - Number of clusters for kmeans clustering (Int)
            max_iter - maximum updates for kmeans clustering (Int)
            e - error tolerance (Float)
    '''

    def __init__(self, n_cluster, max_iter=100, e=0.0001, generator=np.random):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e
        self.generator = generator

    def fit(self, x, centroid_func=get_lloyd_k_means):
        '''
            Finds n_cluster in the data x
            params:
                x - N X D numpy array
            returns:
                A tuple in the following order:
                  - final centroids, a n_cluster X D numpy array,  #(4,2)
                  - a length (N,) numpy array where cell i is the ith sample's assigned cluster's index (start from 0), #(200,)
                  - number of times you update the assignment, an Int (at most self.max_iter) #30
        '''
        assert len(x.shape) == 2, "fit function takes 2-D numpy arrays as input"
        self.generator.seed(42)
        N, D = x.shape

        self.centers = centroid_func(len(x), self.n_cluster, x, self.generator)
        ###################################################################
        # TODO: Update means and membership until convergence
        #   (i.e., average K-mean objective changes less than self.e)
        #   or until you have made self.max_iter updates.
        ###################################################################

        def kmeans_objective(mu, x, r):
            N = x.shape[0]
            return np.sum([np.sum((x[r == k] - mu[k]) ** 2) for k in range(self.n_cluster)]) / N

        mu = x[self.centers]  # initial centers
        r = np.zeros(N, dtype=int)
        J = kmeans_objective(mu, x, r)
        # Loop until convergence/max_iter
        iter = 0
        while iter < self.max_iter:
            # x = x = x[0:10]
            # Compute membership
            l2 = np.sum(((x - np.expand_dims(mu, axis=1)) ** 2), axis=2)
            r = np.argmin(l2, axis=0)
            # Compute kmeans_objective
            J_new = kmeans_objective(mu, x, r)
            if np.absolute(J - J_new) <= self.e:
                break
            J = J_new
            # Compute means
            mu_new = np.array([np.mean(x[r == k], axis=0)
                               for k in range(self.n_cluster)])
            # https://www.youtube.com/watch?v=iAOL2K9yR1g
            index = np.where(np.isnan(mu_new))

            mu_new[index] = mu[index]
            mu = mu_new
            iter += 1
        return (mu, r, iter)


class KMeansClassifier():

    '''
        Class KMeansClassifier:
        Attr:
            n_cluster - Number of clusters for kmeans clustering (Int)
            max_iter - maximum updates for kmeans clustering (Int)
            e - error tolerance (Float)
    '''

    def __init__(self, n_cluster, max_iter=100, e=1e-6, generator=np.random):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e
        self.generator = generator

    def fit(self, x, y, centroid_func=get_lloyd_k_means):
        '''
            Train the classifier
            params:
                x - N X D size  numpy array
                y - (N,) size numpy array of labels
            returns:
                None
            Store following attributes:
                self.centroids : centroids obtained by kmeans clustering (n_cluster X D numpy array)
                self.centroid_labels : labels of each centroid obtained by
                    majority voting (numpy array of length n_cluster)
        '''

        assert len(x.shape) == 2, "x should be a 2-D numpy array"
        assert len(y.shape) == 1, "y should be a 1-D numpy array"
        assert y.shape[0] == x.shape[0], "y and x should have same rows"

        self.generator.seed(42)
        N, D = x.shape
        ################################################################
        # TODO:
        # - assign means to centroids (use KMeans class you implemented,
        #      and "fit" with the given "centroid_func" function)
        # - assign labels to centroid_labels
        ################################################################
        centroids, membership, i = KMeans(
            n_cluster=self.n_cluster, max_iter=self.max_iter, e=self.e).fit(x, centroid_func)
        temp_centroid_labels = np.zeros((self.n_cluster, self.n_cluster))
        for i in range(0, len(y)):
            temp_centroid_labels[membership[i]][y[i]] += 1

        centroid_labels = np.zeros((self.n_cluster))
        for k in range(0, self.n_cluster):
            centroid_labels[k] = np.argmax(temp_centroid_labels[k])

        centroid_labels = np.asarray(centroid_labels)

        # DO NOT CHANGE CODE BELOW THIS LINE
        self.centroid_labels = centroid_labels
        self.centroids = centroids

        assert self.centroid_labels.shape == (
            self.n_cluster,), 'centroid_labels should be a numpy array of shape ({},)'.format(self.n_cluster)

        assert self.centroids.shape == (
            self.n_cluster, D), 'centroid should be a numpy array of shape {} X {}'.format(self.n_cluster, D)

    def predict(self, x):
        '''
            Predict function
            params:
                x - N X D size  numpy array
            returns:
                predicted labels - numpy array of size (N,)
        '''

        assert len(x.shape) == 2, "x should be a 2-D numpy array"

        self.generator.seed(42)
        N, D = x.shape
        ##########################################################################
        # TODO:
        # - for each example in x, predict its label using 1-NN on the stored
        #    dataset (self.centroids, self.centroid_labels)
        ##########################################################################
        predicted_labels = np.zeros((N,))
        for i in range(0, len(x)):
            index = np.argmin(np.linalg.norm(self.centroids-x[i], axis=1) ** 2)
            predicted_labels[i] = self.centroid_labels[index]
        return predicted_labels


def transform_image(image, code_vectors):
    '''
        Quantize image using the code_vectors (aka centroids)

        Return a new image by replacing each RGB value in image with the nearest code vector
          (nearest in euclidean distance sense)

        returns:
            numpy array of shape image.shape
    '''

    assert image.shape[2] == 3 and len(image.shape) == 3, \
        'Image should be a 3-D array with size (?,?,3)'

    assert code_vectors.shape[1] == 3 and len(code_vectors.shape) == 2, \
        'code_vectors should be a 2-D array with size (?,3)'
    ##############################################################################
    # TODO
    # - replace each pixel (a 3-dimensional point) by its nearest code vector
    ##############################################################################
    data = image.reshape(image.shape[0] * image.shape[1], image.shape[2])
    l2 = np.sum(((data - np.expand_dims(code_vectors, axis=1)) ** 2), axis=2)
    r = np.argmin(l2, axis=0)
    return code_vectors[r].reshape(image.shape[0], image.shape[1], image.shape[2])
