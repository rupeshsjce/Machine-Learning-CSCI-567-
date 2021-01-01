import numpy as np
from knn import KNN

############################################################################
# DO NOT MODIFY CODES ABOVE
############################################################################
import sys

def prior_distance(name):
    if name == 'euclidean':
        return 3
    if name == 'minkowski':
        return 2
    if name == 'cosine_dist':
        return 1
    return -1


def prior_scaler(name):
    if name == 'min_max_scale':
        return 2
    if name == 'normalize':
        return 1
    return -1


def return_Nmatched_tag(real_labels, predicted_labels, tag):
    i = 0
    count = 0
    while(i < len(real_labels)):
        if (real_labels[i] == predicted_labels[i] == tag):
            count = count + 1
        i = i+1
    return count

# TODO: implement F1 score


def f1_score(real_labels, predicted_labels):

    # real_labels = [0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0,
    #                0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0]
    # predicted_labels = [0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1,
    #                     0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1]

    real_labels = [int(i) for i in real_labels]
    #real_labels = real_labels.tolist()
    #print(type(real_labels), type(predicted_labels))
    #print("real_labels: \n", real_labels)
    #print("predicted_labels: \n", predicted_labels)
    """
    Information on F1 score - https://en.wikipedia.org/wiki/F1_score
    :param real_labels: List[int]
    :param predicted_labels: List[int]
    :return: float
    """
    assert len(real_labels) == len(predicted_labels)
    """
    TODO:
    1. Make a confusion Matrix
                Real/Actual class
                   P        N

                P  TP=5     FP=2
    Predcited
                N  FN=3     TN=3

    real_labels =      [1,1,1,1,1,1,1,1,0,0,0,0,0] P_A = 8,  N_A = 5
    predicted_labels = [0,0,0,1,1,1,1,1,0,0,0,1,1]
    TP = ( number of 1's such that real_labels[i]=predicted_labels[i]=1)
    P_A = TP + FN ; FN = P_A - TP
    TN = ( number of 0's such that real_labels[i]=predicted_labels[i]=0)
    N_A = FP + TN ; FP = N_A - TN

    2. Calculate Precision or positive predictive value (PPV) =  [TP / (TP+FP)] and Recall or true positive rate (TPR) = [TP / (TP+FN)].
    3. Take HM of Precision and Recall  to calculate f1_score and return it.
    """
    """
    epsilon = 0.0000001
    P_A = real_labels.count(1)
    N_A = real_labels.count(0)
    #print("Real labels P_A = %d N_A = %d" % (P_A, N_A))

    TP = return_Nmatched_tag(real_labels, predicted_labels, 1)
    #print("TP = %d" % TP)
    FN = P_A - TP

    TN = return_Nmatched_tag(real_labels, predicted_labels, 0)
    #print("TN = %d" % TN)
    FP = N_A - TN

    precision = np.divide(TP, (TP+FP+epsilon))
    recall = np.divide(TP, (TP+FN+epsilon))

    f1Score = np.divide((2*precision*recall), (precision + recall + epsilon))

    return f1Score

    """
    
    try: 
        addup = [x + y for x, y in zip(real_labels, predicted_labels)]
        tp = np.sum(np.equal(addup,2)) # true positive
        if tp == 0:
            return 0
        precision = sum(np.equal(addup,2))/sum(np.equal(predicted_labels,1))
        recall = sum(np.equal(addup,2))/sum(np.equal(real_labels,1))
        if precision == 0 and recall ==0:
            f1_score = 0
        else: 
            f1_score = 2*(precision*recall)/(precision+recall)
        #f1_score = np.nan_to_num(f1_score)
    except: 
        f1_score = 0
    return f1_score

class Distances:
    @ staticmethod
    # TODO
    def minkowski_distance(point1, point2):
        """
        Minkowski distance is the generalized version of Euclidean Distance
        It is also know as L-p norm (where p>=1) that you have studied in class
        For our assignment we need to take p=3
        Information on Minkowski distance - https://en.wikipedia.org/wiki/Minkowski_distance
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
        assert len(point1) == len(point2)
        n = len(point1)
        sum = 0
        for i in range(0, n):
            sum += abs(point1[i]-point2[i]) ** 3
        return sum ** (1./3.)

    @ staticmethod
    # TODO
    def euclidean_distance(point1, point2):
        """
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
        assert len(point1) == len(point2)
        n = len(point1)
        sum = 0
        for i in range(0, n):
            sum += abs(point1[i]-point2[i]) ** 2
        return sum ** (1./2.)

    @ staticmethod
    # TODO
    def cosine_similarity_distance(point1, point2):
        """
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
        l2_point1 = np.linalg.norm(point1)
        l2_point2 = np.linalg.norm(point2)
        if l2_point1 == 0 or l2_point2 == 0:
            return 1
        cos_sim = np.dot(point1, point2) / (l2_point1 * l2_point2)
        return 1 - cos_sim


class HyperparameterTuner:
    def __init__(self):
        self.best_k = None
        self.best_distance_function = None
        self.best_scaler = None
        self.best_model = None

    # TODO: find parameters with the best f1 score on validation dataset
    def tuning_without_scaling(self, distance_funcs, x_train, y_train, x_val, y_val):
        
        total = []
        total.append(distance_funcs)
        total.append(x_train)
        total.append(y_train)
        total.append(x_val)
        total.append(y_val)
        #sys.exit(total)
        #print(type(x_train), type(y_train), type(x_val), type(y_val))
        #x_train = x_train.tolist()
        #y_train = y_train.tolist()
        #x_val = x_val.tolist()
        #y_val = y_val.tolist()
        #print(type(x_train), type(y_train), type(x_val), type(y_val))

        """
        In this part, you need to try different distance functions you implemented in part 1.1 and different values of k (among 1, 3, 5, ... , 29), and find the best model with the highest f1-score on the given validation set.

        :param distance_funcs: dictionary of distance functions (key is the function name, value is the function) you need to try to calculate the distance. Make sure you loop over all distance functions for each k value.
        :param x_train: List[List[int]] training data set to train your KNN model
        :param y_train: List[int] training labels to train your KNN model
        :param x_val:  List[List[int]] validation data
        :param y_val: List[int] validation labels

        Find the best k, distance_function (its name), and model (an instance of KNN) and assign them to self.best_k,
        self.best_distance_function, and self.best_model respectively.
        NOTE: self.best_scaler will be None.

        NOTE: When there is a tie, choose the model based on the following priorities:
        First check the distance function:  euclidean > Minkowski > cosine_dist
                (this will also be the insertion order in "distance_funcs", to make things easier).
        For the same distance function, further break tie by prioritizing a smaller k.
        """

        k = 1
        # (1,'euclidean', 0.6666), (1, 'minkowski', 0.5), ....]

        result_list = []
        while k <= 29 and k <= len(x_train):  # 29
            # Now take one by one distance_funcs
            for dist_func in distance_funcs:
                predicted_labels = []
                # Now I have k value, find best f1_score amongst (euclidean, minkowski, cosine_dist)
                model = KNN(k, distance_funcs[dist_func])
                model.train(x_train, y_train)
                predicted_labels = model.predict(x_val)
                # y_val is nump.ndarray ; convert to list y_val.tolist()
                f1_score_val = f1_score(y_val, predicted_labels)
                print("k = %2d, dist_func = %12s, f1_score_val = %s "
                      % (k, dist_func, f1_score_val))

                result_list.append((k, dist_func, None, f1_score_val, model))

            k = k + 2

        # print(result_list) # List[set(int, string, float)]
        self.best_k, self.best_distance_function, self.scaler, self.best_model = self.get_best_hyperparams(
            result_list, False)

    # TODO: find parameters with the best f1 score on validation dataset, with normalized data

    def tuning_with_scaling(self, distance_funcs, scaling_classes, x_train, y_train, x_val, y_val):
        total = []
        total.append(distance_funcs)
        total.append(x_train)
        total.append(y_train)
        total.append(x_val)
        total.append(y_val)
        #sys.exit(total)
        #x_train = x_train.tolist()
        #y_train = y_train.tolist()
        #x_val = x_val.tolist()
        #y_val = y_val.tolist()
        """
        This part is the same as "tuning_without_scaling", except that you also need to try two different scalers implemented in Part 1.3. More specifically, before passing the training and validation data to KNN model, apply the scalers in scaling_classes to both of them.

        :param distance_funcs: dictionary of distance functions (key is the function name, value is the function) you need to try to calculate the distance. Make sure you loop over all distance functions for each k value.
        :param scaling_classes: dictionary of scalers (key is the scaler name, value is the scaler class) you need to try to normalize your data
        :param x_train: List[List[int]] training data set to train your KNN model
        :param y_train: List[int] train labels to train your KNN model
        :param x_val: List[List[int]] validation data
        :param y_val: List[int] validation labels

        Find the best k, distance_function (its name), scaler (its name), and model (an instance of KNN), and assign them to self.best_k, self.best_distance_function, best_scaler, and self.best_model respectively.

        NOTE: When there is a tie, choose the model based on the following priorities:
        First check scaler, prioritizing "min_max_scale" over "normalize" (which will also be the insertion order of scaling_classes). Then follow the same rule as in "tuning_without_scaling".
        """
        k = 1
        # [(1,'euclidean', 'min_max_scale', 0.6666), (1, 'euclidean', 'normalize', 0.5), ....]
        result_list = []
        while k <= 29 and k <= len(x_train):  # 29
            # Now take one by one distance_funcs
            for dist_func in distance_funcs:
                for scaler in scaling_classes:
                    predicted_labels = []
                    scaling = scaling_classes[scaler]()
                    # scaling(features)
                    # Now I have k value, find best f1_score amongst (euclidean, minkowski, cosine_dist)
                    model = KNN(k, distance_funcs[dist_func])
                    # scaled training data
                    model.train(scaling(x_train), y_train)
                    predicted_labels = model.predict(
                        scaling(x_val))  # scaled validation data
                    f1_score_val = f1_score(y_val, predicted_labels)
                    print("k = %2d, dist_func = %12s, scaler = %15s, f1_score_val = %s "
                          % (k, dist_func, scaler, f1_score_val))

                    result_list.append(
                        (k, dist_func, scaler, f1_score_val, model))

            k = k + 2

        # print(result_list) #List[set(int, string, string, float)]
        self.best_k, self.best_distance_function, self.best_scaler, self.best_model = self.get_best_hyperparams(
            result_list, True)

    def get_best_hyperparams(self, result_list, scale):
        """
         result_list = List[set(int, string, None, float, KNN)] if scale is False
                     = List[set(int, string, string, float, KNN)] if scale is True

         return (k, best_distance_function, best_scaler, best_model)    # scaler is None in case scale is False       
        """
        best_f_val = 0
        best_k = 0
        best_distance_function = " "
        best_scaler = " "

        for k, distance_function, scaler, f_val, model in result_list:
            """
            Get the best f_val and in case of tie 
                scale is False: prioritizing euclidean > Minkowski > cosine_dist ; 
                                if still tie (f_val and distance_function) then winner is with smaller k value
                scale is True:  prioritizing "min_max_scale" over "normalize" ;
                                if still tie then use scale is False condition. 

            """
            if best_f_val < f_val:
                best_f_val = f_val
                best_k = k
                best_distance_function = distance_function
                best_scaler = scaler
                best_model = model
            elif best_f_val == f_val:
                if not scale:
                    if prior_distance(best_distance_function) < prior_distance(distance_function):
                        best_f_val = f_val
                        best_k = k
                        best_distance_function = distance_function
                        best_scaler = scaler
                        best_model = model
                    elif prior_distance(best_distance_function) == prior_distance(distance_function):
                        if k < best_k:
                            best_f_val = f_val
                            best_k = k
                            best_distance_function = distance_function
                            best_scaler = scaler
                            best_model = model
                else:  # when scale is True
                    if prior_scaler(best_scaler) < prior_scaler(scaler):
                        best_f_val = f_val
                        best_k = k
                        best_distance_function = distance_function
                        best_scaler = scaler
                        best_model = model
                    elif prior_scaler(best_scaler) == prior_scaler(scaler):
                        if prior_distance(best_distance_function) < prior_distance(distance_function):
                            best_f_val = f_val
                            best_k = k
                            best_distance_function = distance_function
                            best_scaler = scaler
                            best_model = model
                        elif prior_distance(best_distance_function) == prior_distance(distance_function):
                            if k < best_k:
                                best_f_val = f_val
                                best_k = k
                                best_distance_function = distance_function
                                best_scaler = scaler
                                best_model = model

        return (best_k, best_distance_function, best_scaler, best_model)


class NormalizationScaler:
    def __init__(self):
        pass

    # TODO: normalize data
    def __call__(self, features):
        """
        Normalize features for every sample

        Example
        features = [[3, 4], [1, -1], [0, 0]]
        return [[0.6, 0.8], [0.707107, -0.707107], [0, 0]]

        :param features: List[List[float]]
        :return: List[List[float]]
        """
 
        normalisedScaler = []
        for point in features:
            # print(point)
            l2_d = np.linalg.norm(point)
            if l2_d == 0:
                normalisedScaler.append(point)
            else:
                normalisedScaler.append(np.divide(point, l2_d))

        return normalisedScaler


class MinMaxScaler:
    def __init__(self):
        pass

    # TODO: min-max normalize data
    def __call__(self, features):
        """
                For each feature, normalize it linearly so that its value is between 0 and 1 across all samples.
        For example, if the input features are [[2, -1], [-1, 5], [0, 0]],
        the output should be [[1, 0], [0, 1], [0.333333, 0.16667]].
                This is because: take the first feature for example, which has values 2, -1, and 0 across the three samples.
                The minimum value of this feature is thus min=-1, while the maximum value is max=2.
                So the new feature value for each sample can be computed by: new_value = (old_value - min)/(max-min),
                leading to 1, 0, and 0.333333.
                If max happens to be same as min, set all new values to be zero for this feature.
                (For further reference, see https://en.wikipedia.org/wiki/Feature_scaling.)

        :param features: List[List[float]]
        :return: List[List[float]]
        """
        maxminset = []
        minmaxScalerNormalised = []
        for i in range(len(features[0])):
            flist = [p[i] for p in features]
            fmax = max(flist)
            fmin = min(flist)
            maxminset.append((fmax, fmin))

        for points in features:
            example = []
            for i in range(len(points)):
                mmax = maxminset[i][0]
                mmin = maxminset[i][1]
                if mmax == mmin:
                    num = 0
                else:
                    num = np.divide((points[i] - mmin),
                                    (mmax - mmin))
                example.append(num)
            minmaxScalerNormalised.append(example)
        return minmaxScalerNormalised


# if __name__ == '__main__':
#     real_labels = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
#     predicted_labels = [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1]

#     print("f1_score : ", f1_score(real_labels, predicted_labels))
