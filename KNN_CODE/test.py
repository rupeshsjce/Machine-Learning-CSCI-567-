from data import data_processing
from utils import Distances, HyperparameterTuner, NormalizationScaler, MinMaxScaler


def main():
    distance_funcs = {
        'euclidean': Distances.euclidean_distance,
        'minkowski': Distances.minkowski_distance,
        'cosine_dist': Distances.cosine_similarity_distance,
    }

    scaling_classes = {
        'min_max_scale': MinMaxScaler,
        'normalize': NormalizationScaler,
    }

    x_train, y_train, x_val, y_val, x_test, y_test = data_processing()

    print('x_train shape = ', x_train.shape)
    print('y_train shape = ', y_train.shape)

    tuner_without_scaling_obj = HyperparameterTuner()
    
    x_train = [[1, 1, 3, 2, 0, 0, 1], [0, 0, 2, 2, 0, 1, 0], [2, 0, 3, 1, 0, 2, 2], [0, 0, 2, 2, 1, 0, 0], [0, 1, 2, 1, 1, 1, 1], [0, 0, 1, 2, 1, 2, 0], [2, 0, 2, 2, 2, 0, 0], [0, 0, 2, 1, 2, 1, 2], [0, 1, 2, 2, 2, 2, 0], [2, 0, 1, 4, 0, 0, 2], [0, 0, 2, 4, 0, 1, 0], [0, 0, 2, 4, 0, 2, 2], [0, 1, 2, 3, 1, 0, 0], [3, 0, 2, 4, 1, 1, 0], [0, 0, 1, 4, 1, 2, 2], [0, 0, 2, 3, 2, 0, 0], [2, 1, 2, 4, 2, 1, 0], [0, 0, 1, 4, 2, 2, 1], [0, 0, 2, 5, 0, 1, 2], [3, 0, 2, 5, 0, 1, 0], [0, 0, 1, 5, 0, 2, 0], [3, 1, 2, 5, 1, 0, 2], [2, 0, 2, 3, 1, 1, 0], [0, 0, 3, 5, 1, 2, 0], [0, 0, 2, 5, 2, 0, 0], [3, 2, 2, 5, 2, 1, 2], [0, 0, 2, 5, 2, 2, 0], [0, 1, 3, 2, 0, 0, 0], [0, 0, 3, 2, 0, 1, 0], [0, 0, 3, 2, 0, 2, 2], [0, 1, 3, 2, 1, 0, 0], [0, 0, 1, 2, 1, 1, 0], [0, 0, 3, 2, 1, 2, 0], [0, 1, 3, 2, 2, 0, 2], [1, 0, 1, 2, 2, 1, 0], [0, 0, 1, 2, 2, 2, 0], [0, 0, 3, 4, 0, 0, 0], [0, 1, 1, 4, 0, 1, 2], [0, 0, 3, 4, 0, 2, 0], [1, 1, 1, 4, 1, 0, 0]]
    y_train = [1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1]
    x_val = [[1, 0, 2, 4, 2, 0, 1], [1, 1, 3, 3, 0, 0, 1], [0, 0, 3, 4, 2, 0, 0], [0, 0, 3, 4, 2, 2, 1], [0, 1, 4, 4, 0, 0, 0]]
    y_val = [0, 0, 0, 0, 1]
    
    tuner_without_scaling_obj.tuning_without_scaling(distance_funcs, x_train, y_train, x_val, y_val)

    print("**Without Scaling**")
    print("k =", tuner_without_scaling_obj.best_k)
    print("distance function =", tuner_without_scaling_obj.best_distance_function)
     
    tuner_with_scaling_obj = HyperparameterTuner()
    tuner_with_scaling_obj.tuning_with_scaling(distance_funcs, scaling_classes, x_train, y_train, x_val, y_val)

    print("\n**With Scaling**")
    print("k =", tuner_with_scaling_obj.best_k)
    print("distance function =", tuner_with_scaling_obj.best_distance_function)
    print("scaler =", tuner_with_scaling_obj.best_scaler)
    

if __name__ == '__main__':
    main()


