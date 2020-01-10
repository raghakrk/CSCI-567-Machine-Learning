import numpy as np
from knn import KNN

############################################################################
# DO NOT MODIFY ABOVE CODES
############################################################################


# TODO: implement F1 score
def f1_score(real_labels, predicted_labels):
    """
    Information on F1 score - https://en.wikipedia.org/wiki/F1_score
    :param real_labels: List[int]
    :param predicted_labels: List[int]
    :return: float
    """   
    F=(2*np.dot(np.asarray(real_labels).T,np.asarray(predicted_labels)))/(np.sum(real_labels)+np.sum(predicted_labels))
    return F 
    raise NotImplementedError


class Distances:
    @staticmethod
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
        p=3
        csum=0
        for i in range(len(point1)):
            csum=csum+pow(abs(point1[i]-point2[i]),p)
        csum=pow(csum,1/p)
        return csum
        raise NotImplementedError

    @staticmethod
    # TODO
    def euclidean_distance(point1, point2):
        """
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
        csum=0
        csum= np.sqrt(np.dot((np.asarray(point1)-np.asarray(point2)).T,(np.asarray(point1)-np.asarray(point2))))
        return csum
        raise NotImplementedError

    @staticmethod
    # TODO
    def inner_product_distance(point1, point2):
        """
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
        csum=np.dot(np.asarray(point1).T,np.asarray(point2))
        return csum
        raise NotImplementedError

    @staticmethod
    # TODO
    def cosine_similarity_distance(point1, point2):
        """
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
        csum=np.dot(np.asarray(point1).T,np.asarray(point2))/(np.linalg.norm(point1)*np.linalg.norm(point2))
        return 1-csum
        raise NotImplementedError

    @staticmethod
    # TODO        
    def gaussian_kernel_distance(point1, point2):
        """
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
#        csum=0
#        for i,j in zip(point1,point2):
#            csum=csum+np.power((i-j),2) 
        csum=np.dot((np.asarray(point1)-np.asarray(point2)).T,(np.asarray(point1)-np.asarray(point2)))
        csum=-1*np.exp((-1/2)*csum)
        return (csum)
        raise NotImplementedError


class HyperparameterTuner:
    def __init__(self):
        self.best_k = None
        self.best_distance_function = None
        self.best_scaler = None
        self.best_model = None

    # TODO: find parameters with the best f1 score on validation dataset
    def tuning_without_scaling(self, distance_funcs, x_train, y_train, x_val, y_val):
        """
        In this part, you should try different distance function you implemented in part 1.1, and find the best k.
        Use k range from 1 to 30 and increment by 2. Use f1-score to compare different models.

        :param distance_funcs: dictionary of distance functions you must use to calculate the distance.
            Make sure you loop over all distance functions for each data point and each k value.
            You can refer to test.py file to see the format in which these functions will be
            passed by the grading script
        :param x_train: List[List[int]] training data set to train your KNN model
        :param y_train: List[int] train labels to train your KNN model
        :param x_val:  List[List[int]] Validation data set will be used on your KNN predict function to produce
            predicted labels and tune k and distance function.
        :param y_val: List[int] validation labels

        Find(tune) best k, distance_function and model (an instance of KNN) and assign to self.best_k,
        self.best_distance_function and self.best_model respectively.
        NOTE: self.best_scaler will be None

        NOTE: When there is a tie, choose model based on the following priorities:
        Then check distance function  [euclidean > minkowski > gaussian > inner_prod > cosine_dist]
        If they have same distance fuction, choose model which has a less k.
        """
        
        # You need to assign the final values to these variables
        self.best_k = None
        self.best_distance_function = None
        self.best_model = None
        best_fval=-10000
        for k in range (1,30,2):
            for i in range(0,5):
                dist=list(distance_funcs.values())[i]
                model=KNN(k,dist)
                model.train(x_train,y_train)
                ypred=model.predict(x_val)
                fval=f1_score(y_val,ypred)
                if best_fval<fval:
                    best_fval=fval
                    kk=k
                    dd=i
                    md=model
#                if best_fval==fval:
#                    if dd<i:
#                        dd=i
#                        kk=k
#                        best_fval=fval
#                        md=model
#                    if dd==i:
#                        if k<kk:
#                            kk=k
#                            best_fval=fval
#                            md=model
                        
                        
        self.best_distance_function=list(distance_funcs.keys())[dd]
        self.best_k=kk
        self.best_model=md
        #return best_model
       # raise NotImplementedError

    # TODO: find parameters with the best f1 score on validation dataset, with normalized data
    def tuning_with_scaling(self, distance_funcs, scaling_classes, x_train, y_train, x_val, y_val):
        """
        This part is similar to Part 1.3 except that before passing your training and validation data to KNN model to
        tune k and disrance function, you need to create the normalized data using these two scalers to transform your
        data, both training and validation. Again, we will use f1-score to compare different models.
        Here we have 3 hyperparameters i.e. k, distance_function and scaler.

        :param distance_funcs: dictionary of distance funtions you use to calculate the distance. Make sure you
            loop over all distance function for each data point and each k value.
            You can refer to test.py file to see the format in which these functions will be
            passed by the grading script
        :param scaling_classes: dictionary of scalers you will use to normalized your data.
        Refer to test.py file to check the format.
        :param x_train: List[List[int]] training data set to train your KNN model
        :param y_train: List[int] train labels to train your KNN model
        :param x_val: List[List[int]] validation data set you will use on your KNN predict function to produce predicted
            labels and tune your k, distance function and scaler.
        :param y_val: List[int] validation labels

        Find(tune) best k, distance_funtion, scaler and model (an instance of KNN) and assign to self.best_k,
        self.best_distance_function, self.best_scaler and self.best_model respectively

        NOTE: When there is a tie, choose model based on the following priorities:
        For normalization, [min_max_scale > normalize];
        Then check distance function  [euclidean > minkowski > gaussian > inner_prod > cosine_dist]
        If they have same distance function, choose model which has a less k.
        """
        
        # You need to assign the final values to these variables
        self.best_k = None
        self.best_distance_function = None
        self.best_scaler = None
        self.best_model = None
        
        best_fval=-10000
        for f in range(0,2):
            scaler=list(scaling_classes.values())[f]()
            xtrain_n=scaler(x_train)
            xval_n=scaler(x_val)
            for k in range (1,30,2):
                for i in range(0,5):
                    dist=list(distance_funcs.values())[i]
                    model=KNN(k,dist)
                    model.train(xtrain_n,y_train)
                    ypred=model.predict(xval_n)
                    fval=f1_score(y_val,ypred)
                    #print(fval)
                    if best_fval<fval:
                        best_fval=fval
                        kk=k
                        dd=i
                        ss=f
                        md=model
#                    if best_fval==fval:
#                        if f<ss:
#                            ss=f
#                            kk=k
#                            dd=i
#                            md=model
#                            best_fval=fval
#                        if f==ss:
#                            if dd<i:
#                                dd=i
#                                ss=f
#                                kk=k
#                                md=model
#                            if dd==i:
#                                if k<kk:
#                                    kk=k
#                                    dd=i
#                                    ss=f
#                                    md=model
#                                    best_fval=fval
                
        
        self.best_distance_function=list(distance_funcs.keys())[dd]
        self.best_k=kk
        self.best_scaler = list(scaling_classes.keys())[ss]
        self.best_model=md        
        #raise NotImplementedError


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
        fn=[]
        for i in features:
            if np.linalg.norm(i)!=0:
                fn.append(list(i/np.linalg.norm(i)))
            else:
                fn.append(i)
        return np.asarray(fn)
        raise NotImplementedError


class MinMaxScaler:
    """
    Please follow this link to know more about min max scaling
    https://en.wikipedia.org/wiki/Feature_scaling
    You should keep some states inside the object.
    You can assume that the parameter of the first __call__
    will be the training set.

    Hints:
        1. Use a variable to check for first __call__ and only compute
            and store min/max in that case.

    Note:
        1. You may assume the parameters are valid when __call__
            is being called the first time (you can find min and max).

    Example:
        train_features = [[0, 10], [2, 0]]
        test_features = [[20, 1]]

        scaler1 = MinMaxScale()
        train_features_scaled = scaler1(train_features)
        # train_features_scaled should be equal to [[0, 1], [1, 0]]

        test_features_scaled = scaler1(test_features)
        # test_features_scaled should be equal to [[10, 0.1]]

        new_scaler = MinMaxScale() # creating a new scaler
        _ = new_scaler([[1, 1], [0, 0]]) # new trainfeatures
        test_features_scaled = new_scaler(test_features)
        # now test_features_scaled should be [[20, 1]]

    """
  
    def __init__(self):
        self.xmin=[]
        self.xmax=[]
#        pass
    def __call__(self, features):
        """
        normalize the feature vector for each sample . For example,
        if the input features = [[2, -1], [-1, 5], [0, 0]],
        the output should be [[1, 0], [0, 1], [0.333333, 0.16667]]

        :param features: List[List[float]]
        :return: List[List[float]]
        """

        if  self.xmax == [] and self.xmax == []:
            self.xmin=np.min(features,axis=0)
            self.xmax=np.max(features,axis=0)
        minmax=np.array(features)
        for i in range(len(features)):
            for j in range(len(features[0])):
                if self.xmax[j] !=  self.xmin[j]:
                    minmax[i][j] = minmax[i][j] - self.xmin[j]
                    minmax[i][j] = minmax[i][j] / (self.xmax[j] - self.xmin[j])
                else:
                    minmax[i][j] = minmax[i][j] - self.xmin[j]
                    minmax[i][j] = minmax[i][j] / len(features[0])
        
        return minmax.tolist()
        raise NotImplementedError
