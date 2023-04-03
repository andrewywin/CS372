from collections import Counter
import numpy as np
class MyDecisionTreeClassifier:

    def __init__(self):
        """
        One typically initializes shared class variables and data structures in the constructor.

        Variables which you wish to modify in train(X, y) and then utilize again in predict(X)
        should be explicitly initialized here (even only as self.my_variable = None).
        """
        self.tree = None
        
    def fit(self, X, y):
        """
        This is the method which will be called to train the model. We can assume that train will
        only be called one time for the purposes of this project.

        :param X: The samples and features which will be used for training. The data should have
        the shape:
        X =\
        [
         [feature_1_value, feature_2_value, feature_3_value, ..., feature_m_value],  # Sample a
         [feature_1_value, feature_2_value, feature_3_value, ..., feature_m_value],  # Sample b
         ...
         [feature_1_value, feature_2_value, feature_3_value, ..., feature_m_value]  # Sample n
        ]
        :param y: The target/response variable used for training. The data should have the shape:
        y = [target_for_sample_a, target_for_sample_b, ..., target_for_sample_n]

        :return: self Think of this method not having a return statement at all. The idea to
        "return self" is a convention of scikit learn; the underlying model should have some
        internally saved trained state.
        """
        self.tree = self.build_tree(X, y)
        return self
    def predict(self, X):
        """
        This is the method which will be used to predict the output targets/responses of a given
        list of samples.

        It should rely on mechanisms saved after train(X, y) was called.
        You can assume that train(X, y) has already been called before this method is invoked for
        the purposes of this project.

        :param X: The samples and features which will be used for prediction. The data should have
        the shape:
        X =\
        [
         [feature_1_value, feature_2_value, feature_3_value, ..., feature_m_value],  # Sample a
         [feature_1_value, feature_2_value, feature_3_value, ..., feature_m_value],  # Sample b
         ...
         [feature_1_value, feature_2_value, feature_3_value, ..., feature_m_value]  # Sample n
        ]
        :return: The target/response variables the model decides is optimal for the given samples.
        The data should have the shape:
        y = [prediction_for_sample_a, prediction_for_sample_b, ..., prediction_for_sample_n]
        """
        return [self.predict_instance(x, self.tree) for x in X]

    def entropy(self, y):
        '''
        calculates entropy of set
        '''
        freq = Counter(y)
        probs = [freq[c] / len(y) for c in set(y)]
        return -sum(p * np.log2(p) for p in probs)
    
    def split(self, X, y, feature, threshold):
        '''
        splits group based on threshold
        '''
        left_X, left_y, right_X, right_y = [], [], [], []
        for i in range(len(y)):
            if X[i][feature] < threshold:
                left_X.append(X[i])
                left_y.append(y[i])
            else:
                right_X.append(X[i])
                right_y.append(y[i])
        return np.array(left_X), np.array(left_y), np.array(right_X), np.array(right_y)
    
    def best_split(self, X, y):
        '''
        function to get the max information gain
        '''
        best_feature, best_threshold, best_info_gain = None, None, -1
        n_features = X.shape[1]
        for feature in range(n_features):
            feature_values = X[:, feature]
            thresholds = np.unique(feature_values)
            for threshold in thresholds:
                left_X, left_y, right_X, right_y = self.split(X, y, feature, threshold)
                if len(left_y) > 0 and len(right_y) > 0:
                    info_gain = self.entropy(y) - (len(left_y) / len(y) * self.entropy(left_y) + len(right_y) / len(y) * self.entropy(right_y))
                    if info_gain > best_info_gain:
                        best_feature = feature
                        best_threshold = threshold
                        best_info_gain = info_gain
        return best_feature, best_threshold
    
    def build_tree(self, X, y):
        '''
        build the decision tree
        '''
        if len(set(y)) == 1:
            return y[0]
        if X.shape[1] == 0:
            return Counter(y).most_common(1)[0][0]
        best_feature, best_threshold = self.best_split(X, y)
        if best_feature is None or best_threshold is None:
            return Counter(y).most_common(1)[0][0]
        left_X, left_y, right_X, right_y = self.split(X, y, best_feature, best_threshold)
        return {'feature': best_feature,
                'threshold': best_threshold,
                'left': self.build_tree(left_X, left_y),
                'right': self.build_tree(right_X, right_y)}
    
    def predict_instance(self, x, tree):
        '''
        recursion to get prediction of left and right side
        '''
        if isinstance(tree, dict):
            if x[tree['feature']] < tree['threshold']:
                return self.predict_instance(x, tree['left'])
            else:
                return self.predict_instance(x, tree['right'])
        else:
            return tree

