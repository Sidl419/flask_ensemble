import numpy as np
from sklearn.tree import DecisionTreeRegressor
from scipy.optimize import minimize_scalar


class RandomForestMSE:
    def __init__(self, n_estimators, max_depth=None, feature_subsample_size=None,
                 **trees_parameters):
        """
        n_estimators : int
            The number of trees in the forest.
        
        max_depth : int
            The maximum depth of the tree. If None then there is no limits.
        
        feature_subsample_size : float
            The size of feature set for each tree. If None then use recommendations.
        """
        self.n_estimators = 0
        self.loop_estimators = n_estimators
        self.max_depth = max_depth
        self.feature_subsample_size = feature_subsample_size
        self.trees_parameters = trees_parameters
        self.reFit = False
        
    def fit(self, X, y, X_val=None, y_val=None):
        """
        X : numpy ndarray
            Array of size n_objects, n_features
            
        y : numpy ndarray
            Array of size n_objects
        X_val : numpy ndarray
            Array of size n_val_objects, n_features
            
        y_val : numpy ndarray
            Array of size n_val_objects           
        """
        if not self.reFit:
            if self.feature_subsample_size is None:
                self.feature_subsample_size = 1
            self.n_features = int(round(X.shape[1] * self.feature_subsample_size))
            self.trees = []
            self.feat_ids_by_tree = []
            self.reFit = True

        for i in range(self.loop_estimators):
            selected_features = np.random.choice(X.shape[1], self.n_features, replace=False)
            self.feat_ids_by_tree.append(selected_features)
            sample = np.random.randint(X.shape[0], size=X.shape[0])

            dt = DecisionTreeRegressor(criterion = 'mse', max_depth = self.max_depth, **self.trees_parameters)
            dt.fit(X[sample][:,selected_features], y[sample])
            self.trees.append(dt)

        self.n_estimators += self.loop_estimators

        if not (X_val is None or y_val is None):
            pred = self.predict(X_val)
            return np.power(pred - y_val, 2).mean()
        
    def predict(self, X):
        """
        X : numpy ndarray
            Array of size n_objects, n_features
            
        Returns
        -------
        y : numpy ndarray
            Array of size n_objects
        """
        sum_preds = np.zeros(X.shape[0])
        for i in range(self.n_estimators):
            sum_preds += self.trees[i].predict(X[:,self.feat_ids_by_tree[i]])
        return sum_preds / self.n_estimators


class GradientBoostingMSE:
    def __init__(self, n_estimators, learning_rate=0.1, max_depth=5, feature_subsample_size=None,
                 **trees_parameters):
        """
        n_estimators : int
            The number of trees in the forest.
        
        learning_rate : float
            Use learning_rate * gamma instead of gamma
        max_depth : int
            The maximum depth of the tree. If None then there is no limits.
        
        feature_subsample_size : float
            The size of feature set for each tree. If None then use recommendations.
        """
        self.n_estimators = 0
        self.loop_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.feature_subsample_size = feature_subsample_size
        self.trees_parameters = trees_parameters
        self.reFit = False
   
    def fit(self, X, y, X_val=None, y_val=None):
        """
        X : numpy ndarray
            Array of size n_objects, n_features
            
        y : numpy ndarray
            Array of size n_objects
        """
        if not self.reFit:
            if self.feature_subsample_size is None:
                self.feature_subsample_size = 1
            self.n_features = int(round(X.shape[1] * self.feature_subsample_size))
            self.trees = []
            self.feat_ids_by_tree = []
            self.gamma = []
            self.f_0 = minimize_scalar(lambda x: np.power(y - x, 2).sum()).x
            self.reFit = True
            self.res = self.f_0 * np.ones(y.shape[0])

        for i in range(self.loop_estimators):
            g = (self.res - y) * 2

            selected_features = np.random.choice(X.shape[1], self.n_features, replace=False)
            self.feat_ids_by_tree.append(selected_features)
            dt = DecisionTreeRegressor(criterion = 'mse', max_depth = self.max_depth, **self.trees_parameters)
            dt.fit(X[:,selected_features], g)
            self.trees.append(dt)

            regions = dt.apply(X[:,selected_features])
            gammas = dict()
            for j in np.unique(regions):
                sample = np.where(regions == j)[0]
                gammas[j] = minimize_scalar(lambda x: np.power(self.res[sample] - x - y[sample], 2).sum()).x
            self.gamma.append(gammas)

            change = np.vectorize(lambda j: gammas[j])(regions)
            self.res -= change * self.learning_rate

        self.n_estimators += self.loop_estimators

        if not (X_val is None or y_val is None):
            pred = self.predict(X_val)
            return np.power(pred - y_val, 2).mean()

    def predict(self, X):
        """
        X : numpy ndarray
            Array of size n_objects, n_features
            
        Returns
        -------
        y : numpy ndarray
            Array of size n_objects
        """
        res = self.f_0
        for i in range(self.n_estimators):
            regions = self.trees[i].apply(X[:,self.feat_ids_by_tree[i]])
            change = np.vectorize(lambda j: self.gamma[i][j])(regions)
            res -= change * self.learning_rate

        return res