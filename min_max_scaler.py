import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

class MinMaxScaler(BaseEstimator, TransformerMixin):
    def __init__(self, negative_scale=False):
        """
        """
        self.negative_scale_ = negative_scale

    def fit(self, X):
        self.feature_atts_ = []
        self.n_features_ = X.shape[1]
        for i in range(self.n_features_):
            feat = X[:,i] 
            min_feat = np.min(feat)
            max_feat = np.max(feat)
            self.feature_atts_.append((min_feat, max_feat))

    def transform(self, X):
        check_is_fitted(self, 'feature_atts_')
        if self.n_features_ != X.shape[1]:
            raise Exception(f'MinMaxScaler was fitted to {self.n_features} features, but passed in data has {X.shape[1]} features')

        X_c = X.copy()
        for i in range(self.n_features_):
            min_feat = self.feature_atts_[i][0]
            max_feat = self.feature_atts_[i][1]

            if max_feat-min_feat == 0:
                continue

            feat = X_c[:,i]
            feat = (feat - min_feat)/(max_feat-min_feat)

            if self.negative_scale_:
                feat = feat*2-1

            X_c[:,i] = feat

        return X_c 


    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)