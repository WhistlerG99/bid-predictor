import pandas as pd
import numpy as np
from typing import Optional
from feature_engine.outliers import ArbitraryOutlierCapper
from feature_engine.discretisation import ArbitraryDiscretiser
from feature_engine.imputation import (
    ArbitraryNumberImputer,
    AddMissingIndicator,
    MeanMedianImputer,
)


class AddMissingIndicatorCustom(AddMissingIndicator):

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        Learn the variables for which the missing indicators will be created.

        Parameters
        ----------
        X: pandas dataframe of shape = [n_samples, n_features]
            The training dataset.

        y: pandas Series, default=None
            y is not needed in this imputation. You can pass None or y.
        """
        if self.variables:
            self.variables = [v for v in self.variables if v in X]

        super().fit(X, y)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if isinstance(X, (pd.DataFrame, np.ndarray)):
            return super().transform(X)
        else:
            return X


class ArbitraryNumberImputerCustom(ArbitraryNumberImputer):

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        This method does not learn any parameter.

        Parameters
        ----------
        X: pandas dataframe of shape = [n_samples, n_features]
            The training dataset.

        y: None
            y is not needed in this imputation. You can pass None or y.
        """
        if self.imputer_dict:
            self.imputer_dict = {k: v for k, v in self.imputer_dict.items() if k in X}

        super().fit(X, y)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if isinstance(X, (pd.DataFrame, np.ndarray)):
            return super().transform(X)
        else:
            return X


class MeanMedianImputerCustom(MeanMedianImputer):
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        Learn the mean or median value for each variable.

        Parameters
        ----------
        X: pandas dataframe of shape = [n_samples, n_features]
            The training dataset.

        y: pandas Series, default=None
            y is not needed in this imputation. You can pass None or y.
        """
        if self.variables:
            self.variables = [v for v in self.variables if v in X]

        super().fit(X, y)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if isinstance(X, (pd.DataFrame, np.ndarray)):
            return super().transform(X)
        else:
            return X


class ArbitraryOutlierCapperCustom(ArbitraryOutlierCapper):
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        This transformer does not learn any parameter.

        Parameters
        ----------
        X: pandas dataframe of shape = [n_samples, n_features]
            The training input samples.

        y: pandas Series, default=None
            y is not needed in this transformer. You can pass y or None.
        """
        if self.min_capping_dict:
            self.min_capping_dict = {
                k: v for k, v in self.min_capping_dict.items() if k in X
            }
            if not self.min_capping_dict:
                self.min_capping_dict = None
        if self.max_capping_dict:
            self.max_capping_dict = {
                k: v for k, v in self.max_capping_dict.items() if k in X
            }
            if not self.max_capping_dict:
                self.max_capping_dict = None

        if self.min_capping_dict or self.max_capping_dict:
            super().fit(X, y)
        else:
            self.variables_ = []
            self.right_tail_caps_ = {}
            self.left_tail_caps_ = {}
            self.feature_names_in_ = X.columns.to_list()
            self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if isinstance(X, (pd.DataFrame, np.ndarray)):
            return super().transform(X)
        else:
            return X


class ArbitraryDiscretiserCustom(ArbitraryDiscretiser):
    def __init__(
        self,
        binning_dict: Optional[dict] = None,
        bin_names_dict: Optional[dict] = None,
        **kwargs,
    ):
        super().__init__(binning_dict=binning_dict, **kwargs)
        self.bin_names_dict = bin_names_dict

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        This transformer does not learn any parameter.

        Parameters
        ----------
        X: pandas dataframe of shape = [n_samples, n_features]
            The training input samples.

        y: pandas Series, default=None
            y is not needed in this transformer. You can pass y or None.
        """
        if self.binning_dict:
            self.binning_dict = {k: v for k, v in self.binning_dict.items() if k in X}
            if not self.binning_dict:
                self.binning_dict = None

        if self.binning_dict:
            super().fit(X, y)
        else:
            self.variables_ = []
            self.feature_names_in_ = X.columns.to_list()
            self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if isinstance(X, (pd.DataFrame, np.ndarray)):
            X_tf = super().transform(X)
            if self.bin_names_dict:
                X_tf = X_tf.replace(self.bin_names_dict)
            return X_tf
        else:
            return X
