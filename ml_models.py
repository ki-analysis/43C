# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 14:44:10 2021

@author: serge
"""

import sys
import os

sys.stderr = open(os.devnull, "w")  # silence stderr
from constants import RANDOM_STATE
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.gaussian_process import GaussianProcessClassifier

sys.stderr = sys.__stderr__


"""
Classification models with feature selection
"""

SELECTOR_PERCENTILE = [100]
C = [1e-3, 1e-2, 1e-1, 1e0]
IMPUTER_STRATEGY = ["median"]
FRAC = [0.1, 0.5, 0.9]

# Linear SVC classifier
svc_selector_pipeline = Pipeline(
    [
        ("scaler", MinMaxScaler()),
        ("imputer", SimpleImputer(add_indicator=True)),
        ("selector", SelectPercentile(mutual_info_classif, percentile=100)),
        ("svc", SVC(kernel="linear", class_weight="balanced", probability=True, tol=1e-4, random_state=RANDOM_STATE),),
    ]
)
svc_selector_grid = {
    "svc__C": C,
    "imputer__strategy": IMPUTER_STRATEGY,
    "selector__percentile": SELECTOR_PERCENTILE,
}

# logistic regression classifier
lr_selector_pipeline = Pipeline(
    [
        ("scaler", MinMaxScaler()),
        ("imputer", SimpleImputer(add_indicator=True)),
        ("selector", SelectPercentile(mutual_info_classif, percentile=100)),
        (
            "logistic",
            LogisticRegression(
                solver="saga", max_iter=20000, penalty="elasticnet", class_weight="balanced", random_state=RANDOM_STATE
            ),  # without elasticnet penalty, LR can get awful performance
        ),
    ]
)
lr_selector_grid = {
    "logistic__C": C,
    "imputer__strategy": IMPUTER_STRATEGY,
    "logistic__l1_ratio": FRAC,
    "selector__percentile": SELECTOR_PERCENTILE,
}

# HistGradientBoostingClassifier as a lightgbm variant
gdbt_selector_pipeline = Pipeline(
    [
        (
            "gbdt",
            HistGradientBoostingClassifier(max_iter=500, max_bins=50, early_stopping=False, random_state=RANDOM_STATE),
        ),
    ]
)

gdbt_selector_grid = {
    "gbdt__learning_rate": [0.01, 0.05, 0.1, 0.5],
    "gbdt__min_samples_leaf": [2, 5, 10],
}

# random forest
rf_selector_pipeline = Pipeline(
    [
        ("imputer", SimpleImputer(add_indicator=True)),
        ("selector", SelectPercentile(mutual_info_classif, percentile=100)),
        ("rf", RandomForestClassifier(random_state=RANDOM_STATE)),
    ]
)


rf_selector_grid = {
    "rf__max_depth": [1, 2, 4, 8, 16, None],
    "imputer__strategy": IMPUTER_STRATEGY,
    "rf__max_features": FRAC,
    "selector__percentile": SELECTOR_PERCENTILE,
}

# QDA
qda_selector_pipeline = Pipeline(
    [
        ("scaler", MinMaxScaler()),
        ("imputer", SimpleImputer(add_indicator=True)),
        ("selector", SelectPercentile(mutual_info_classif, percentile=100)),
        ("qda", QuadraticDiscriminantAnalysis()),
    ]
)


qda_selector_grid = {
    "qda__reg_param": [1e-3, 1e-2, 1e-1, 0.5],
    "imputer__strategy": IMPUTER_STRATEGY,
    "selector__percentile": SELECTOR_PERCENTILE,
}

# GPR
gpr_selector_pipeline = Pipeline(
    [
        ("scaler", MinMaxScaler()),
        ("imputer", SimpleImputer(add_indicator=True)),
        ("selector", SelectPercentile(mutual_info_classif, percentile=100)),
        ("gpr", GaussianProcessClassifier(warm_start=True, random_state=RANDOM_STATE)),
    ]
)


gpr_selector_grid = {
    "imputer__strategy": IMPUTER_STRATEGY,
    "selector__percentile": SELECTOR_PERCENTILE,
}

"""
Classification models without feature selection for the stacker phase
note: currently not being used. only LR is used for simplicity.
"""
svc = SVC(kernel="linear", class_weight="balanced", probability=True, tol=1e-4, random_state=RANDOM_STATE)

lr = LogisticRegression(max_iter=10000, solver="liblinear", random_state=RANDOM_STATE)

# Create a pipeline
meta_pipeline = Pipeline([("classifier", lr)])

# Create space of candidate learning algorithms and their hyperparameters
meta_grid = [{"classifier": [svc, lr]}]
