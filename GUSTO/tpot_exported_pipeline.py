import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import RobustScaler
from tpot.builtins import StackingEstimator
from xgboost import XGBClassifier
from tpot.export_utils import set_param_recursive

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=42)

# Average CV score on the training set was: 0.5164279360226546
exported_pipeline = make_pipeline(
    PCA(iterated_power=2, svd_solver="randomized"),
    StackingEstimator(estimator=BernoulliNB(alpha=10.0, fit_prior=False)),
    RobustScaler(),
    StackingEstimator(estimator=GradientBoostingClassifier(learning_rate=0.1, max_depth=10, max_features=0.5, min_samples_leaf=14, min_samples_split=10, n_estimators=100, subsample=0.4)),
    StackingEstimator(estimator=XGBClassifier(learning_rate=0.1, max_depth=1, min_child_weight=1, n_estimators=100, nthread=1, subsample=0.9500000000000001)),
    StackingEstimator(estimator=GradientBoostingClassifier(learning_rate=0.001, max_depth=3, max_features=0.6500000000000001, min_samples_leaf=20, min_samples_split=12, n_estimators=100, subsample=0.5)),
    MLPClassifier(alpha=0.0001, learning_rate_init=0.01)
)
# Fix random state for all the steps in exported pipeline
set_param_recursive(exported_pipeline.steps, 'random_state', 42)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
