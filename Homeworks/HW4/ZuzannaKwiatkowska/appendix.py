from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import xgboost
import warnings
import numpy as np
import pandas as pd
from ceteris_paribus.explainer import explain
from ceteris_paribus.profiles import individual_variable_profile
from ceteris_paribus.plots.plots import plot
import cloudpickle

if __name__ == '__main__':

    warnings.filterwarnings('ignore')
    np.random.seed(1)

    data = load_breast_cancer(as_frame=True)
    data, target = data["data"], data["target"]
    data = (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0))

    data_train, data_test, target_train, target_test = train_test_split(data, target, test_size=0.1)

    # train xgboost and calculate prediction
    xgboost_model = xgboost.XGBClassifier(use_label_encoder=False, eval_metric="logloss").fit(data_train, target_train)
    results_xgboost = pd.DataFrame({
        "prediction": xgboost_model.predict(data_test),
        "target": target_test
    })
    results_xgboost["match"] = results_xgboost["prediction"] == results_xgboost["target"]

    # choose interesting samples (misclassified, correct benign, correct malignant)
    interesting_predictions = pd.concat([
        results_xgboost[-results_xgboost["match"]].sample(n=1),
        results_xgboost[results_xgboost["match"] & (results_xgboost["target"] == 1)].sample(n=1),
        results_xgboost[results_xgboost["match"] & (results_xgboost["target"] == 0)].sample(n=1),
    ])
    print(interesting_predictions)

    # train MLP and calculate predictions
    mlp_model = MLPClassifier(max_iter=1000).fit(data_train, target_train)

    # cp decomposition
    explainer_xgb = explain(xgboost_model, data=data_test, y=target_test, label='XGBoost',
                            predict_function=lambda X: xgboost_model.predict_proba(X)[::, 1])
    explainer_mlp = explain(mlp_model, data=data_test, y=target_test, label='MLP',
                            predict_function=lambda X: mlp_model.predict_proba(X)[::, 1])

    # for ix in interesting_predictions.index:
    #     cp_xgb = individual_variable_profile(explainer_xgb, data_test.loc[ix], target_test.loc[ix])
    #     cp_mlp = individual_variable_profile(explainer_mlp, data_test.loc[ix], target_test.loc[ix])
    #     for column in data_test.columns[:10]:
    #         plot(cp_xgb, cp_mlp, selected_variables=[column], destination="browser")

    for column in data_test.columns[:20]:
        for ix in interesting_predictions.index:
            cp_xgb = individual_variable_profile(explainer_xgb, data_test.loc[ix], target_test.loc[ix])
            plot(cp_xgb, selected_variables=[column], destination="browser")
