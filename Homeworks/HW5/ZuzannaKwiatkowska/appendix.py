from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import xgboost
import warnings
import numpy as np
import dalex as dx


def generate_vip(model, test_data, test_target, plot_title, save_file_name):
    explainer = dx.Explainer(model, test_data, test_target, label=plot_title)
    plot = explainer.model_parts().plot(show=False)
    plot.write_image(f'resources/{save_file_name}.png')


if __name__ == '__main__':

    warnings.filterwarnings('ignore')
    np.random.seed(1)

    data = load_breast_cancer(as_frame=True)
    data, target = data["data"], data["target"]
    data_min = data.min(axis=0)
    data_max = data.max(axis=0)
    data_mean = data.mean(axis=0)
    data_stdev = np.std(data, axis=0)

    # raw data, min-max norm, standardization
    data_train, data_test, target_train, target_test = train_test_split(data, target, test_size=0.1)

    data_train_minmax_norm = (data_train - data_min) / (data_max - data_min)
    data_test_minmax_norm = (data_test - data_min) / (data_max - data_min)

    data_train_standard = (data_train - data_mean) / data_stdev
    data_test_standard = (data_test - data_mean) / data_stdev

    # xgboost, no norm
    xgboost1_model = xgboost.XGBClassifier(use_label_encoder=False, eval_metric="logloss").fit(data_train, target_train)
    generate_vip(xgboost1_model, data_test, target_test, "XGBoost, No Norm", "xgboost_no_norm")

    # xgboost, min-max norm
    xgboost2_model = xgboost.XGBClassifier(use_label_encoder=False, eval_metric="logloss").fit(data_train_minmax_norm,
                                                                                               target_train)
    generate_vip(xgboost2_model, data_test_minmax_norm, target_test, "XGBoost, Min-Max Norm", "xgboost_min_max_norm")

    # xgboost, standardization
    xgboost3_model = xgboost.XGBClassifier(use_label_encoder=False, eval_metric="logloss").fit(data_train_standard,
                                                                                               target_train)
    generate_vip(xgboost3_model, data_test_standard, target_test, "XGBoost, Standardization", "xgboost_standard")

    # MLP, no norm
    mlp1_model = MLPClassifier(max_iter=1000).fit(data_train, target_train)
    generate_vip(mlp1_model, data_test, target_test, "MLP, No Norm", "mlp_no_norm")

    # MLP, min-max norm
    mlp2_model = MLPClassifier(max_iter=1000).fit(data_train_minmax_norm, target_train)
    generate_vip(mlp2_model, data_test_minmax_norm, target_test, "MLP, Min-Max Norm", "mlp_min_max_norm")

    # MLP, standardization
    mlp3_model = MLPClassifier(max_iter=1000).fit(data_train_standard, target_train)
    generate_vip(mlp3_model, data_test_standard, target_test, "MLP, Standardization", "mlp_standard")
