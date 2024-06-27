# used Python 3.10
import os, json
from time import time
from datetime import datetime
from collections import Counter
from warnings import simplefilter

import numpy as np
from pandas import read_csv, DataFrame

from scipy.signal import correlate2d
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import(
    accuracy_score, confusion_matrix, classification_report, 
    mean_squared_error, mean_absolute_error, r2_score,)

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.ensemble import RandomForestClassifier as RFC

from sklearn.linear_model import TheilSenRegressor as TSR
from sklearn.neural_network import MLPRegressor as NNR
from sklearn.ensemble import GradientBoostingRegressor as GBR

from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from seaborn import heatmap

import streamlit as st

simplefilter("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"


def extract_data():

    csv_values_data = []
    csv_intervals_data = []
    json_annotations_data = []
    
    directory_path = system_parameters["path"]
    
    files_lists = {key: [] for key in system_parameters["files extensions"]}

    for key, extension in system_parameters["files extensions"].items():
        files_lists[key] = sorted(
            [file for file in os.listdir(directory_path) 
             if file.endswith(extension)],
             )
    
    for i in range(len(files_lists["csv values files"])):
        
        df_values = read_csv(
            os.path.join(directory_path, 
                         files_lists["csv values files"][i]), 
            skiprows=1,
            )
        
        columns_to_drop = processing_parameters["csv values: columns to drop"]

        for j in range(len(columns_to_drop)):
            if columns_to_drop[j] in df_values.columns:
                df_values = df_values.drop(
                    columns_to_drop[j], 
                    axis=1,
                    )
        csv_values_data.append(df_values)
        
        df_intervals = read_csv(
            os.path.join(directory_path, 
                         files_lists["csv intervals files"][i]),
                         )
        csv_intervals_data.append(
            df_intervals[processing_parameters[
                "csv intervals: timestamps column name"]].tolist()
                )
        
        with open(os.path.join(
            directory_path, 
            files_lists["json annotations files"][i]), 
                  "r") as json_file:
            
            json_data = json.load(json_file)

            json_annotations_data.append(
                {key: json_data.get(key, "") 
                 for key in processing_parameters["game annotations"]},
                 )
            
    return csv_values_data, csv_intervals_data, json_annotations_data


def processing(csv_values_data: list, 
               csv_intervals_data: list,
               json_annotations_data: list):
    
    eeg_amplitude = []
    eeg_power = []
    
    for i in range(len(csv_values_data)):
        current_df = csv_values_data[i]
        
        timestamps_column_index = current_df.columns.get_loc(
            processing_parameters["csv values: timestamps column name"],
            )
        amp_1st_column_index = current_df.columns.get_loc(
            processing_parameters["csv values: amp 1st column name"],
            )
        pow_1st_column_index = current_df.columns.get_loc(
            processing_parameters["csv values: pow 1st column name"],
            )
        
        timestamps = current_df.iloc[:, timestamps_column_index].values
        
        indexes = np.where(
            (timestamps > csv_intervals_data[i][0]) & 
            (timestamps < csv_intervals_data[i][1])
            )[0]
        
        eeg_amplitude.append(
            current_df.iloc[:, amp_1st_column_index : amp_1st_column_index + 5].
            values[indexes],
            )
        eeg_power.append(
            current_df.iloc[:, pow_1st_column_index:].
            values[indexes],
            )
    
    column_names = create_columns_names(
        current_df, 
        amp_1st_column_index, 
        pow_1st_column_index,
        )
    
    eeg_amplitude_w, eeg_power_w = windowing(eeg_amplitude, eeg_power)

    return eeg_amplitude_w, eeg_power_w, json_annotations_data, column_names


def create_columns_names(current_df: DataFrame,
                         amp_1st_column_index: int, 
                         pow_1st_column_index: int):
    # INNER
    columns_names = current_df.columns.tolist()

    amp_names = columns_names[amp_1st_column_index : amp_1st_column_index + 5]
    
    columns_names = [f"{j}.{stat}" for stat in ['mean', 'var'] 
                     for j in amp_names + ["Corr", "Xcov"]] \
                        + columns_names[pow_1st_column_index : ] \
                            + processing_parameters["game annotations"]

    return columns_names
            
            
def windowing(eeg_amplitude: list, 
              eeg_power: list):
    # INNER
    samples_per_window = np.round(
        system_parameters["window length"] * 
        system_parameters["fs"],
        ).astype(int)
    
    eeg_amplitude_w = []
    eeg_power_w = []
    
    for i in range(len(eeg_amplitude)):
        eeg_amplitude_i = eeg_amplitude[i]
        eeg_power_i = eeg_power[i]
        
        eeg_amplitude_i_len = len(eeg_amplitude_i)
        
        eeg_amplitude_i_w = []
        eeg_power_i_w = []
        
        for j in range(0, eeg_amplitude_i_len, samples_per_window):
            if j + samples_per_window <= eeg_amplitude_i_len:
                eeg_amplitude_i_w.append(
                    eeg_amplitude_i[j:j + samples_per_window],
                    )
                eeg_power_i_w.append(
                    eeg_power_i[j:j + samples_per_window],
                    )
                
        eeg_amplitude_w.append(
            np.array(eeg_amplitude_i_w),
            )
        eeg_power_w.append(
            np.array(eeg_power_i_w),
            )
        
    return eeg_amplitude_w, eeg_power_w


def eeg_feature_extraction(eeg_amplitude_w: list, 
                           eeg_power_w: list, 
                           json_annotations_data: list, 
                           column_names: list):
    eeg_amplitude_means = []
    eeg_amplitude_vars = []

    eeg_amplitude_corr_means = []
    eeg_amplitude_corr_vars = []
    eeg_amplitude_xcov_means = []
    eeg_amplitude_xcov_vars = []

    eeg_power_means = []
    
    for i in range(len(eeg_amplitude_w)):
        eeg_amplitude_w_i = eeg_amplitude_w[i]
        eeg_amplitude_w_i_len = len(eeg_amplitude_w_i)

        eeg_amplitude_means_i = []
        eeg_amplitude_vars_i = []

        eeg_power_w_i = eeg_power_w[i]
        eeg_power_means_i = []
        
        correlation_matrix = np.zeros(
            shape=(eeg_amplitude_w_i_len, eeg_amplitude_w_i_len),
            )
        
        cross_covariance_matrix = np.zeros_like(
            a=correlation_matrix,
            )
        
        for j in range(eeg_amplitude_w_i_len):
            current_window_amp = eeg_amplitude_w_i[j]

            eeg_amplitude_means_i.append(
                np.mean(current_window_amp, axis=0),
                )
            
            eeg_amplitude_vars_i.append(
                np.var(current_window_amp, axis=0),
                )
            
            current_window_pow = eeg_power_w_i[j]
            
            current_window_pow = np.mean(
                current_window_pow[
                    ~np.isnan(current_window_pow[:, 0])], 
                axis=0,
                )
 
            eeg_power_means_i.append(current_window_pow)
            
            for k in range(eeg_amplitude_w_i_len):
                correlation_matrix[j, k] = np.corrcoef(
                    eeg_amplitude_w_i[j].flatten(), 
                    eeg_amplitude_w_i[k].flatten(),
                    )[0, 1]
                
                if j != k:
                    cross_covariance_matrix[j, k] = correlate2d(
                        eeg_amplitude_w_i[j], 
                        eeg_amplitude_w_i[k], 
                        mode="valid",
                        )[0, 0]
                
        eeg_amplitude_means.append(
            np.mean(np.array(eeg_amplitude_means_i), axis=0),
            )
        eeg_amplitude_vars.append(
            np.mean(np.array(eeg_amplitude_vars_i), axis=0),
            )
        eeg_amplitude_corr_means.append(
            np.mean(correlation_matrix),
            )
        eeg_amplitude_corr_vars.append(
            np.var(correlation_matrix),
            )
        eeg_amplitude_xcov_means.append(
            np.mean(cross_covariance_matrix),
            )
        eeg_amplitude_xcov_vars.append(
            np.var(cross_covariance_matrix),
            )
        eeg_power_means.append(
            np.mean(np.array(eeg_power_means_i), axis=0),
            )
    
    eeg_features = np.hstack(
        tup=(np.array(eeg_amplitude_means), 
             np.array(eeg_amplitude_vars), 
             np.array(eeg_amplitude_corr_means).reshape(-1, 1),
             np.array(eeg_amplitude_corr_vars).reshape(-1, 1),
             (np.array(eeg_amplitude_xcov_means) / 1e10).reshape(-1, 1),
             (np.array(eeg_amplitude_xcov_vars) / 1e19).reshape(-1, 1),
             np.array(eeg_power_means),
             ),
        )
    
    return eeg_features, json_annotations_data, column_names


def creating_the_feature_matrix(eeg_features: np.ndarray,
                                game_annotations: list, 
                                column_names: list):
    
    game_features = {key: [] for key in game_annotations[0].keys()}

    for game_annotation in game_annotations:
        for key, values in game_annotation.items():
            game_features[key].append(values)
    
    time_in_sec = []
    for time_str in game_features["Time"]:
        minutes, seconds = map(int, time_str.split(":"))
        time_in_sec.append(minutes * 60 + seconds)
    game_features["Time"] = time_in_sec
    
    outcome_mapping = {"Win": 1, "Lose": 0}
    game_features["Outcome"] = [outcome_mapping[outcome] 
                                for outcome in game_features["Outcome"]
                                ]
    
    difficulty_mapping = {"Easy": 10, "Normal": 100, "Advanced": 1000}
    keys_to_process = ["Difficulty", "Fields", "Mines"]

    if use_streamlit or \
        prediction_parameters["use non essential game annotations"]:
        for key in keys_to_process:
            if key in game_features:
                if key == "Difficulty":
                    game_features[key] = [difficulty_mapping[difficulty]
                                          for difficulty in game_features[key]]
                else:
                    game_features[key] = [int(value) for value in game_features[key]]

    
    for key, values in game_features.items():
        game_features[key] = np.array(values).reshape(-1, 1)

    game_features = np.hstack(
        tup=list(game_features.values()),
        )

    feature_matrix = DataFrame(
        data=np.hstack(
            tup=(eeg_features, game_features)), 
        columns=column_names,
        )
    
    feature_matrix.to_csv(f"{current_datetime}__feature_matrix.csv", index=False)
    
    return feature_matrix


def classification_prediction_and_evaluation(feature_matrix: DataFrame):
    game_outcomes = feature_matrix.pop("Outcome")
    
    classification_metrics = {
        "accuracies": [],
        "confusion_matrices": [],
        "classification_reports": [],
        }
    
    outcomes_str = "GAME OUTCOMES:\n\n"
    
    start = time()
    for i in range(system_parameters["number of runs"]):
        while True:
            random_state = 42 if prediction_parameters["use fixed random state"] \
                else np.random.randint(1000)
            
            X_train, X_test, y_train, y_test = train_test_split(
                feature_matrix.values, game_outcomes,
                test_size=0.3, random_state=random_state,
                stratify=game_outcomes if prediction_parameters["use stratify"] \
                    else None,
                    )
            
            class_counts = Counter(y_train)
            
            if all(count >= 2 for count in class_counts.values()) \
                and len(set(y_test)) >= 2 and len(set(y_train)) >= 2:
                break

        if prediction_parameters["use scaling and oversampling for classification"]:
            scaler = StandardScaler()
            X_train= scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            
            smote = SMOTE(
                sampling_strategy="auto", random_state=random_state,
                k_neighbors=1, n_jobs=-1)
            X_train, y_train = smote.fit_resample(X_train, y_train)
        
        classifier_random_state = random_state \
            if prediction_parameters["use same random state for classifier"] \
                else None
        
        if use_streamlit:
            st_classifier = prediction_parameters["pick classifier"]
            combo_map = {
                "use SVC": ("use SVC", False, False),
                "use DTC": (False, "use DTC", False),
                "use RFC": (False, False, "use RFC"),
                }
            prediction_parameters.update(
                dict(zip(["use SVC", "use DTC", "use RFC"], 
                         combo_map[st_classifier])))
                
        if prediction_parameters["use SVC"]:
            classifier = SVC(random_state=classifier_random_state)
            param_grid = {
                "C": [0.1, 1],
                "kernel": ["linear", "rbf"],
                "gamma": ["scale", "auto", 0.1],
                "class_weight": [None, "balanced"],
                }
        else:
            param_grid = {
                "criterion": ["gini", "entropy"],
                "max_depth": [None, 2],
                "min_samples_split": [2, 4],
                "min_samples_leaf": [1, 2],
                "max_features": ["auto", "sqrt"],
                }
            if prediction_parameters["use DTC"]:
                classifier = DTC(random_state=classifier_random_state)
                
            elif prediction_parameters["use RFC"]:
                classifier = RFC(random_state=classifier_random_state)
                param_grid["n_estimators"]: [100, 200]
        
        grid_search = GridSearchCV(
            classifier, param_grid, cv=3, n_jobs=-1,
            scoring=prediction_parameters["scoring for classifier grid search"],
            )
        
        grid_search.fit(X_train, y_train)
        best_parameters = grid_search.best_params_
        
        for param_name, param_value in best_parameters.items():
            setattr(classifier, param_name, param_value)
        
        classifier.fit(X_train, y_train)
        predicted_outcomes = classifier.predict(X_test)

        classification_metrics["accuracies"].append(
            accuracy_score(y_test, predicted_outcomes),
            )
        
        classification_metrics["confusion_matrices"].append(
            confusion_matrix(y_test, predicted_outcomes),
            )
        
        classification_metrics["classification_reports"].append(
            classification_report(y_test, predicted_outcomes, output_dict=True),
            )
        
        outcomes_str += f"""
        RUN {i+1}
        \tpredicted
        {np.where(predicted_outcomes == 1.0, "Win", "Lose")}
        \treal
        {np.where(y_test == 1.0, "Win", "Lose")}
        """ 
    end = time()
    
    with open(f"{current_datetime}__predictions.txt", "w") as f:
        f.write(f"Total execution time - {classifier}:\n \
            {np.round(end - start, 3)} s\n\n")
        f.write(outcomes_str + "\n\n")
    
    feature_matrix["Outcome"] = game_outcomes
    
    return feature_matrix, classification_metrics


def calculate_the_average_classification_report(classification_reports):
    # INNER
    average_classification_report = {}

    for report in classification_reports:
        for key, value in report.items():
            if key not in average_classification_report:
                average_classification_report[key] = value
            else:
                if isinstance(value, dict):
                    for inner_key, inner_value in value.items():
                        average_classification_report[key][inner_key] += inner_value
                else:
                    average_classification_report[key] += value

    for key, value in average_classification_report.items():
        if isinstance(value, dict):
            for inner_key, inner_value in value.items():
                average_classification_report[key][inner_key] /= \
                    len(classification_reports)
        else:
            average_classification_report[key] /= len(classification_reports)
            
    return DataFrame(average_classification_report).transpose()


def regression_prediction_and_evaluation(feature_matrix: DataFrame, 
                                         classification_metrics: dict):
    completion_time = feature_matrix.pop("Time")
    
    regression_metrics = {
        "mse_values": [],
        "mae_values": [],
        "r2_values": [],
        }
    
    completion_times_str = "COMPLETION TIMES:\n\n"
    
    start = time()
    for i in range(system_parameters["number of runs"]):
        random_state = 42 if prediction_parameters["use fixed random state"] \
            else np.random.randint(1000)
            
        X_train, X_test, y_train, y_test = train_test_split(
            feature_matrix.values, completion_time, 
            test_size=0.3, random_state=random_state)
        
        if prediction_parameters["use augmentation for regression"]:
            X_train_augmented = np.vstack((X_train, X_train + 0.01 * 
                                        np.random.standard_normal(X_train.shape)))
            y_train_augmented = np.hstack((y_train, y_train))
        else:
            X_train_augmented, y_train_augmented = X_train, y_train
        
        regressor_random_state = random_state \
            if prediction_parameters["use same random state for regressor"] \
                else None
        
        if use_streamlit:
            st_regressor = prediction_parameters["pick regressor"]
            combo_map = {
                "use TSR": ("use TSR", False, False),
                "use NNR": (False, "use NNR", False),
                "use GBR": (False, False, "use GBR"),
                }
            prediction_parameters.update(
                dict(zip(["use TSR", "use NNR", "use GBR"], 
                         combo_map[st_regressor])))
    
        if prediction_parameters["use TSR"]:
            regressor = TSR(random_state=regressor_random_state)
            
        elif prediction_parameters["use NNR"]:
            regressor = NNR(random_state=regressor_random_state)
            param_grid = {
                "alpha": [0.1, 0.5, 1.0],
                "hidden_layer_sizes": [(50,), (100,), (50, 50), (100, 50)],
                "activation": ["relu", "tanh"],
                "learning_rate_init": [0.001, 0.01],
                }
        elif prediction_parameters["use GBR"]:
            regressor = GBR(random_state=regressor_random_state)
            param_grid = {
                "n_estimators": [100, 200],
                "learning_rate": [0.01, 0.05],
                "max_depth": [2, 3],
                "min_samples_split": [2, 4],
                "min_samples_leaf": [1, 2],
                }
        
        if not prediction_parameters["use TSR"]:
            grid_search = GridSearchCV(
                regressor, param_grid, cv=3, n_jobs=-1,
                scoring=prediction_parameters["scoring for regressor grid search"])
            
            grid_search.fit(X_train_augmented, y_train_augmented)
            
            best_parameters = grid_search.best_params_
            for param_name, param_value in best_parameters.items():
                setattr(regressor, param_name, param_value)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_augmented)
        X_test = scaler.transform(X_test)
        
        regressor.fit(X_train, y_train_augmented)  
        predicted_completion_times = regressor.predict(X_test)

        regression_metrics["mse_values"].append(
            mean_squared_error(y_test, predicted_completion_times))
        regression_metrics["mae_values"].append(
            mean_absolute_error(y_test, predicted_completion_times))
        regression_metrics["r2_values"].append(
            r2_score(y_test, predicted_completion_times))
        
        completion_times_str += f"""
        RUN {i+1}
        \tpredicted
        {predicted_completion_times.astype(int)}
        \treal
        {y_test.astype(int).values}
        """
    end = time()
    
    with open(f"{current_datetime}__predictions.txt", "a") as f:
        f.write(f"Total execution time - {regressor}:\n \
            {np.round(end - start, 3)} s\n\n")
        f.write(completion_times_str)
        
    feature_matrix["Time"] = completion_time
    
    return classification_metrics, regression_metrics


def scorings_for_grid_search():
    # INNER
    scorings_for_classifier = (
        "accuracy", "balanced_accuracy", "top_k_accuracy", 
        "f1", "f1_macro", "f1_micro", "f1_samples", "f1_weighted", 
        "jaccard", "jaccard_macro", "jaccard_micro", "jaccard_samples", 
        "jaccard_weighted", 
        "neg_brier_score", "neg_log_loss", 
        "precision", "average_precision", "precission_macro", 
        "precission_micro", "precission_samples", "precission_weighted", 
        "recall", "recall_macro", "recall_micro", "recall_samples", 
        "recall_weighted", 
        "roc_auc", "roc_auc_ovo", "roc_auc_ovo_weighted", 
        "roc_auc_ovr", "roc_auc_ovr_weighted", 
        )
    scorings_for_regressor = (
        "explained_variance", "max_error", 
        "neg_mean_absolute_error", "neg_mean_absolute_percentage_error", 
        "neg_mean_squared_log_error", "neg_mean_poisson_deviance", 
        "neg_mean_squared_error", "neg_mean_squared_log_error", 
        "neg_median_absolute_error", "neg_root_mean_squared_error", 
        "r2", 
        )
    return scorings_for_classifier, scorings_for_regressor


def metrics_visualization(classification_metrics: dict, regression_metrics: dict):
    runs_range = range(1, system_parameters["number of runs"] + 1)
    
    fig, axs = plt.subplots(nrows=2, ncols=3)
    fig.set_size_inches(w=12, h=7)
    fig.subplots_adjust(wspace=0.35, hspace=0.35)


    def plot_1D_metric(ax, x, y, xlabel, title, color):
        ax.plot(x, y, color=color, linewidth=3)
        ax.set_xlabel(xlabel)
        ax.set_title(title)
        ax.grid(True)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))


    def plot_2D_metric(ax, data, xlabel, ylabel, title, fmt, cmap):
        heatmap(data, annot=True, fmt=fmt, cmap=cmap, cbar=False, ax=ax)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if "report" in title:
            ax.yaxis.set_label_coords(-0.2, 0.8)
        ax.set_title(title)


    plot_1D_metric(
        ax=axs[0, 0], x=runs_range, y=classification_metrics["accuracies"], 
        xlabel="run", title="Accuracy over runs", color="orange")
    
    plot_2D_metric(
        ax=axs[0, 1], 
        data=np.mean(
            classification_metrics["confusion_matrices"], axis=0), 
        xlabel="Predicted labels", ylabel="True labels", 
        title="Avg confusion matrix", fmt=".2f", cmap="Greens")

    plot_2D_metric(
        ax=axs[0, 2], 
        data=calculate_the_average_classification_report(
            classification_metrics["classification_reports"]), 
        xlabel="Metrics", ylabel="Classes",
        title="Avg classification report", fmt=".2f", cmap="BuPu")
    
    plot_1D_metric(
        ax=axs[1, 0], x=runs_range, y=regression_metrics["mse_values"], 
        xlabel="run", title="MSE over runs", color="blue")
    
    plot_1D_metric(
        ax=axs[1, 1], x=runs_range, y=regression_metrics["mae_values"], 
        xlabel="run", title="MAE over runs", color="green")
    
    plot_1D_metric(
        ax=axs[1, 2], x=runs_range, y=regression_metrics["r2_values"], 
        xlabel="run", title="R2 over runs", color="red")
    
    plt.savefig(f"{current_datetime}__evaluations.png")
    plt.show()
    

def if_running_a_regular_script():
    # 13 + 2 elemenata
    pred_parameters = {
            "use non essential game annotations" : False,
            "use fixed random state" : False,
            "use stratify" : False,
            "use scaling and oversampling for classification" : False,
            "use same random state for classifier" : False,
            "use SVC" : True,
            "use DTC" : False,
            "use RFC" : False,
            "use same random state for regressor" : False,
            "use augmentation for regression" : False,
            "use TSR" : False,
            "use NNR" : True,
            "use GBR" : False,
            "scoring for classifier grid search" : "accuracy",
            "scoring for regressor grid search" : "neg_mean_squared_error",
            }
    
    # 6 elemenata
    proc_parameters = {
        "csv values: columns to drop" : ["OriginalTimestamp"],
        "csv values: timestamps column name" : "Timestamp",
        "csv values: pow 1st column name" : "POW.AF3.Theta",
        "csv values: amp 1st column name" : "EEG.AF3",
        "csv intervals: timestamps column name" : "timestamp",
        "game annotations" : ["Time", "Outcome"],
        }
    
    # 5 (3) elemenata
    sys_parameters = {
        "path" : r"C:\Users\neman\Desktop\master_python\newdata",
        "number of runs" : 3,
        "window length" : 2.0,
        "fs" : 128,
        "files extensions" : {
            "csv values files" : ".md.mc.pm.fe.bp.csv",
            "csv intervals files" : "intervalMarker.csv",
            "json annotations files" : ".json",
            },
        }
    
    if pred_parameters["use non essential game annotations"]:
        proc_parameters["game annotations"].extend(
            ["Difficulty", "Fields", "Mines"])
    
    check, errors_text = check_various_conditions(
        pred_parameters, proc_parameters, sys_parameters)
    
    if check:
        return pred_parameters, proc_parameters, sys_parameters, None
    else:
        return None, None, None, errors_text


def check_various_conditions(pred_parameters: dict, 
                             proc_parameters: dict, 
                             sys_parameters: dict):
    # INNER
    failed_vconditions = "\n\nGreške:\n"

    scorings_for_classifier, scorings_for_regressor = scorings_for_grid_search()

    vcondition_0 = os.path.exists(sys_parameters["path"])

    vcondition_1 = all(
        [isinstance(value, bool) for key, value in pred_parameters.items()
         if key not in ["scoring for classifier grid search",
                        "scoring for regressor grid search"]])
    
    vcondition_1a = pred_parameters["scoring for classifier grid search"] in \
        scorings_for_classifier
    
    vcondition_1b = pred_parameters["scoring for regressor grid search"] in \
        scorings_for_regressor
        
    vcondition_2 = np.sum([pred_parameters["use SVC"], 
                           pred_parameters["use DTC"],
                           pred_parameters["use RFC"]]
                           ) == 1
    
    vcondition_3 = np.sum([pred_parameters["use TSR"], 
                           pred_parameters["use NNR"],
                           pred_parameters["use GBR"]]
                           ) == 1
    
    vcondition_4 = True
    try:
        if isinstance(int(sys_parameters["number of runs"]), int):
            pass
    except:
        vcondition_4 = False
        
    vcondition_5 = sys_parameters["number of runs"] > 0

    vcondition_6 = isinstance(sys_parameters["window length"], float)

    vcondition_7 = 0.1 <= sys_parameters["window length"] <= 5

    vcondition_8 = sys_parameters["fs"] in [2**i for i in range(7, 10 + 1)]
    
    vcondition_9 = True
    for key, value in proc_parameters.items():
        if key in ["csv values: columns to drop", "game annotations"]:
            if isinstance(value, list):
                if key == "game annotations" and len(value) == 0:
                    vcondition_9 = False
                    break
            else:
                vcondition_9 = False
                break
            for list_value in value:
                outer_break = False
                if not isinstance(list_value, str):
                    outer_break = True
                    vcondition_9 = False
                    break
            if outer_break:
                break
        else:
            if not isinstance(value, str):
                vcondition_9 = False
                break
    
    vcondition_10 = True
    for suffix in sys_parameters["files extensions"].values():
        if not isinstance(suffix, str):
            vcondition_10 = False
            break
        elif suffix == sys_parameters["files extensions"]["json annotations files"]:
            if not suffix.endswith(".json"):
                vcondition_10 = False
                break
        elif not suffix.endswith(".csv"):
            vcondition_10 = False
            break
        
    vcondition_11 = (len(pred_parameters) == 16 and 
                     len(proc_parameters) == 6 and 
                     len(sys_parameters) == 5 and 
                     len(sys_parameters["files extensions"]) == 3)
        
    if not vcondition_0:
        failed_vconditions += "- Zadata je nepostojeća putanja.\n"

    if not vcondition_1:
        failed_vconditions += """- Parametri koji se koriste u sklopu predikcija 
        (osim scoring-a) moraju svi biti definisani kao boolean-i.\n"""

    if not vcondition_1a:
        failed_vconditions += """- Metoda za ocenjivanje klasifikatora pri traženju
        optimalnih parametara nije nađena u predefinisanom skupu.\n"""

    if not vcondition_1b:
        failed_vconditions += """- Metoda za ocenjivanje regresora pri traženju 
        optimalnih parametara nije nađena u predefinisanom skupu.\n"""

    if not vcondition_2:
        failed_vconditions += "- Isključivo jedan klasifikator mora biti aktivan.\n"

    if not vcondition_3:
        failed_vconditions += "- Isključivo jedan regresor mora biti aktivan.\n"

    if not vcondition_4:
        failed_vconditions += "- Broj iteracija sistema mora biti intedžer.\n"

    if not vcondition_5:
        failed_vconditions += "- Broj iteracija sistema mora biti > 1.\n"

    if not vcondition_6:
        failed_vconditions += "- Dužina prozora mora biti float vrednost.\n"

    if not vcondition_7:
        failed_vconditions += "- Nedozvoljena dužina prozora.\n"

    if not vcondition_8:
        failed_vconditions += "- Nedozvoljena frekvencija odabiranja.\n"

    if not vcondition_9:

        failed_vconditions += """- Svi parametri koji označavaju imena kolona
        moraju biti str (sem imena kolona za izbacivanje i anotacija igre;
        oni moraju biti liste ispunjene str).\n"""

    if not vcondition_10:
        failed_vconditions += "- Sufiksi fajlova nisu pravilno definisani.\n"

    if not vcondition_11:
        failed_vconditions += "- Rečnici parametara nisu originalne dužine.\n"

    if failed_vconditions != "\n\nGreške:\n":
        return False, failed_vconditions
    else:
        return True, None
    

def if_running_a_streamlit_app():
    st.set_page_config(layout="wide")
    
    edit_streamlit_style = """
        <style>
            MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            #root > div:nth-child(1) > div > div > div > div > \
            section > div {padding-top: 2rem;}
        </style>
        """
    st.markdown(body=edit_streamlit_style, unsafe_allow_html=True)
    
    col_1, _, col_2, col_3 = st.columns(spec=[8, 1, 5, 2])
    
    col_1.text(body="")
    col_1.subheader(body="Pomoćni alat za master tezu")
    
    col_2.text(body="")
    col_2.info(body="Nemanja Peruničić, B1 3/22", icon="🎓")

    try:
        col_3.image(image="ftn_logo_fun.gif")
    except:
        pass
    
    st.divider()
    _, col_a, _, col_b = st.columns(spec=[1, 14, 1, 12])
    
    with col_a:
        scorings_for_classifier, scorings_for_regressor = scorings_for_grid_search()

        pred_parameters = {
            "use fixed random state" : st.checkbox(
                label="use fixed random state",
                ),
            "divider 1" : st.divider(
                ),
            "use stratify" :  st.checkbox(
                label="use stratify",
                ),
            "use scaling and oversampling for classification" : st.checkbox(
                label="use scaling and oversampling for classification",
                ),
            "use same random state for classifier" : st.checkbox(
                label="use same random state for classifier",
                ),
            "pick classifier" : st.selectbox(
                label="pick classifier",
                options=["use SVC", "use DTC", "use RFC"],
                ),
            "use grid search for classifier": st.checkbox(
                label="use grid search for classifier",
                ),
            "scoring for classifier grid search" : st.selectbox(
                label="pick classifier scoring",
                options=scorings_for_classifier,
                ),
            "divider 2" : st.divider(
                ),
            "use same random state for regressor" : st.checkbox(
                label="use same random state for regressor",
                ),
            "use augmentation for regression" : st.checkbox(
                label="use augmentation for regression",
                ),
            "pick regressor" : st.selectbox(
                label="pick regressor",
                options=["use TSR", "use NNR", "use GBR"],
                ),
            "use grid search for regressor": st.checkbox(
                label="use grid search for regressor",
                ),
            "scoring for regressor grid search" : st.selectbox(
                label="pick regressor scoring",
                options=scorings_for_regressor,
                ),
            "fake divider" : st.text(
                body="",
                ),
            }
        try:
            st.image(image="eeg_potato.jpg")
        except:
            pass
            
    with col_b:
        proc_parameters = {
            "csv values: columns to drop" : st.multiselect(
                label="csv values: columns to drop",
                options=["OriginalTimestamp"],
                default=["OriginalTimestamp"],
                ),
            "csv values: timestamps column name" : st.selectbox(
                label="csv values: timestamps column name", 
                options=["Timestamp"],
                ),
            "csv values: pow 1st column name" : st.selectbox(
                label="csv values: pow 1st column name", 
                options=["POW.AF3.Theta"],
                ),
            "csv values: amp 1st column name" : st.selectbox(
                label="csv values: amp 1st column name", 
                options=["EEG.AF3"],
                ),
            "csv intervals: timestamps column name" : st.selectbox(
                label="csv intervals: timestamps column name",
                options=["timestamp"],
                ),
            "game annotations" : st.multiselect(
                label = "game annotations",
                options=["Time", "Outcome", "Difficulty", "Fields", "Mines"],
                default=["Time", "Outcome"],
                help = "Ne izostaviti obavezne parametre!"),
            }
        
        sys_parameters = {
            "path" : st.selectbox(
                label="path", 
                options=["C:/Users/neman/Desktop/master_python/newdata"],
                ),
            "divider 1" : st.divider(
                ),
            "number of runs" : st.slider(
                label="number of runs", 
                min_value=1, max_value=50, step=1, value=5,
                ),
            "window length" : st.slider(
                label="window length", 
                min_value=0.1, max_value=5.0, step=0.1, value=2.0,
                ),
            "fs" : st.select_slider(
                label="sampling frequency", 
                options=[2**i for i in range(7, 10 + 1)], value=2**7,
                ),
            "divider 2" : st.divider(
                ),
            "files extensions" : {
                "csv values files" : st.selectbox(
                    label="csv values files", 
                    options=[".md.mc.pm.fe.bp.csv"],
                    ),
                "csv intervals files" : st.selectbox(
                    label="csv intervals files", 
                    options=["intervalMarker.csv"],
                    ),
                "json annotations files" : st.selectbox(
                    label="json annotations files", 
                    options=[".json"],
                    ),
                },
            }
    st.divider()
    
    return pred_parameters, proc_parameters, sys_parameters


def remember_parameters():
    # INNER
    combined_dict = {**prediction_parameters, 
                     **processing_parameters, 
                     **system_parameters}
    combined_dict = {k: v for k, v in combined_dict.items() if "divider" not in k}
    
    df = DataFrame(list(combined_dict.items()), columns=['Key', 'Value'])
    df.to_csv(f"{current_datetime}__parameters.csv", index=False)
    

def experimental(how_many_parameters, range_classifiers, range_regressors):
    # INNER
    combo_prediction_parameters = []
    x = f"{how_many_parameters:02d}b"
    
    for i in range(2^how_many_parameters):
        binary = format(i, x)

        if any(binary[range_classifiers].count('1'), 
               binary[range_regressors].count('1'))  == 1:
            
            boolean_list = [bool(int(bit)) for bit in binary]
            combo_prediction_parameters.append(boolean_list)

    return combo_prediction_parameters

    
def machine_learning_system():
    metrics_visualization(
        *regression_prediction_and_evaluation(
            *classification_prediction_and_evaluation(
                creating_the_feature_matrix(
                    *eeg_feature_extraction(
                        *processing(
                            *extract_data(
                            )))))))
    _ = """
    start lines:
    
    extract_data - 113
    processing - 171
        create_columns_names - 218
        windowing - 234
    eeg_feature_extraction - 273
    creating_the_feature_matrix - 375
    classification_prediction_and_evaluation - 428
        calculate_the_average_classification_report - 551
    regression_prediction_and_evaluation - 577
        scorings_for_grid_search - 685
    metrics_visualization - 712
    _____________________________________________________
    
    if_running_a_regular_script - 771
        check_various_conditions - 827
    if_running_a_streamlit_app - 968
    remember_parameters - 1125
    """


if __name__ == "__main__":
    # !!!    
    use_streamlit = True
    _ = """         ^^^^^
    if True >> u terminalu >> "streamlit run nemanja_perunicic_mas.py"
    
    if False >> proverite vrednosti parametara u funkciji:
    if_running_a_regular_script() >> pa potom standardno Run dugme/komanda
    """
 
    if not use_streamlit:
        (prediction_parameters, 
         processing_parameters, 
         system_parameters, 
         errors
         ) = if_running_a_regular_script()
        
        current_datetime = datetime.now().strftime("%y-%m-%d_%H-%M-%S")
        
        if errors:
            print(errors)
        else:
            remember_parameters()
            machine_learning_system()
            
    else:
        (prediction_parameters, 
         processing_parameters, 
         system_parameters
         ) = if_running_a_streamlit_app()
        
        status = st.empty()
        start = st.button(
            label="**:green[START THE SYSTEM]** 🧠💻📊", 
            use_container_width=True,
            )
        
        if start:
            status.subheader("🏃 🏃🏽 🏃🏻 🏃🏾 🏃🏼 🏃🏿 ↗️")
            
            current_datetime = datetime.now().strftime("%y-%m-%d_%H-%M-%S")
            remember_parameters()
            machine_learning_system()
            
            status.subheader("🌬️ ... gotovo! ➡️ Možete skrolovati nadole...")
            
            try:
                st.image(f"{current_datetime}__evaluations.png")
            except:
                pass
            try:
                with open(f"{current_datetime}__predictions.txt", "r") as f:
                    st.divider()
                    st.text(body=f.read())
            except:
                pass
            try:
                st.divider()
                st.table(data=read_csv(f"{current_datetime}__feature_matrix.csv"))
            except:
                pass
            try:
                st.divider()
                st.table(data=read_csv(f"{current_datetime}__parameters.csv"))
            except:
                pass

            
#                               \__(*~*)__/   ta-da        