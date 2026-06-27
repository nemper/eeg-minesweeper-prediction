import base64
import os
import threading
from datetime import datetime

import matplotlib

matplotlib.use("Agg")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)

import main
import pandas as pd
from fastapi import FastAPI, Form, Request
from fastapi.templating import Jinja2Templates


app = FastAPI()
_progress = {"running": False, "step": "", "index": 0, "total": 7}
_progress_lock = threading.Lock()
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))
pipeline_lock = threading.Lock()


def _set_progress(index: int, step: str, running: bool = True):
    with _progress_lock:
        _progress["index"] = index
        _progress["step"] = step
        _progress["running"] = running


def _read_evaluations_img(filename: str) -> str:
    try:
        with open(filename, "rb") as file:
            encoded = base64.b64encode(file.read()).decode()
        return "data:image/png;base64," + encoded
    except Exception:
        return ""


def _read_text(filename: str) -> str:
    try:
        with open(filename, encoding="utf-8") as file:
            return file.read()
    except Exception:
        return ""


def _read_csv_table(filename: str) -> str:
    try:
        return pd.read_csv(filename).to_html(index=False, border=0)
    except Exception:
        return ""


@app.get("/progress")
def progress():
    with _progress_lock:
        return dict(_progress)


@app.post("/run")
def run(
    request: Request,
    pick_classifier: str = Form("SVC"),
    scoring_classifier: str = Form("accuracy"),
    use_grid_search_classifier: bool = Form(False),
    use_scaling_and_oversampling: bool = Form(False),
    use_same_random_state_classifier: bool = Form(False),
    pick_regressor: str = Form("TSR"),
    scoring_regressor: str = Form("explained_variance"),
    use_grid_search_regressor: bool = Form(False),
    use_augmentation_regression: bool = Form(False),
    use_same_random_state_regressor: bool = Form(False),
    use_fixed_random_state: bool = Form(False),
    use_stratify: bool = Form(False),
    columns_to_drop: list[str] = Form([]),
    timestamps_column: str = Form("Timestamp"),
    amp_1st_column: str = Form("EEG.AF3"),
    pow_1st_column: str = Form("POW.AF3.Theta"),
    intervals_timestamp_column: str = Form("timestamp"),
    game_annotations: list[str] = Form([]),
    number_of_runs: int = Form(5),
    window_length: float = Form(2.0),
    fs: int = Form(128),
    csv_values_ext: str = Form(".md.mc.pm.fe.bp.csv"),
    csv_intervals_ext: str = Form("intervalMarker.csv"),
    json_ext: str = Form(".json"),
):
    prediction_parameters = {
        "use fixed random state": use_fixed_random_state,
        "use stratify": use_stratify,
        "use scaling and oversampling for classification": use_scaling_and_oversampling,
        "use same random state for classifier": use_same_random_state_classifier,
        "use SVC": pick_classifier == "SVC",
        "use DTC": pick_classifier == "DTC",
        "use RFC": pick_classifier == "RFC",
        "use grid search for classifier": use_grid_search_classifier,
        "scoring for classifier grid search": scoring_classifier,
        "use same random state for regressor": use_same_random_state_regressor,
        "use augmentation for regression": use_augmentation_regression,
        "use TSR": pick_regressor == "TSR",
        "use NNR": pick_regressor == "NNR",
        "use GBR": pick_regressor == "GBR",
        "use grid search for regressor": use_grid_search_regressor,
        "scoring for regressor grid search": scoring_regressor,
    }
    processing_parameters = {
        "csv values: columns to drop": list(columns_to_drop),
        "csv values: timestamps column name": timestamps_column,
        "csv values: pow 1st column name": pow_1st_column,
        "csv values: amp 1st column name": amp_1st_column,
        "csv intervals: timestamps column name": intervals_timestamp_column,
        "game annotations": list(game_annotations),
    }
    system_parameters = {
        "path": main.DATA_DIR,
        "number of runs": number_of_runs,
        "window length": float(window_length),
        "fs": fs,
        "files extensions": {
            "csv values files": csv_values_ext,
            "csv intervals files": csv_intervals_ext,
            "json annotations files": json_ext,
        },
    }

    with pipeline_lock:
        main.prediction_parameters = prediction_parameters
        main.processing_parameters = processing_parameters
        main.system_parameters = system_parameters
        main.current_datetime = datetime.now().strftime("%y-%m-%d_%H-%M-%S")
        main.remember_parameters()
        try:
            _set_progress(1, "Extracting data…")
            extracted = main.extract_data()
            _set_progress(2, "Processing & windowing…")
            processed = main.processing(*extracted)
            _set_progress(3, "Extracting EEG features…")
            features = main.eeg_feature_extraction(*processed)
            _set_progress(4, "Building feature matrix…")
            feature_matrix = main.creating_the_feature_matrix(*features)
            _set_progress(5, "Classifying outcomes…")
            classified = main.classification_prediction_and_evaluation(feature_matrix)
            _set_progress(6, "Regressing completion time…")
            regressed = main.regression_prediction_and_evaluation(*classified)
            _set_progress(7, "Rendering results…")
            main.metrics_visualization(*regressed)
        finally:
            _set_progress(_progress["index"], "", running=False)
        dt = main.current_datetime

        evaluations_img = _read_evaluations_img(f"{dt}__evaluations.png")
        predictions_text = _read_text(f"{dt}__predictions.txt")
        feature_matrix_table = _read_csv_table(f"{dt}__feature_matrix.csv")
        parameters_table = _read_csv_table(f"{dt}__parameters.csv")

    return templates.TemplateResponse(
        request,
        "results.html",
        {
            "evaluations_img": evaluations_img,
            "predictions_text": predictions_text,
            "feature_matrix_table": feature_matrix_table,
            "parameters_table": parameters_table,
        },
    )


app.frontend("/", directory=os.path.join(BASE_DIR, "frontend"))
