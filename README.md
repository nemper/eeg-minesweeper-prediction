# EEG Data Analysis and Machine Learning System

This repository contains a comprehensive Python script developed for the analysis of EEG data and the application of machine learning techniques to classify outcomes and predict completion times based on EEG features. The script is designed to be flexible and user-friendly, offering both a command-line interface and a graphical interface via Streamlit.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [Command-Line Interface](#command-line-interface)
  - [Streamlit Application](#streamlit-application)
- [Data Processing Pipeline](#data-processing-pipeline)
  - [Data Extraction](#data-extraction)
  - [Preprocessing](#preprocessing)
  - [Feature Extraction](#feature-extraction)
  - [Machine Learning Models](#machine-learning-models)
- [Parameters Configuration](#parameters-configuration)
  - [System Parameters](#system-parameters)
  - [Processing Parameters](#processing-parameters)
  - [Prediction Parameters](#prediction-parameters)
- [Results Visualization](#results-visualization)
- [License](#license)

## Overview

The script processes EEG data recorded during gameplay sessions to extract meaningful features and apply machine learning algorithms for:

- **Classification**: Predicting the outcome of a game (Win or Lose).
- **Regression**: Predicting the completion time of a game.

The script supports various machine learning models and includes functionalities for data preprocessing, feature extraction, hyperparameter tuning, and results visualization.

## Features

- **Data Extraction**: Reads EEG data, timestamps, and game annotations from CSV and JSON files.
- **Preprocessing**: Cleans and segments the data based on specified intervals.
- **Feature Extraction**: Computes statistical features from the EEG signals, including means, variances, correlations, and power spectral densities.
- **Machine Learning Models**: Implements classifiers (SVC, Decision Tree, Random Forest) and regressors (Theil-Sen, Neural Network, Gradient Boosting).
- **Hyperparameter Tuning**: Utilizes GridSearchCV for optimal parameter selection.
- **Results Visualization**: Generates plots for metrics such as accuracy, confusion matrices, and regression errors.
- **User Interface**: Offers both command-line and Streamlit graphical interfaces for parameter selection.

## Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/your_username/your_repository.git
   ```

2. **Navigate to the directory**:

   ```bash
   cd your_repository
   ```

3. **Install the required packages**:

   The script requires Python 3.10 and the following packages:

   - numpy
   - pandas
   - scipy
   - scikit-learn
   - imbalanced-learn
   - matplotlib
   - seaborn
   - streamlit

   Install them using:

   ```bash
   pip install -r requirements.txt
   ```

   Or install them individually:

   ```bash
   pip install numpy pandas scipy scikit-learn imbalanced-learn matplotlib seaborn streamlit
   ```

## Usage

### Command-Line Interface

By default, the script is set up to run as a Streamlit application. To run it as a regular script from the command line:

1. **Open the script in a text editor**.

2. **Set `use_streamlit` to `False`** in the `if __name__ == "__main__":` block:

   ```python
   if __name__ == "__main__":
       use_streamlit = False
       # Rest of the code...
   ```

3. **Configure Parameters**:

   Modify the parameters in the `if_running_a_regular_script()` function to suit your dataset and preferences.

4. **Run the script**:

   ```bash
   python your_script_name.py
   ```

### Streamlit Application

To run the script as a Streamlit application:

1. **Ensure `use_streamlit` is set to `True`** in the `if __name__ == "__main__":` block:

   ```python
   if __name__ == "__main__":
       use_streamlit = True
       # Rest of the code...
   ```

2. **Run the Streamlit app**:

   ```bash
   streamlit run your_script_name.py
   ```

3. **Interact with the app**:

   - Adjust the parameters using the provided widgets.
   - Click the **"START THE SYSTEM"** button to run the analysis.
   - View the results and visualizations directly in the browser.

## Data Processing Pipeline

The script processes data through several stages, encapsulated in functions for modularity and clarity.

### Data Extraction

- **Function**: `extract_data()`
- **Purpose**: Reads EEG data, intervals, and game annotations from the specified directory.
- **Process**:
  - Scans the directory for files matching the specified extensions.
  - Reads CSV files containing EEG values and intervals.
  - Reads JSON files containing game annotations.
  - Stores data in lists for further processing.

### Preprocessing

- **Function**: `processing()`
- **Purpose**: Cleans and segments the EEG data based on game intervals.
- **Process**:
  - Extracts relevant columns (amplitude and power) from the EEG data.
  - Filters data within the start and end times of each game.
  - Applies windowing to segment the data into fixed-length windows.

### Feature Extraction

- **Function**: `eeg_feature_extraction()`
- **Purpose**: Computes statistical features from the EEG signals.
- **Features**:
  - **Amplitude Statistics**: Mean and variance for each EEG channel.
  - **Correlation Metrics**: Mean and variance of correlation matrices between windows.
  - **Cross-Covariance Metrics**: Mean and variance of cross-covariance matrices.
  - **Power Spectral Densities**: Mean power in different frequency bands.
- **Process**:
  - Calculates statistics for each windowed segment.
  - Aggregates features across all windows for each game session.

### Machine Learning Models

- **Classification**:
  - **Function**: `classification_prediction_and_evaluation()`
  - **Purpose**: Predicts game outcomes using selected classifiers.
  - **Models**: Support Vector Classifier (SVC), Decision Tree Classifier (DTC), Random Forest Classifier (RFC).
  - **Process**:
    - Splits the data into training and testing sets.
    - Optionally applies scaling and oversampling.
    - Performs hyperparameter tuning with GridSearchCV.
    - Evaluates the model using metrics like accuracy and confusion matrices.

- **Regression**:
  - **Function**: `regression_prediction_and_evaluation()`
  - **Purpose**: Predicts game completion times using selected regressors.
  - **Models**: Theil-Sen Regressor (TSR), Neural Network Regressor (NNR), Gradient Boosting Regressor (GBR).
  - **Process**:
    - Splits the data into training and testing sets.
    - Optionally performs data augmentation.
    - Performs hyperparameter tuning with GridSearchCV.
    - Evaluates the model using metrics like Mean Squared Error (MSE) and R² score.

## Parameters Configuration

Parameters can be set either via the Streamlit interface or by modifying the code when running as a script.

### System Parameters

- **Path**: Directory containing the data files.
- **Number of Runs**: How many times the system will run for evaluation.
- **Window Length**: Duration of each window in seconds.
- **Sampling Frequency (fs)**: The sampling rate of the EEG data.
- **Files Extensions**:
  - CSV values files extension.
  - CSV intervals files extension.
  - JSON annotations files extension.

### Processing Parameters

- **CSV Values**:
  - Columns to Drop: Columns in the EEG data CSV files to be excluded.
  - Timestamps Column Name: Name of the column containing timestamps.
  - Amplitude First Column Name: Name of the first amplitude column.
  - Power First Column Name: Name of the first power column.
- **CSV Intervals**:
  - Timestamps Column Name: Name of the column containing interval timestamps.
- **Game Annotations**:
  - List of annotations to include (e.g., Time, Outcome, Difficulty).

### Prediction Parameters

- **Random State**:
  - Use Fixed Random State: Ensures reproducibility across runs.
  - Use Same Random State for Classifier/Regressor: Uses the same random state for model initialization.
- **Data Handling**:
  - Use Stratify: Maintains the class distribution during splitting.
  - Use Scaling and Oversampling for Classification: Applies standard scaling and SMOTE oversampling.
  - Use Augmentation for Regression: Augments training data by adding noise.
- **Model Selection**:
  - Classifiers: Choose one from SVC, DTC, RFC.
  - Regressors: Choose one from TSR, NNR, GBR.
- **Hyperparameter Tuning**:
  - Use Grid Search for Classifier/Regressor: Enables hyperparameter optimization.
  - Scoring for Grid Search: Selects the metric used for evaluating models during grid search.

## Results Visualization

- **Function**: `metrics_visualization()`
- **Purpose**: Generates plots to visualize the performance of the models.
- **Visualizations**:
  - Classification:
    - Accuracy over runs.
    - Average confusion matrix.
    - Average classification report.
  - Regression:
    - MSE over runs.
    - MAE over runs.
    - R² score over runs.
- **Output**:
  - Saves the plots as PNG files.
  - Displays them directly in the Streamlit app or shows them when running as a script.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

Feel free to explore the code and adjust the parameters to suit your data and research needs. If you encounter any issues or have suggestions for improvement, please open an issue or submit a pull request.
