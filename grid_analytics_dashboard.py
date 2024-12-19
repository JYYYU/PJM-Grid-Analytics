import dash
from dash import dcc, html, Input, Output, State, callback_context
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime
import os
import json
import joblib
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from lightgbm import Booster
import grid_analytics_helper
import pjm_retrieve_data

import logging

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)

MODEL_FOLDER = (
    "./models"  # Folder containing each model folders with the respective saved models
)
EMERGENCY_TRIGGER_FOLDER = "emergency_triggered"  # Model folder for Emergency Triggers
LMP_VOLATILITY_FOLDER = "lmp_volatility"  # Model folder for LMP Volatility
FORCED_OUTAGE_FOLDER = "forced_outages_mw"  # Model folder for Forced Outages
ZONE_TO_REGION = "zone_to_region"  # Custom zone to region mapping
DATAFRAME_FOLDER = (
    "./model_dataframes"  # Dataframe folder containing data used for modelling and EDA
)
NEAR_EMERGENCY = "near_emergency"  # Global variable set for feature 'near_emergency'


def get_pjm_features(
    start_datetime=None,
    end_datetime=None,
    dataframe_file_path=DATAFRAME_FOLDER,
    zone_to_region_name=ZONE_TO_REGION,
    final_df_name="",
):
    """Generate the final merged dataset with engineered features used for modelling, forecasting, or data analysis.

    Args:
        start_datetime (datetime, optional): A datetime object of the start date (Earliest date in datetime_beginning_ept). Defaults to None.
        end_datetime (datetime, optional): A datetime object of the end date; When included with a start_datetime, this datetime will be INCLUSIVE in the data retrieved (Latest date in datetime_ending_ept). Defaults to None.
        dataframe_file_path (str, optional): File path to save parquet file. Defaults to "./dataframes/".
        zone_to_region_name (str, optional): Name of the file containing the mapping between region and zone. Defaults to "zone_to_region".
        final_df_name (str, optional): Name of the final dataframe for saving purposes (Ex. jun_2017_jun_2019_grid_data). Defaults to "".

    Raises:
        ValueError: If there is a zone in the merged dataframe that cannot be mapped to a region. This prompts user to update the zone_to_region mapping.

    Returns:
        str: File name of the new merged dataset that was saved.
    """
    lmp_data = pjm_retrieve_data.PJMMonthlyFTRZonalLmps(
        start_datetime=start_datetime, end_datetime=end_datetime
    ).fetch_data()
    generation_capacity = pjm_retrieve_data.PJMDailyGenerationCapacity(
        start_datetime=start_datetime, end_datetime=end_datetime
    ).fetch_data()
    outage_seven_days = pjm_retrieve_data.PJMGenerationOutageForSevenDays(
        start_datetime=start_datetime, end_datetime=end_datetime
    ).fetch_data()
    zone_to_region = pd.read_parquet(
        os.path.join(dataframe_file_path, f"{zone_to_region_name}.parquet"),
        engine="pyarrow",
    )
    merged_data = grid_analytics_helper.merge_historical_data(
        lmp_df=lmp_data,
        gen_cap_df=generation_capacity,
        outage_df=outage_seven_days,
        zone_to_region_df=zone_to_region,
    )

    mapped_zones = zone_to_region.region.unique()
    merged_data_zones = merged_data.region.unique()
    zones_not_in_mapped = set(merged_data_zones) - set(mapped_zones)

    # Check if there any unmapped regions
    if zones_not_in_mapped:
        raise ValueError(
            f"Zones not in mapped_zones: {zones_not_in_mapped} \nPlease update file: {os.path.join(dataframe_file_path, zone_to_region_name)}"
        )

    final_data = grid_analytics_helper.feature_gen_set_up(merged_data)
    # save data
    start_dte = start_datetime.strftime("%Y%m%d_%H%M%S")
    end_dte = end_datetime.strftime("%Y%m%d_%H%M%S")
    if final_df_name == "":
        file_name = f"{start_dte}_{end_dte}_grid_data"
    else:
        file_name = final_df_name
    file_path = os.path.join(dataframe_file_path, f"{file_name}.parquet")
    final_data.to_parquet(file_path, index=False, engine="pyarrow")
    return file_name


def format_forecast_data(
    preprocess_data,
    predictor,
    y_predicted,
    datetime_col="datetime_beginning_ept",
):
    """Takes the preprocess data for forecasting and creates a new dataframe containing the actual and predicted values, with a region column depending on the predictor.

    Args:
        preprocess_data (pd.DataFrame): A dataframe containing the preprocessed data.
        predictor (str): The predictor feature name in the preprocess data columns.
        y_predicted (np.array): A numpy array containing the predicted results.
        datetime_col (str): Column containing datetime values. Defaults to "datetime_beginning_ept".

    Returns:
        pd.DataFrame: A dataframe with a datetime column containing the actual and predict values in forecasting.
        pd.Series: A series containing the actual values of the predictor or None.
    """
    if predictor in preprocess_data.columns:
        y_actual = preprocess_data[predictor]
        final_df = pd.DataFrame(y_actual)
    else:
        y_actual = None
        final_df = pd.DataFrame()
        final_df[datetime_col] = preprocess_data.index

    final_df["y_predicted"] = y_predicted

    if "region_catWestern" in preprocess_data.columns:
        final_df["region"] = preprocess_data["region_catWestern"].apply(
            lambda x: "Western" if x == 0 else "Mid Atlantic - Dominion"
        )
    if predictor == LMP_VOLATILITY_FOLDER:
        final_df["pnode_id"] = preprocess_data["pnode_id"]
    final_df.reset_index(inplace=True)
    return final_df, y_actual


def plot_interactive_timeseries_dash(
    data,
    feature,
    feature_units,
    region_filter=True,
    region_col="region",
    region="All",
    pnode_filter=True,
    pnode_col="pnode_id",
    pnode="All",
    datetime_col="datetime_beginning_ept",
):
    """Interactive Time Series Plot that allows for region filtering for visual evaluation of comparing predicted value vs. actual value.

    Args:
        data (pd.DataFrame): Input data containing the time series.
        feature (str): Column name for the feature to plot.
        feature_units (str): Units in which the plotted feature is in.
        region_filter (bool, optional): Whether to include region-specific filtering. Defaults to True.
        region_col (str): Column representing regions. Defaults to "region".
        region (str): Determine what region to filter for. Defaults to "All".
        region_filter (bool, optional): Whether to include pricing node specific filtering. Defaults to True.
        pnode_col (str): Column representing pricing node ids. Defaults to "pnode_id".
        pnode (str): Determine what pricing node to filter for. Defaults to "All".
        datetime_col (str): Column containing datetime values. Defaults to "datetime_beginning_ept".

    Raises:
        ValueError: Region feature name is not in the dataset.
        ValueError: The feature to plot is not in the dataset.

    Returns:
        plotly.graph_objsFigure: A Plotly Figure object representing the interactive time series plot.
    """
    if region_filter and region_col not in data.columns:
        raise ValueError(f"'{region_col}' column must exist in the data.")
    if feature not in data.columns:
        raise ValueError(f"'{feature}' column must exist in the data.")

    filtered_data = data.copy()
    if region_filter and region != "All":
        filtered_data = filtered_data[filtered_data[region_col] == region]

    if pnode_filter and pnode != "All":
        filtered_data = filtered_data[filtered_data[pnode_col] == pnode]

    fig = go.Figure()
    if not filtered_data.empty:
        if region_filter and region != "All":
            for region, group in filtered_data.groupby(region_col):
                if feature in group.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=group[datetime_col],
                            y=group[feature],
                            mode="lines",
                            name=f"{region} - Actual",
                        )
                    )
                if "y_predicted" in group.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=group[datetime_col],
                            y=group["y_predicted"],
                            mode="lines",
                            name=f"{region} - Predicted",
                        )
                    )
        else:
            if feature in filtered_data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=filtered_data[datetime_col],
                        y=filtered_data[feature],
                        mode="lines",
                        name="Actual",
                    )
                )
            if "y_predicted" in filtered_data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=filtered_data[datetime_col],
                        y=filtered_data["y_predicted"],
                        mode="lines",
                        name="Predicted",
                    )
                )

    _title = f"{feature.replace('_', ' ').title()} Timeseries"
    if region_filter and region != "All":
        _title += f" (Region: {region})"
    if pnode_filter and pnode != "All":
        _title += f" (Pricing Node: {pnode})"

    fig.update_layout(
        title=_title,
        xaxis_title="Datetime",
        yaxis_title=f"{feature_units}",
        template="plotly_white",
    )
    return fig


def plot_interactive_rolling_timeseries_dash(
    data,
    feature,
    feature_units,
    rolling_window,
    region_filter=True,
    region_col="region",
    region="All",
    pnode_filter=True,
    pnode_col="pnode_id",
    pnode="All",
    datetime_col="datetime_beginning_ept",
):
    """Interactive Time Series Plot that allows for region filtering for visual evaluation of comparing predicted value vs. actual value.

    Args:
        data (pd.DataFrame): Input data containing the time series.
        feature (str): Column name for the feature to plot.
        feature_units (str): Units in which the plotted feature is in.
        rolling_window (int): Size of the rolling window in days.
        region_filter (bool, optional): Whether to include region-specific filtering. Defaults to True.
        region_col (str): Column representing regions. Defaults to "region".
        region (str): Determine what region to filter for. Defaults to "All".
        region_filter (bool, optional): Whether to include pricing node specific filtering. Defaults to True.
        pnode_col (str): Column representing pricing node ids. Defaults to "pnode_id".
        pnode (str): Determine what pricing node to filter for. Defaults to "All".
        datetime_col (str): Column containing datetime values. Defaults to "datetime_beginning_ept".

    Raises:
        ValueError: Region feature name is not in the dataset.
        ValueError: The feature to plot is not in the dataset.

    Returns:
        plotly.graph_objsFigure: A Plotly Figure object representing the interactive time series plot.
    """
    if region_filter and region_col not in data.columns:
        raise ValueError(f"'{region_col}' column must exist in the data.")
    if feature not in data.columns:
        raise ValueError(f"'{feature}' column must exist in the data.")

    filtered_data = data.copy()
    if region_filter and region != "All":
        filtered_data = filtered_data[filtered_data[region_col] == region]

    if pnode_filter and pnode != "All":
        filtered_data = filtered_data[filtered_data[pnode_col] == pnode]

    filtered_data = filtered_data.sort_values(datetime_col)
    filtered_data[f"{feature}_rolling"] = (
        filtered_data[feature].rolling(window=rolling_window, min_periods=1).mean()
    )

    fig = go.Figure()
    if not filtered_data.empty:
        fig.add_trace(
            go.Scatter(
                x=filtered_data[datetime_col],
                y=filtered_data[f"{feature}_rolling"],
                mode="lines",
                name=f"{feature.replace('_', ' ').title()} (Rolling {rolling_window}-Day Avg)",
            )
        )
    fig.update_layout(
        title=f"Rolling {rolling_window}-Day Average Timeseries",
        xaxis_title="Datetime",
        yaxis_title=f"{feature_units}",
        legend_title="Features",
        template="plotly_white",
    )

    return fig


def list_available_datasets(folder_path):
    """Lists all available parquet files in the specified folder.

    Args:
        folder_path (str): File path in which all parquet files within will be accesssed

    Returns:
        list of dict: List containing the file names of all parquet files in the folder_path and its respective file paths.
    """
    try:
        datasets = [
            {"label": f, "value": os.path.join(folder_path, f)}
            for f in os.listdir(folder_path)
            if f.endswith(".parquet") and f != (ZONE_TO_REGION + ".parquet")
        ]
        if not datasets:
            logging.debug("No .parquet files found in the folder.")
        return datasets
    except Exception as e:
        logging.error(f"Error listing datasets: {str(e)}")
        return []


def load_model_metadata(model_folder_name):
    """Retrieves/loads the save models metadata from model_folder.

    Args:
        model_folder_name (str): File name containing the saved metadata.

    Returns:
        list of dict: Each dict containing the metadata for a specific model type.
    """
    metadata = []
    model_folder = os.path.join(MODEL_FOLDER, model_folder_name)
    for file in os.listdir(model_folder):
        if file.endswith(".json"):
            model_name = " ".join(file.split("_")[:2]).title()
            with open(os.path.join(model_folder, file), "r") as f:
                data = json.load(f)
                data["model_name"] = model_name
                metadata.append(data)
    return metadata


def add_seasons(
    data,
    datetime_col="datetime_beginning_ept",
):
    """Adds a season column to the DataFrame based on the month column.

    Args:
        data (pd.DataFrame): DataFrame containing a `month` column.

    Returns:
        pd.DataFrame: Updated DataFrame with a `season` column.
    """
    df = data.copy()
    df.reset_index(inplace=True)
    df[datetime_col] = pd.to_datetime(df[datetime_col])
    df["month"] = df[datetime_col].dt.month
    month_to_season = {
        12: "Winter",
        1: "Winter",
        2: "Winter",
        6: "Summer",
        7: "Summer",
        8: "Summer",
    }
    df["season"] = df["month"].map(month_to_season).fillna("Spring & Fall")
    df.drop(columns=["month"], inplace=True)
    df.set_index(datetime_col, inplace=True)
    return df


def preprocess_lmp_data(dataset_path):
    """Preprocesses LMP Volatility data for EDA.

    Args:
        dataset_path (str): Path to grab the specified dataset by user.

    Returns:
        pd.DataFrame: A dataset for LMP Volatility EDA.
    """
    data = pd.read_parquet(dataset_path, engine="pyarrow")
    data = grid_analytics_helper.lmp_volatility_set_up(data)
    if "season" not in data.columns:
        data = add_seasons(data)
    if "region_catWestern" in data.columns:
        data["region"] = data["region_catWestern"].apply(
            lambda x: "Western" if x == 0 else "Mid Atlantic - Dominion"
        )
        data.drop(columns=["region_catWestern"], inplace=True)
    data.reset_index(inplace=True)
    return data


def preprocess_fo_data(dataset_path):
    """Preprocesses Forced Outages data for EDA.

    Args:
        dataset_path (str): Path to grab the specified dataset by user.

    Returns:
        pd.DataFrame: A dataset for Forced Outages Percentage EDA.
    """
    data = pd.read_parquet(dataset_path, engine="pyarrow")
    data = grid_analytics_helper.handle_forced_outage_pct_data(data)
    if "season" not in data.columns:
        data = add_seasons(data)
    data.reset_index(inplace=True)
    try:
        data.drop(columns=["index"], inplace=True)
    finally:
        return data


def preprocess_calendar_data(data, datetime_col="datetime_beginning_ept"):
    """Preprocesses data for the calendar heatmap.

    Args:
        data (pd.DataFrame): Processed Forced Outage Percentage DataFrame to be reprocessed for the calendar heatmap.
        datetime_col (str, optional): Column containing datetime values. Defaults to "datetime_beginning_ept".

    Returns:
        pd.DataFrame: DataFrame for calendar heatmap plotting.
    """

    data[datetime_col] = pd.to_datetime(data[datetime_col])
    data["year"] = data[datetime_col].dt.year
    data["month"] = data[datetime_col].dt.month
    data["day"] = data[datetime_col].dt.day
    if "season" not in data.columns:
        data = add_seasons(data)
    return data


def preprocess_trigger_data(dataset_path):
    """Preprocesses Emergency Trigger data for EDA.

    Args:
        dataset_path (str): Path to grab the specified dataset by user.

    Returns:
        pd.DataFrame: A dataset for Emergency Trigger EDA.
    """
    data = pd.read_parquet(dataset_path, engine="pyarrow")
    et_data = grid_analytics_helper.emergency_trigger_set_up(data)
    month_to_season = {
        1: "Winter",
        3: "Summer",
    }
    et_data["season"] = et_data["season"].map(month_to_season).fillna("Spring & Fall")

    # Handle near_emergency
    ne_data = (
        data.groupby(["datetime_beginning_ept"]).mean(numeric_only=True).reset_index()
    ).copy()
    ne_data.set_index("datetime_beginning_ept", inplace=True)
    data = pd.merge(et_data, ne_data[NEAR_EMERGENCY], left_index=True, right_index=True)
    data.reset_index(inplace=True)
    return data


app = dash.Dash(__name__, suppress_callback_exceptions=True)

app.layout = html.Div(
    [
        dcc.Tabs(
            [
                dcc.Tab(
                    label="Model Evaluation & Training",
                    children=[
                        html.Div(
                            [
                                html.H1(
                                    "Evaluate Existing Models:",
                                    style={"textDecoration": "underline"},
                                ),
                                dcc.Dropdown(
                                    id="prediction-type-selection-setup",
                                    options=[
                                        {
                                            "label": "Emergency Trigger",
                                            "value": EMERGENCY_TRIGGER_FOLDER,
                                        },
                                        {
                                            "label": "LMP Volatility",
                                            "value": LMP_VOLATILITY_FOLDER,
                                        },
                                        {
                                            "label": "Forced Outages",
                                            "value": FORCED_OUTAGE_FOLDER,
                                        },
                                    ],
                                    placeholder="Select a prediction 'feature'.",
                                    style={"width": "50%"},
                                ),
                                html.Div(
                                    [
                                        dcc.Dropdown(
                                            id="model-selection-setup",
                                            placeholder="Select a trained model",
                                            style={"width": "50%"},
                                        )
                                    ],
                                    id="model-dropdown-container-setup",
                                    style={"marginTop": "20px"},
                                ),
                                dcc.Store(id="stored-metadata-setup"),
                                html.Div(
                                    id="model-setup-metadata-display",
                                    style={"marginTop": "20px"},
                                ),
                                ################################################################################
                                html.Hr(),
                                html.H1(
                                    "Generate New Data:",
                                    style={"textDecoration": "underline"},
                                ),
                                html.P(
                                    "Specify the Start and End Dates, including timestamps, to generate new data. The Start Date will be the earliest date in datetime_beginning_ept and the End Date will be the latest date in datetime_ending_ept.",
                                    style={"marginBottom": "20px"},
                                ),
                                html.Div(
                                    [
                                        html.Label(
                                            "Start Date:", style={"marginRight": "10px"}
                                        ),
                                        dcc.DatePickerSingle(
                                            id="start-date-picker-setup",
                                            placeholder="Start Date",
                                            style={
                                                "marginRight": "10px",
                                                "height": "40px",
                                            },
                                        ),
                                        dcc.Input(
                                            id="start-time-picker-setup",
                                            type="text",
                                            placeholder="HH:MM:SS",
                                            style={
                                                "width": "100px",
                                                "marginRight": "20px",
                                                "height": "40px",
                                                "marginTop": "10px",
                                            },
                                        ),
                                        html.Label(
                                            "End Date:", style={"marginRight": "10px"}
                                        ),
                                        dcc.DatePickerSingle(
                                            id="end-date-picker-setup",
                                            placeholder="End Date",
                                            style={
                                                "marginRight": "10px",
                                                "height": "40px",
                                            },
                                        ),
                                        dcc.Input(
                                            id="end-time-picker-pickup",
                                            type="text",
                                            placeholder="HH:MM:SS",
                                            style={
                                                "width": "100px",
                                                "marginRight": "20px",
                                                "height": "40px",
                                                "marginTop": "10px",
                                            },
                                        ),
                                        html.Button(
                                            "Generate Data",
                                            id="generate-data-button-setup",
                                            style={
                                                "height": "40px",
                                                "marginTop": "8px",
                                            },
                                        ),
                                    ],
                                    style={
                                        "display": "flex",
                                        "alignItems": "center",
                                        "marginBottom": "10px",
                                    },
                                ),
                                html.Div(
                                    id="data-generation-status-setup",
                                    style={"marginTop": "10px"},
                                ),
                                ################################################################################
                                html.Hr(),
                                html.Div(
                                    [
                                        html.H1(
                                            "Train New Models:",
                                            style={
                                                "marginRight": "20px",
                                                "display": "inline",
                                                "textDecoration": "underline",
                                            },
                                        ),
                                        html.Button(
                                            "Refresh Dataset Selection",
                                            id="refresh-dataset-button-build-models-setup",
                                            style={
                                                "height": "40px",
                                                "marginLeft": "0px",
                                            },
                                        ),
                                    ],
                                    style={
                                        "display": "inline-flex",
                                        "alignItems": "center",
                                    },
                                ),
                                html.P(
                                    "Press the 'Refresh Dataset Selection' button and choose a dataset for you want to build models with.",
                                    style={"marginBottom": "20px"},
                                ),
                                dcc.Dropdown(
                                    id="build-models-dataset-selection-setup",
                                    placeholder="Select dataset",
                                    style={"width": "50%", "marginRight": "0px"},
                                ),
                                html.P(
                                    "After selecting a dataset above, select predictors and model types to build new models."
                                ),
                                html.Div(
                                    [
                                        html.Div(
                                            [
                                                html.Label(
                                                    "Available Predictors:",
                                                    style={
                                                        "fontWeight": "bold",
                                                        "marginBottom": "10px",
                                                    },
                                                ),
                                                dcc.Checklist(
                                                    id="build_models_predictors-selection-setup",
                                                    options=[
                                                        {
                                                            "label": "Emergency Trigger",
                                                            "value": "EMERGENCY_TRIGGER_FOLDER",
                                                        },
                                                        {
                                                            "label": "LMP Volatility",
                                                            "value": "LMP_VOLATILITY_FOLDER",
                                                        },
                                                        {
                                                            "label": "Forced Outages",
                                                            "value": "FORCED_OUTAGE_FOLDER",
                                                        },
                                                    ],
                                                ),
                                            ],
                                            style={
                                                "marginRight": "20px",
                                                "width": "30%",
                                            },
                                        ),
                                        html.Div(
                                            [
                                                html.Label(
                                                    "Available Models:",
                                                    style={
                                                        "fontWeight": "bold",
                                                        "marginBottom": "10px",
                                                    },
                                                ),
                                                dcc.Checklist(
                                                    id="models-selection-setup",
                                                    options=[
                                                        {
                                                            "label": "Decision Tree",
                                                            "value": "decision_tree",
                                                        },
                                                        {
                                                            "label": "Random Forest",
                                                            "value": "random_forest",
                                                        },
                                                        {
                                                            "label": "Gradient Boosting (LightGBM)",
                                                            "value": "light_gbm",
                                                        },
                                                    ],
                                                ),
                                                html.P(
                                                    "* Note: Random Forest cannot be used with LMP Volatility due to computational limitations.",
                                                    style={
                                                        "fontStyle": "italic",
                                                        "color": "red",
                                                        "fontSize": "12px",
                                                    },
                                                ),
                                            ],
                                            style={
                                                "marginRight": "20px",
                                                "width": "30%",
                                            },
                                        ),
                                        html.Div(
                                            [
                                                html.Button(
                                                    "Build Models",
                                                    id="build-models-button-setup",
                                                    style={"marginTop": "20px"},
                                                ),
                                            ],
                                            style={"width": "20%"},
                                        ),
                                    ],
                                    style={
                                        "display": "flex",
                                        "alignItems": "flex-start",
                                        "marginBottom": "20px",
                                    },
                                ),
                                html.Div(
                                    id="model-building-status-setup",
                                    style={"marginTop": "20px"},
                                ),
                            ]
                        )
                    ],
                ),
                dcc.Tab(
                    label="Model Forecasting (Visualizations)",
                    children=[
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.H1(
                                            "Model Forecasting:",
                                            style={"textDecoration": "underline"},
                                        ),
                                        html.Button(
                                            "Refresh Dataset Selection",
                                            id="refresh-dataset-button-forecast-models",
                                            style={
                                                "height": "40px",
                                                "marginLeft": "10px",
                                            },
                                        ),
                                    ],
                                    style={
                                        "display": "inline-flex",
                                        "alignItems": "center",
                                    },
                                ),
                                html.P(
                                    "Press the 'Refresh Dataset Selection' button and choose a dataset for you want to work with.",
                                    style={"marginBottom": "20px"},
                                ),
                                html.Label(
                                    "Dataset Selection:",
                                    style={
                                        "fontWeight": "bold",
                                        "marginBottom": "15px",
                                    },
                                ),
                                dcc.Dropdown(
                                    id="forecast-models-dataset-selection",
                                    placeholder="Select dataset",
                                    style={"width": "67%", "marginBottom": "20px"},
                                ),
                                html.Label(
                                    "Select a Predictor:",
                                    style={
                                        "fontWeight": "bold",
                                        "marginBottom": "15px",
                                    },
                                ),
                                dcc.Dropdown(
                                    id="forecast-predictor-selection",
                                    options=[
                                        {
                                            "label": "Emergency Trigger",
                                            "value": EMERGENCY_TRIGGER_FOLDER,
                                        },
                                        {
                                            "label": "LMP Volatility",
                                            "value": LMP_VOLATILITY_FOLDER,
                                        },
                                        {
                                            "label": "Forced Outages",
                                            "value": FORCED_OUTAGE_FOLDER,
                                        },
                                    ],
                                    placeholder="Select a predictor",
                                    style={"width": "67%", "marginBottom": "20px"},
                                ),
                                html.Div(
                                    [
                                        html.Div(
                                            [
                                                html.Div(
                                                    [
                                                        html.Label(
                                                            "Select a Trained Model:",
                                                            style={
                                                                "fontWeight": "bold",
                                                                "marginBottom": "5px",
                                                            },
                                                        ),
                                                        dcc.Dropdown(
                                                            id="forecast-model-selection",
                                                            placeholder="Select a trained model",
                                                            style={"width": "100%"},
                                                        ),
                                                    ],
                                                    style={
                                                        "width": "45%",
                                                        "marginRight": "5px",
                                                    },
                                                ),
                                                html.Div(
                                                    [
                                                        html.Label(
                                                            "Select a Region:",
                                                            style={
                                                                "fontWeight": "bold",
                                                                "marginBottom": "5px",
                                                            },
                                                        ),
                                                        dcc.Dropdown(
                                                            id="region-selection-forecast",
                                                            placeholder="Select a region",
                                                            style={
                                                                "width": "100%",
                                                                "minWidth": "500px",
                                                            },
                                                        ),
                                                    ],
                                                    id="region-dropdown-container-forecast",
                                                    style={
                                                        "width": "45%",
                                                        "visibility": "hidden",
                                                    },
                                                ),
                                                html.Div(
                                                    [
                                                        html.Label(
                                                            "Select a Pricing Node ID:",
                                                            style={
                                                                "fontWeight": "bold",
                                                                "marginBottom": "5px",
                                                            },
                                                        ),
                                                        dcc.Dropdown(
                                                            id="pnode-id-selection-forecast",
                                                            placeholder="Select a PNode ID",
                                                            style={
                                                                "width": "100%",
                                                                "minWidth": "500px",
                                                            },
                                                        ),
                                                    ],
                                                    id="pnode-dropdown-container-forecast",
                                                    style={
                                                        "width": "45%",
                                                        "visibility": "hidden",
                                                    },
                                                ),
                                            ],
                                            style={
                                                "display": "flex",
                                                "alignItems": "flex-start",
                                                "justifyContent": "flex-start",
                                                "marginTop": "20px",
                                            },
                                        ),
                                    ],
                                    id="model-dropdown-container-forecast",
                                    style={"marginTop": "20px"},
                                ),
                                html.Button(
                                    "View Model Predictions",
                                    id="view-forecast-model-button",
                                    style={"marginTop": "10px"},
                                ),
                                dcc.Graph(
                                    id="forecast-graph", style={"marginTop": "20px"}
                                ),
                                html.Div(
                                    [
                                        html.Div(
                                            id="forecast-risk-metrics-display",
                                            style={"marginTop": "20px", "width": "50%"},
                                        ),
                                        html.Div(
                                            id="forecast-feature-importance-display",
                                            style={"marginTop": "20px", "width": "50%"},
                                        ),
                                    ],
                                    style={
                                        "display": "flex",
                                        "justifyContent": "space-between",
                                    },
                                ),
                            ],
                        )
                    ],
                ),
                dcc.Tab(
                    label="LMP Volatility (EDA)",
                    children=[
                        html.Div(
                            [
                                html.H1(
                                    "LMP Volatility Exploratory Data Analysis:",
                                    style={"textDecoration": "underline"},
                                ),
                                dcc.Markdown("""
                     LMP Volatility is calculated using a rolling standard deviation over a 24-hour period, grouped by individual pricing nodes. This metric provides a measure of the variability in Locational Marginal Prices (LMPs) and can help identify areas of price instability within specific regions or nodes of the grid.
                     """),
                                html.Div(
                                    [
                                        html.P(
                                            "Press the 'Refresh Dataset Selection' button and choose a dataset for to explore.",
                                            style={"marginBottom": "20px"},
                                        ),
                                        html.Button(
                                            "Refresh Dataset Selection",
                                            id="refresh-dataset-button-lmp",
                                            style={
                                                "height": "40px",
                                                "marginLeft": "10px",
                                            },
                                        ),
                                    ],
                                    style={
                                        "display": "inline-flex",
                                        "alignItems": "center",
                                    },
                                ),
                                html.Div(
                                    [
                                        html.Label(
                                            "Dataset Selection:",
                                            style={
                                                "fontWeight": "bold",
                                                "marginBottom": "15px",
                                            },
                                        ),
                                        dcc.Dropdown(
                                            id="lmp-dataset-selection",
                                            placeholder="Select dataset",
                                            style={
                                                "width": "57%",
                                                "marginBottom": "20px",
                                            },
                                        ),
                                    ]
                                ),
                                # Filters
                                html.Div(
                                    [
                                        html.Div(
                                            [
                                                html.Label(
                                                    "Select Season:",
                                                    style={"fontWeight": "bold"},
                                                ),
                                                dcc.Dropdown(
                                                    id="lmp-season-filter",
                                                    placeholder="Select a Season",
                                                    options=[
                                                        {"label": "All", "value": "All"}
                                                    ],
                                                    value="All",
                                                ),
                                            ],
                                            style={
                                                "width": "33%",
                                                "marginRight": "10px",
                                            },
                                        ),
                                        html.Div(
                                            [
                                                html.Label(
                                                    "Region Filter:",
                                                    style={"fontWeight": "bold"},
                                                ),
                                                dcc.Dropdown(
                                                    id="lmp-region-filter",
                                                    placeholder="Select a Region",
                                                    options=[
                                                        {"label": "All", "value": "All"}
                                                    ],
                                                    value="All",
                                                    style={"width": "100%"},
                                                ),
                                            ],
                                            style={
                                                "width": "33%",
                                                "marginRight": "10px",
                                            },
                                        ),
                                        html.Div(
                                            [
                                                html.Label(
                                                    "Pricing Node Filter:",
                                                    style={"fontWeight": "bold"},
                                                ),
                                                dcc.Dropdown(
                                                    id="lmp-pnode-filter",
                                                    placeholder="Select a Pricing Node",
                                                    options=[
                                                        {"label": "All", "value": "All"}
                                                    ],
                                                    value="All",
                                                    style={"width": "100%"},
                                                ),
                                            ],
                                            style={
                                                "width": "33%",
                                                "marginRight": "10px",
                                            },
                                        ),
                                    ],
                                    style={
                                        "display": "flex",
                                        "justifyContent": "space-between",
                                        "marginBottom": "20px",
                                    },
                                ),
                                dcc.Tabs(
                                    id="eda-tabs-lmp",
                                    value="histogram-tab-lmp",
                                    children=[
                                        dcc.Tab(
                                            label="Histogram", value="histogram-tab-lmp"
                                        ),
                                        dcc.Tab(
                                            label="Boxplot", value="boxplot-tab-lmp"
                                        ),
                                        dcc.Tab(
                                            label="Timeseries",
                                            value="timeseries-tab-lmp",
                                        ),
                                        dcc.Tab(
                                            label="Rolling-Timeseries",
                                            value="rolling-timeseries-tab-lmp",
                                        ),
                                        dcc.Tab(
                                            label="Scatterplot (Outliers)",
                                            value="scatterplot-tab-lmp",
                                        ),
                                        dcc.Tab(
                                            label="Outlier PNode Table",
                                            value="pnode-outlier-tab-lmp",
                                        ),
                                    ],
                                ),
                                html.Div(id="eda-tab-content-lmp"),
                            ]
                        )
                    ],
                ),
                dcc.Tab(
                    label="Forced Outages Percentage (EDA)",
                    children=[
                        html.Div(
                            [
                                html.H1(
                                    "Forced Outage Percentage Exploratory Data Analysis",
                                    style={"textDecoration": "underline"},
                                ),
                                dcc.Markdown(
                                    """
            Forced Outage Percentage measures the proportion of outages that occur due to unplanned (forced) events, reflecting the reliability and operational stability of generation assets. It is an important metric in understanding how often and to what extent power plants or other energy resources experience unexpected downtime, often due to mechanical failures, equipment malfunctions, or other unforeseen circumstances.
                     
            **Higher Fourced Outage Percentage** indicates greater system stress or unexpected maintenance issues.

            **Formula:** 
            """
                                ),
                                dcc.Markdown(
                                    """
            $$
            \\text{Forced Outage Percentage} = \\frac{\\text{Forced Outages (MW)}}{\\text{Total Outages (MW)}} \\times 100 
            $$
            """,
                                    mathjax=True,
                                ),
                                html.Div(
                                    [
                                        html.P(
                                            "Press the 'Refresh Dataset Selection' button and choose a dataset for to explore.",
                                            style={"marginBottom": "20px"},
                                        ),
                                        html.Button(
                                            "Refresh Dataset Selection",
                                            id="refresh-dataset-button-fo",
                                            style={
                                                "height": "40px",
                                                "marginLeft": "10px",
                                            },
                                        ),
                                    ],
                                    style={
                                        "display": "inline-flex",
                                        "alignItems": "center",
                                    },
                                ),
                                html.Div(
                                    [
                                        html.Label(
                                            "Dataset Selection:",
                                            style={
                                                "fontWeight": "bold",
                                                "marginTop": "10px",
                                                "marginBottom": "15px",
                                            },
                                        ),
                                        dcc.Dropdown(
                                            id="fo-dataset-selection",
                                            placeholder="Select dataset",
                                            style={
                                                "width": "70.2%",
                                                "marginBottom": "20px",
                                            },
                                        ),
                                    ]
                                ),
                                html.Div(
                                    [
                                        html.Div(
                                            [
                                                html.Label(
                                                    "Select Season:",
                                                    style={"fontWeight": "bold"},
                                                ),
                                                dcc.Dropdown(
                                                    id="fo-season-filter",
                                                    placeholder="Select a Season",
                                                    options=[
                                                        {"label": "All", "value": "All"}
                                                    ],
                                                    value="All",
                                                ),
                                            ],
                                            style={
                                                "width": "50%",
                                                "marginRight": "10px",
                                            },
                                        ),
                                        html.Div(
                                            [
                                                html.Label(
                                                    "Region Filter:",
                                                    style={"fontWeight": "bold"},
                                                ),
                                                dcc.Dropdown(
                                                    id="fo-region-filter",
                                                    placeholder="Select a Region",
                                                    options=[
                                                        {"label": "All", "value": "All"}
                                                    ],
                                                    value="All",
                                                    style={"width": "100%"},
                                                ),
                                            ],
                                            style={
                                                "width": "50%",
                                                "marginRight": "10px",
                                            },
                                        ),
                                    ],
                                    style={
                                        "display": "flex",
                                        "justifyContent": "space-between",
                                        "marginBottom": "20px",
                                    },
                                ),
                                dcc.Tabs(
                                    id="eda-tabs-fo",
                                    value="histogram-tab-fo",
                                    children=[
                                        dcc.Tab(
                                            label="Histogram", value="histogram-tab-fo"
                                        ),
                                        dcc.Tab(
                                            label="Boxplot", value="boxplot-tab-fo"
                                        ),
                                        dcc.Tab(
                                            label="Timeseries",
                                            value="timeseries-tab-fo",
                                        ),
                                        dcc.Tab(
                                            label="Rolling-Timeseries",
                                            value="rolling-timeseries-tab-fo",
                                        ),
                                        dcc.Tab(
                                            label="Calendar Heatmap",
                                            value="calendar-heatmap-tab-fo",
                                        ),
                                    ],
                                ),
                                html.Div(id="eda-tab-content-fo"),
                            ]
                        )
                    ],
                ),
                dcc.Tab(
                    label="Emergency Trigger (EDA)",
                    children=[
                        html.Div(
                            [
                                html.H1(
                                    "Emergency Triggers Exploratory Data Analysis:",
                                    style={"textDecoration": "underline"},
                                ),
                                dcc.Markdown(
                                    """
            Emergency Trigger is a binary feature that indicates whether the total committed capacity has exceeded the Economic Max in the energy market. When this occurs, it signals that demand has spiked above typical market conditions, potentially requiring the use of additional, often more expensive or less efficient, emergency resources to meet that demand. This feature helps identify when the market is reliant on emergency capacity, signaling potential system stress or high-demand conditions.

            **Formula:** 
            """
                                ),
                                dcc.Markdown(
                                    """
            $$
            \\text{Emergency Triggered} = \\text{Total Committed} \\gt \\text{Economic Max}
            $$
            """,
                                    mathjax=True,
                                ),
                                html.Div(
                                    [
                                        html.P(
                                            "Press the 'Refresh Dataset Selection' button and choose a dataset for to explore.",
                                            style={"marginBottom": "20px"},
                                        ),
                                        html.Button(
                                            "Refresh Dataset Selection",
                                            id="refresh-dataset-button-et",
                                            style={
                                                "height": "40px",
                                                "marginLeft": "10px",
                                            },
                                        ),
                                    ],
                                    style={
                                        "display": "inline-flex",
                                        "alignItems": "center",
                                    },
                                ),
                                html.P(
                                    "Note: This feature is hourly granular and is not region specific.",
                                    style={
                                        "marginBottom": "20px",
                                        "fontWeight": "bold",
                                    },
                                ),
                                html.Div(
                                    [
                                        html.Div(
                                            [
                                                html.Label(
                                                    "Dataset Selection:",
                                                    style={
                                                        "fontWeight": "bold",
                                                    },
                                                ),
                                                dcc.Dropdown(
                                                    id="et-dataset-selection",
                                                    placeholder="Select dataset",
                                                    style={"width": "100%"},
                                                ),
                                            ],
                                            style={
                                                "width": "50%",
                                                "marginRight": "10px",
                                            },
                                        ),
                                        html.Div(
                                            [
                                                html.Label(
                                                    "Select Season:",
                                                    style={"fontWeight": "bold"},
                                                ),
                                                dcc.Dropdown(
                                                    id="et-season-filter",
                                                    placeholder="Select a Season",
                                                    options=[
                                                        {"label": "All", "value": "All"}
                                                    ],
                                                    value="All",
                                                    style={"width": "100%"},
                                                ),
                                            ],
                                            style={
                                                "width": "50%",
                                                "marginRight": "10px",
                                            },
                                        ),
                                    ],
                                    style={
                                        "display": "flex",
                                        "justifyContent": "space-between",
                                        "marginBottom": "20px",
                                    },
                                ),
                                dcc.Tabs(
                                    id="eda-tabs-et",
                                    value="barchart-tab-et",
                                    children=[
                                        dcc.Tab(
                                            label="Barchart", value="barchart-tab-et"
                                        ),
                                        dcc.Tab(
                                            label="Frequency Timeseries",
                                            value="freq-ts-tab-et",
                                        ),
                                    ],
                                ),
                                html.Div(id="eda-tab-content-et"),
                            ],
                        )
                    ],
                ),
            ],
        ),
    ]
)


################################################################################################################################
# Set-up


@app.callback(
    [
        Output("model-dropdown-container-setup", "children"),
        Output("stored-metadata-setup", "data"),
    ],
    Input("prediction-type-selection-setup", "value"),
)
def update_metadata_and_models(prediction_type):
    logging.debug(
        f"Triggered update_metadata_and_models with prediction_type={prediction_type}"
    )

    ctx = callback_context
    triggered_inputs = [trigger["prop_id"] for trigger in ctx.triggered]
    logging.debug(f"Triggered inputs: {triggered_inputs}")
    logging.debug(f"ctx states: {ctx.states}")

    if not ctx.triggered:  # For dashboard start-up
        # logging.info("App startup: No user interaction yet.")
        return dash.no_update, dash.no_update

    if not prediction_type:
        return None, html.P(
            "Please select a prediction type to load models.",
            style={"fontStyle": "italic"},
        )

    metadata = load_model_metadata(prediction_type)
    if not metadata:
        return None, html.P(
            "No models found for the selected prediction type.",
            style={"color": "red"},
        )

    metric_key = (
        "test_f1_score" if prediction_type == EMERGENCY_TRIGGER_FOLDER else "test_rmse"
    )
    metric_name = "F1 Score" if prediction_type == EMERGENCY_TRIGGER_FOLDER else "RMSE"

    data_to_store = {
        "metadata": metadata,
        "metric_key": metric_key,
        "metric_name": metric_name,
        "prediction_type": prediction_type,
    }

    model_dropdown = dcc.Dropdown(
        id="model-selection-setup",
        options=[
            {
                "label": f"{meta['model_name']} (Trained on: {datetime.strptime(meta['training_time'], '%Y%m%d_%H%M%S').strftime('%Y-%m-%d %H:%M:%S')}, {metric_name}: {meta[metric_key]:.3f})",
                "value": meta["save_path"],
            }
            for meta in metadata
        ],
        placeholder="Select a trained model",
        style={"width": "50%"},
    )
    return model_dropdown, data_to_store


@app.callback(
    Output("model-setup-metadata-display", "children"),
    [Input("stored-metadata-setup", "data"), Input("model-selection-setup", "value")],
    prevent_initial_call=True,
)
def display_metadata(stored_data, model_selected):
    ctx = callback_context
    logging.debug(f"Callback context triggered inputs: {ctx.triggered}")
    # logging.debug(f"Stored metadata: {stored_data}")
    # logging.debug(f"Model selected value: {model_selected}")

    if not stored_data:
        logging.debug("No stored data available.")
        return html.P("No metadata available. Please select a prediction type.")

    if not model_selected:
        logging.debug("No model selected.")
        return html.P("Please select a model to view metadata.")

    metadata = stored_data.get("metadata", [])
    metric_key = stored_data.get("metric_key", "Unknown Key")
    metric_name = stored_data.get("metric_name", "Unknown Metric")
    prediction_type = stored_data.get("prediction_type", "Unknown Prediction Type")

    best_model = max(metadata, key=lambda x: x.get(metric_key, float("-inf")))

    selected_metadata = next(
        (m for m in metadata if m["save_path"] == model_selected), None
    )

    if not selected_metadata:
        logging.debug("Selected model not found in metadata.")
        return html.P("Metadata for selected model not found.")
    logging.debug(f"Selected metadata: {selected_metadata}")

    return html.Div(
        [
            html.H4(
                f"The Best Model Overall for '{prediction_type.replace("_", " ").title()}' is {best_model['model_name']}.",
                style={"textDecoration": "underline", "color": "red"},
            ),
            html.Hr(),
            html.P(
                [
                    html.B("Model Type: "),
                    f"{selected_metadata['model_name']}",
                ]
            ),
            html.P(
                [
                    html.B(f"Best Metric {metric_name}: "),
                    f"{selected_metadata.get(metric_key, 'N/A'):.10f}",
                ]
            ),
            html.P(
                [
                    html.B("Trained from: "),
                    f"{selected_metadata['date_range']['train_start']} to {selected_metadata['date_range']['train_end']}",
                ]
            ),
            html.P(
                [
                    html.B("Tested from: "),
                    f"{selected_metadata['date_range']['test_start']} to {selected_metadata['date_range']['test_end']}",
                ]
            ),
            html.P(
                [
                    html.B("Features used: "),
                    f"{', '.join(selected_metadata['features'])}",
                ]
            ),
            html.P(
                [
                    html.B("Model Training Date: "),
                    f"{datetime.strptime(selected_metadata['training_time'], '%Y%m%d_%H%M%S').strftime('%Y-%m-%d %H:%M:%S')}",
                ]
            ),
        ],
        style={"border": "1px solid #ddd", "padding": "10px", "borderRadius": "5px"},
    )


@app.callback(
    Output("data-generation-status-setup", "children"),
    Input("generate-data-button-setup", "n_clicks"),
    State("start-date-picker-setup", "date"),
    State("start-time-picker-setup", "value"),
    State("end-date-picker-setup", "date"),
    State("end-time-picker-pickup", "value"),
)
def generate_new_data(n_clicks, start_date, start_time, end_date, end_time):
    if n_clicks == 0:
        return ""

    start_time = start_time if start_time else "00:00:00"
    end_time = end_time if end_time else "00:00:00"

    if not start_date or not end_date:
        return "Please enter both a Start Date and a End Date."

    try:
        start_datetime = pd.to_datetime(f"{start_date} {start_time}")
        end_datetime = pd.to_datetime(f"{end_date} {end_time}")

        if start_datetime > end_datetime:
            return "Please enter a End Date after the Start Date."

        file_name = get_pjm_features(
            start_datetime=start_datetime,
            end_datetime=end_datetime,
        )
        return f"Data successfully generated: {file_name}."

    except Exception as e:
        return f"Error generating data: {str(e)}"


@app.callback(
    Output("build-models-dataset-selection-setup", "options"),
    Input("refresh-dataset-button-build-models-setup", "n_clicks"),
    prevent_initial_call=True,
)
def update_build_datasets(_):
    datasets = list_available_datasets(DATAFRAME_FOLDER)
    return datasets if datasets else [{"label": "No datasets available", "value": ""}]


@app.callback(
    Output("model-building-status-setup", "children"),
    [
        Input("build-models-button-setup", "n_clicks"),
    ],
    [
        State("build-models-dataset-selection-setup", "value"),  # Selected dataset
        State(
            "build_models_predictors-selection-setup", "value"
        ),  # Selected predictors
        State("models-selection-setup", "value"),  # Selected models
    ],
    prevent_initial_call=True,
)
def build_models(n_clicks, dataset_path, selected_predictors, selected_models):
    if n_clicks == 0:
        return ""
    if not dataset_path or not selected_predictors or not selected_models:
        return html.P(
            "Please select a dataset, predictors, and models to proceed.",
            style={"color": "red"},
        )

    try:
        # Load the dataset
        data = pd.read_parquet(dataset_path, engine="pyarrow")
        preprocess_functions = {
            "EMERGENCY_TRIGGER_FOLDER": grid_analytics_helper.emergency_trigger_set_up,
            "LMP_VOLATILITY_FOLDER": grid_analytics_helper.lmp_volatility_set_up,
            "FORCED_OUTAGE_FOLDER": grid_analytics_helper.forced_outages_set_up,
        }

        if (
            "LMP_VOLATILITY" in selected_predictors
            and "random_forest" in selected_models
        ):
            selected_models.remove("random_forest")

        # preprocessed_data = {}
        for predictor in selected_predictors:
            if predictor not in preprocess_functions:
                return html.P(f"Unknown predictor: {predictor}", style={"color": "red"})

            # Preprocess the data for the selected predictor
            modeling_data = preprocess_functions[predictor](data)

            # Determine the target column
            target_column = (
                "emergency_triggered"
                if predictor == "EMERGENCY_TRIGGER_FOLDER"
                else "lmp_volatility"
                if predictor == "LMP_VOLATILITY"
                else "forced_outages_mw"
            )
            if predictor == "EMERGENCY_TRIGGER_FOLDER":
                # Classification for emergency trigger
                grid_analytics_helper.walk_forward_validation_classification(
                    data=modeling_data,
                    target_column=target_column,
                    model_save_path=MODEL_FOLDER,
                    models_to_use=selected_models,
                )
            else:
                # Regression for LMP Volatility and Forced Outages
                grid_analytics_helper.walk_forward_validation_regression(
                    data=modeling_data,
                    target_column=target_column,
                    model_save_path=MODEL_FOLDER,
                    models_to_use=selected_models,
                )
        return html.P("Models successfully built and saved!", style={"color": "green"})

    except Exception as e:
        return html.P(f"Error during model building: {str(e)}", style={"color": "red"})


################################################################################################################################
# Forecast


@app.callback(
    Output("forecast-models-dataset-selection", "options"),
    Input("refresh-dataset-button-forecast-models", "n_clicks"),
    prevent_initial_call=True,
)
def update_forecast_datasets(_):
    datasets = list_available_datasets(DATAFRAME_FOLDER)
    return datasets if datasets else [{"label": "No datasets available", "value": ""}]


@app.callback(
    Output("forecast-model-selection", "options"),
    Input("forecast-predictor-selection", "value"),
)
def update_model_selection(predictor):
    if not predictor:
        return []
    metadata = load_model_metadata(predictor)
    return [
        {
            "label": f"{meta['model_name']} (Trained On: {meta['date_range']['train_start']} - {meta['date_range']['train_end']}, Created On: {datetime.strptime(meta['training_time'], '%Y%m%d_%H%M%S').strftime('%Y-%m-%d %H:%M:%S')})",
            "value": meta["save_path"],
        }
        for meta in metadata
    ]


@app.callback(
    [
        Output("region-dropdown-container-forecast", "style"),
        Output("region-selection-forecast", "options"),
        Output("pnode-dropdown-container-forecast", "style"),
        Output("pnode-id-selection-forecast", "options"),
    ],
    [
        Input("forecast-predictor-selection", "value"),
        Input("view-forecast-model-button", "n_clicks"),
    ],
    [
        State("forecast-models-dataset-selection", "value"),
        State("forecast-model-selection", "value"),
    ],
    prevent_initial_call=True,
)
def update_region_and_pnode_dropdown(predictor, n_clicks, dataset_path, model_path):
    logging.debug("REGION AND PNODE DROPDOWN CALL")
    if predictor == EMERGENCY_TRIGGER_FOLDER or not (
        n_clicks and dataset_path and model_path
    ):
        return {"visibility": "hidden"}, [], {"visibility": "hidden"}, []

    try:
        data = pd.read_parquet(dataset_path, engine="pyarrow")
        preprocess_functions = {
            EMERGENCY_TRIGGER_FOLDER: grid_analytics_helper.emergency_trigger_set_up,
            LMP_VOLATILITY_FOLDER: grid_analytics_helper.lmp_volatility_set_up,
            FORCED_OUTAGE_FOLDER: grid_analytics_helper.forced_outages_set_up,
        }
        preprocessed_data = preprocess_functions[predictor](data)
        plot_data, _ = format_forecast_data(
            preprocessed_data,
            predictor,
            np.arange(0, len(preprocessed_data)),
        )

        # Populate region options
        if "region" in plot_data.columns:
            regions = plot_data["region"].unique()
            region_options = [{"label": region, "value": region} for region in regions]
            region_style = {"visibility": "visible"}
        else:
            region_options = []
            region_style = {"visibility": "hidden"}

        # Populate pnode_id options
        if "pnode_id" in plot_data.columns:
            pnode_ids = plot_data["pnode_id"].unique()
            pnode_options = [
                {"label": pnode_id, "value": pnode_id} for pnode_id in pnode_ids
            ]
            pnode_style = {"visibility": "visible"}
        else:
            pnode_options = []
            pnode_style = {"visibility": "hidden"}

        return region_style, region_options, pnode_style, pnode_options

    except Exception as e:
        logging.error(f"Error updating dropdowns: {e}")
        return {"visibility": "hidden"}, [], {"visibility": "hidden"}, []


@app.callback(
    [
        Output("forecast-graph", "figure"),
        Output("forecast-risk-metrics-display", "children"),
        Output("forecast-feature-importance-display", "children"),
    ],
    [
        Input("view-forecast-model-button", "n_clicks"),
    ],
    [
        State("forecast-models-dataset-selection", "value"),
        State("forecast-predictor-selection", "value"),
        State("forecast-model-selection", "value"),
        State("region-selection-forecast", "value"),
        State("pnode-id-selection-forecast", "value"),
    ],
    prevent_initial_call=True,
)
def generate_forecast(n_clicks, dataset_path, predictor, model_path, region, pnode_id):
    if not (n_clicks and dataset_path and predictor and model_path):
        return go.Figure().update_layout(title="Missing Inputs"), html.P(
            "Please select all inputs."
        )
    try:
        data = pd.read_parquet(dataset_path, engine="pyarrow")
        preprocess_functions = {
            EMERGENCY_TRIGGER_FOLDER: grid_analytics_helper.emergency_trigger_set_up,
            LMP_VOLATILITY_FOLDER: grid_analytics_helper.lmp_volatility_set_up,
            FORCED_OUTAGE_FOLDER: grid_analytics_helper.forced_outages_set_up,
        }
        preprocessed_data = preprocess_functions[predictor](data)
        if model_path.endswith(".txt"):
            model = Booster(model_file=model_path)
            feature_importance = model.feature_importance(importance_type="gain")
            feature_names = model.feature_name()
        else:
            model = joblib.load(model_path)
            if hasattr(model, "feature_importances_"):
                feature_importance = model.feature_importances_
                feature_names = preprocessed_data.columns.drop(predictor)
            else:
                feature_importance = None
                feature_names = None

        if feature_importance is not None:
            importance_df = pd.DataFrame(
                {
                    "Feature": feature_names,
                    "Importance": feature_importance,
                }
            ).sort_values(by="Importance", ascending=False)
            feature_importance_table = html.Table(
                [html.Tr([html.Th("Feature"), html.Th("Importance")])]
                + [
                    html.Tr(
                        [html.Td(row["Feature"]), html.Td(f"{row['Importance']:.2f}")]
                    )
                    for _, row in importance_df.iterrows()
                ]
            )
        else:
            feature_importance_table = html.P("Feature importance not available.")

        X = preprocessed_data.drop(columns=[predictor])
        if predictor == EMERGENCY_TRIGGER_FOLDER:
            y_predicted = (
                model.predict(X)
                if isinstance(model, Booster)
                else model.predict_proba(X)[:, 1]
            )
            y_predicted = (y_predicted >= 0.5).astype(int)
        else:
            y_predicted = model.predict(X)

        plot_data, y_actual = format_forecast_data(
            preprocessed_data, predictor, y_predicted
        )

        if region and region != "All" and y_actual is not None:
            plot_data = plot_data[plot_data["region"] == region]

        if pnode_id and pnode_id != "All" and "pnode_id" in plot_data.columns:
            plot_data = plot_data[plot_data["pnode_id"] == pnode_id]

        y_actual = plot_data[predictor]
        y_predicted = plot_data["y_predicted"]

        if y_actual is not None:  # If pure forecasting with no y_actual data
            figure = plot_interactive_timeseries_dash(
                data=plot_data,
                feature=predictor,
                feature_units=" "
                if predictor == EMERGENCY_TRIGGER_FOLDER
                else "($/MW)"
                if predictor == LMP_VOLATILITY_FOLDER
                else "(MW)",
                region_filter=("region" in plot_data.columns),
                region=region if region else "All",
                pnode_filter=("pnode_id" in plot_data.columns),
                pnode=pnode_id if pnode_id else "All",
            )
            if predictor == EMERGENCY_TRIGGER_FOLDER:
                accuracy = accuracy_score(y_actual, y_predicted)
                precision = precision_score(y_actual, y_predicted, zero_division=0)
                recall = recall_score(y_actual, y_predicted, zero_division=0)
                f1 = f1_score(y_actual, y_predicted, zero_division=0)
                metrics = html.Div(
                    [
                        html.P(f"Accuracy: {accuracy:.2f}"),
                        html.P(f"Precision: {precision:.2f}"),
                        html.P(f"Recall: {recall:.2f}"),
                        html.P(f"F1 Score: {f1:.2f}"),
                    ]
                )
            else:
                mse = mean_squared_error(y_actual, y_predicted)
                mae = mean_absolute_error(y_actual, y_predicted)
                r2 = r2_score(y_actual, y_predicted)
                metrics = html.Div(
                    [
                        html.P(f"MSE: {mse:.2f}"),
                        html.P(f"MAE: {mae:.2f}"),
                        html.P(f"R²: {r2:.2f}"),
                    ]
                )
        else:
            figure = plot_interactive_timeseries_dash(
                data=plot_data,
                feature=predictor,
                feature_units=" "
                if predictor == EMERGENCY_TRIGGER_FOLDER
                else "($/MW)"
                if predictor == LMP_VOLATILITY_FOLDER
                else "(MW)",
                region_filter=("region" in plot_data.columns),
                region=region if region else "All",
            )
            metrics = html.P("Strictly forecasting; No available data for comparison.")
        return figure, metrics, feature_importance_table

    except Exception as e:
        return go.Figure().update_layout(title=f"Error: {e}"), html.P(
            f"Error: {e}", style={"color": "red"}
        )


################################################################################################################################
# LMP EDA


@app.callback(
    Output("lmp-dataset-selection", "options"),
    Input("refresh-dataset-button-lmp", "n_clicks"),
    prevent_initial_call=True,
)
def update_forecast_datasets_lmp(_):
    datasets = list_available_datasets(DATAFRAME_FOLDER)
    return datasets if datasets else [{"label": "No datasets available", "value": ""}]


@app.callback(
    [Output("lmp-season-filter", "options"), Output("lmp-season-filter", "value")],
    [Input("lmp-dataset-selection", "value")],
    prevent_initial_call=True,
)
def update_season_filter_lmp(dataset_path):
    if not dataset_path:
        return [], "All"
    try:
        # Load and preprocess the dataset
        data = preprocess_lmp_data(dataset_path)
        data = add_seasons(data)

        seasons = [{"label": "All", "value": "All"}] + [
            {"label": season, "value": season} for season in data["season"].unique()
        ]
        return seasons, "All"
    except Exception as e:
        logging.error(f"Error updating season filter: {e}")
        return [], "All"


@app.callback(
    [Output("lmp-region-filter", "options"), Output("lmp-region-filter", "value")],
    [Input("lmp-dataset-selection", "value")],
    prevent_initial_call=True,
)
def update_lmp_region_filters(dataset_path):
    if not dataset_path:
        return [], "All"

    try:
        data = preprocess_lmp_data(dataset_path)
        regions = [{"label": "All", "value": "All"}] + [
            {"label": region, "value": region} for region in data["region"].unique()
        ]
        return regions, "All"
    except Exception as e:
        logging.error(f"Error updating region filters: {e}")
        return [], "All"


@app.callback(
    [Output("lmp-pnode-filter", "options"), Output("lmp-pnode-filter", "value")],
    [Input("lmp-dataset-selection", "value"), Input("lmp-region-filter", "value")],
    prevent_initial_call=True,
)
def update_lmp_pnode_filters(dataset_path, selected_region):
    if not dataset_path:
        return [], "All"

    try:
        data = preprocess_lmp_data(dataset_path)
        if selected_region != "All":
            data = data[data["region"] == selected_region]
        # Pricing Nodes depends on region
        pnodes = [{"label": "All", "value": "All"}] + [
            {"label": pnode, "value": pnode} for pnode in data["pnode_id"].unique()
        ]

        return pnodes, "All"
    except Exception as e:
        logging.error(f"Error updating pnode filters: {e}")
        return [], "All"


def render_lmp_histogram_tab():
    return html.Div(
        [
            html.Label("Bins:"),
            dcc.Slider(
                id="lmp-histogram-bins-slider",
                min=10,
                max=100,
                step=5,
                value=30,
                marks={i: str(i) for i in range(10, 101, 10)},
            ),
            html.Div(
                [
                    html.Label("Log Transformation:"),
                    dcc.Checklist(
                        id="lmp-histogram-log-transform",
                        options=[{"label": "Log Transform", "value": "log"}],
                        value=[],
                    ),
                ],
                style={"marginTop": "10px"},
            ),
            html.Button(
                "Generate Histogram",
                id="generate-lmp-histogram-button",
                style={"marginTop": "10px"},
            ),
            dcc.Graph(id="lmp-histogram-plot", style={"marginTop": "20px"}),
        ],
        style={"padding": "20px"},
    )


def render_lmp_boxplot_tab():
    return html.Div(
        [
            html.Div(
                [
                    html.Label("Lower Outlier Threshold:"),
                    dcc.Slider(
                        id="lmp-boxplot-lower-threshold-slider",
                        min=-1000,
                        max=0,
                        step=100,
                        value=-100,
                        marks={i: str(i) for i in range(-1000, 1, 100)},
                    ),
                ],
                style={"marginBottom": "20px"},
            ),
            html.Div(
                [
                    html.Label("Upper Outlier Threshold:"),
                    dcc.Slider(
                        id="lmp-boxplot-upper-threshold-slider",
                        min=0,
                        max=1000,
                        step=100,
                        value=100,
                        marks={i: str(i) for i in range(0, 1001, 100)},
                    ),
                ],
                style={"marginBottom": "20px"},
            ),
            html.Button(
                "Generate Boxplot",
                id="generate-lmp-boxplot-button",
                style={"marginTop": "10px"},
            ),
            dcc.Graph(id="lmp-boxplot-plot", style={"marginTop": "20px"}),
        ],
        style={"padding": "20px"},
    )


def render_lmp_timeseries():
    return html.Div(
        [
            html.Button(
                "Generate Timeseries",
                id="generate-lmp-timeseries-button",
                style={"marginTop": "10px"},
            ),
            dcc.Graph(id="lmp-timeseries-plot", style={"marginTop": "20px"}),
        ]
    )


def render_lmp_rolling_timeseries():
    return html.Div(
        [
            html.Div(
                [
                    html.Label("Rolling Window (Days):"),
                    dcc.Slider(
                        id="lmp-rolling-window-slider",
                        min=1,
                        max=35,
                        step=1,
                        value=7,
                        marks={i: str(i) for i in range(1, 36)},
                    ),
                ],
                style={"marginBottom": "20px"},
            ),
            html.Button(
                "Generate Rolling Timeseries",
                id="generate-lmp-rolling-timeseries-button",
                style={"marginBottom": "20px"},
            ),
            dcc.Graph(id="lmp-rolling-timeseries-plot", style={"marginTop": "20px"}),
        ],
        style={"padding": "20px"},
    )


def render_lmp_scatterplot_tab():
    return html.Div(
        [
            html.Div(
                [
                    html.Label("Outlier Detection Method:"),
                    dcc.Dropdown(
                        id="lmp-scatterplot-outlier-method",
                        options=[
                            {"label": "IQR-Based", "value": "iqr"},
                            {"label": "Dollar Threshold", "value": "dollar"},
                        ],
                        value="iqr",
                        placeholder="Select Outlier Method",
                    ),
                ],
                style={"marginBottom": "20px"},
            ),
            html.Div(
                [
                    html.Label("IQR Multiplier (if IQR-Based):"),
                    dcc.Slider(
                        id="lmp-scatterplot-iqr-multiplier-slider",
                        min=1.0,
                        max=5.0,
                        step=0.1,
                        value=1.5,
                        marks={i: str(i) for i in np.arange(1.0, 5.1, 0.5)},
                    ),
                ],
                id="lmp-scatterplot-iqr-slider-container",
                style={"marginBottom": "20px"},
            ),
            html.Div(
                [
                    html.Label("Dollar Threshold (if Dollar-Based):"),
                    dcc.Slider(
                        id="lmp-scatterplot-dollar-threshold-slider",
                        min=10,
                        max=500,
                        step=10,
                        value=50,
                        marks={i: str(i) for i in range(10, 501, 50)},
                    ),
                ],
                id="lmp-scatterplot-dollar-slider-container",
                style={"marginBottom": "20px", "display": "none"},
            ),
            html.Button(
                "Generate Scatterplot",
                id="generate-lmp-scatterplot-button",
                style={"marginTop": "10px"},
            ),
            dcc.Graph(id="lmp-scatterplot-plot", style={"marginTop": "20px"}),
        ],
        style={"padding": "20px"},
    )


def render_pnode_outlier_tab():
    return html.Div(
        [
            html.Div(
                [
                    html.Label("Outlier Detection Method:"),
                    dcc.Dropdown(
                        id="lmp-pnode-outlier-method",
                        options=[
                            {"label": "IQR-Based", "value": "iqr"},
                            {"label": "Dollar Threshold", "value": "dollar"},
                        ],
                        value="iqr",
                        placeholder="Select Outlier Method",
                    ),
                ],
                style={"marginBottom": "20px"},
            ),
            html.Div(
                [
                    html.Label("IQR Multiplier (if IQR-Based):"),
                    dcc.Slider(
                        id="lmp-pnode-iqr-multiplier-slider",
                        min=1.0,
                        max=5.0,
                        step=0.1,
                        value=1.5,
                        marks={i: str(i) for i in np.arange(1.0, 5.1, 0.5)},
                    ),
                ],
                id="lmp-pnode-iqr-slider-container",
                style={"marginBottom": "20px"},
            ),
            html.Div(
                [
                    html.Label("Dollar Threshold (if Dollar-Based):"),
                    dcc.Slider(
                        id="lmp-pnode-dollar-threshold-slider",
                        min=10,
                        max=500,
                        step=10,
                        value=50,
                        marks={i: str(i) for i in range(10, 501, 50)},
                    ),
                ],
                id="lmp-pnode-dollar-slider-container",
                style={"marginBottom": "20px", "display": "none"},
            ),
            html.Div(
                [
                    html.Label("Top N PNodes:"),
                    dcc.Slider(
                        id="lmp-pnode-top-n-slider",
                        min=0,
                        max=50,
                        step=1,
                        value=5,
                        marks={i: str(i) for i in range(0, 51, 5)},
                    ),
                ],
                style={"marginBottom": "20px"},
            ),
            html.Button(
                "Generate Outlier Table",
                id="generate-lmp-pnode-table-button",
                style={"marginTop": "10px"},
            ),
            html.Div(id="lmp-pnode-outlier-table", style={"marginTop": "20px"}),
        ],
        style={"padding": "20px"},
    )


@app.callback(
    Output("eda-tab-content-lmp", "children"),
    Input("eda-tabs-lmp", "value"),
)
def render_lmp_tab_content(tab_value):
    if tab_value == "histogram-tab-lmp":
        return render_lmp_histogram_tab()
    elif tab_value == "boxplot-tab-lmp":
        return render_lmp_boxplot_tab()
    elif tab_value == "timeseries-tab-lmp":
        return render_lmp_timeseries()
    elif tab_value == "rolling-timeseries-tab-lmp":
        return render_lmp_rolling_timeseries()
    elif tab_value == "scatterplot-tab-lmp":
        return render_lmp_scatterplot_tab()
    elif tab_value == "pnode-outlier-tab-lmp":
        return render_pnode_outlier_tab()
    return html.Div("Other tabs under construction.")


@app.callback(
    Output("lmp-histogram-plot", "figure"),
    [
        Input("generate-lmp-histogram-button", "n_clicks"),
        State("lmp-dataset-selection", "value"),
        State("lmp-season-filter", "value"),
        State("lmp-region-filter", "value"),
        State("lmp-pnode-filter", "value"),
        State("lmp-histogram-bins-slider", "value"),
        State("lmp-histogram-log-transform", "value"),
    ],
    prevent_initial_call=True,
)
def update_lmp_histogram(_, dataset_path, season, region, pnode, bins, log_transform):
    if not dataset_path:
        return go.Figure().update_layout(title="No Dataset Selected")

    data = preprocess_lmp_data(dataset_path)
    if region != "All":
        data = data[data["region"] == region]
    if pnode != "All":
        data = data[data["pnode_id"] == pnode]
    if season != "All":
        data = data[data["season"] == season]

    # Apply log transformation if selected
    if log_transform and "log" in log_transform:
        min_value = data[LMP_VOLATILITY_FOLDER].min()
        if min_value <= 0:
            data["lmp_volatility_log"] = data[LMP_VOLATILITY_FOLDER] - min_value + 1
        else:
            data["lmp_volatility_log"] = data[LMP_VOLATILITY_FOLDER] + 1

    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=data["lmp_volatility_log"]
            if log_transform
            else data[LMP_VOLATILITY_FOLDER],
            nbinsx=bins,
            marker=dict(color="blue"),
            opacity=0.75,
        )
    )
    fig.update_layout(
        title="LMP Volatility Distribution (Log Transformed)"
        if log_transform
        else "LMP Volatility Distribution",
        xaxis_title="LMP Volatility ($/MWh)",
        yaxis_title="Count",
        template="plotly_white",
    )
    return fig


@app.callback(
    Output("lmp-boxplot-plot", "figure"),
    [
        Input("generate-lmp-boxplot-button", "n_clicks"),
        State("lmp-dataset-selection", "value"),
        State("lmp-season-filter", "value"),
        State("lmp-region-filter", "value"),
        State("lmp-pnode-filter", "value"),
        State("lmp-boxplot-lower-threshold-slider", "value"),
        State("lmp-boxplot-upper-threshold-slider", "value"),
    ],
    prevent_initial_call=True,
)
def update_lmp_boxplot(
    _, dataset_path, season, region, pnode, lower_threshold, upper_threshold
):
    if not dataset_path:
        return go.Figure().update_layout(title="No Dataset Selected")

    data = preprocess_lmp_data(dataset_path)
    if region != "All":
        data = data[data["region"] == region]
    if pnode != "All":
        data = data[data["pnode_id"] == pnode]
    if season != "All":
        data = data[data["season"] == season]

    # Apply threshold filtering
    filtered_data = data[
        (data[LMP_VOLATILITY_FOLDER] >= lower_threshold)
        & (data[LMP_VOLATILITY_FOLDER] <= upper_threshold)
    ]

    # Create the boxplot figure
    fig = go.Figure()
    fig.add_trace(
        go.Box(
            y=filtered_data[LMP_VOLATILITY_FOLDER],
            x=filtered_data["region"] if "region" in filtered_data.columns else None,
            boxmean=True,
            name="Boxplot",
        )
    )
    fig.update_layout(
        title="LMP Volatility ($/MWh) Boxplot",
        xaxis_title="Region" if "region" in filtered_data.columns else None,
        yaxis_title="($/MWh)",
        template="plotly_white",
    )
    return fig


@app.callback(
    Output("lmp-timeseries-plot", "figure"),
    [
        Input("generate-lmp-timeseries-button", "n_clicks"),
        State("lmp-dataset-selection", "value"),
        State("lmp-season-filter", "value"),
        State("lmp-region-filter", "value"),
        State("lmp-pnode-filter", "value"),
    ],
    prevent_initial_call=True,
)
def update_lmp_timeseries(_, dataset_path, season, region, pnode):
    if not dataset_path:
        return go.Figure().update_layout(title="No Dataset Selected")
    data = preprocess_lmp_data(dataset_path)
    if region != "All":
        data = data[data["region"] == region]
    if pnode != "All":
        data = data[data["pnode_id"] == pnode]
    if season != "All":
        data = data[data["season"] == season]
    fig = plot_interactive_timeseries_dash(data, LMP_VOLATILITY_FOLDER, "($/MWh)")

    return fig


@app.callback(
    Output("lmp-rolling-timeseries-plot", "figure"),
    [
        Input("generate-lmp-rolling-timeseries-button", "n_clicks"),
        State("lmp-dataset-selection", "value"),
        State("lmp-rolling-window-slider", "value"),
        State("lmp-season-filter", "value"),
        State("lmp-region-filter", "value"),
        State("lmp-pnode-filter", "value"),
    ],
    prevent_initial_call=True,
)
def update_lmp_rolling_timeseries(
    _, dataset_path, rolling_window, season, region, pnode
):
    if not dataset_path:
        return go.Figure().update_layout(title="No Dataset Selected")
    data = preprocess_lmp_data(dataset_path)
    if region != "All":
        data = data[data["region"] == region]
    if pnode != "All":
        data = data[data["pnode_id"] == pnode]
    if season != "All":
        data = data[data["season"] == season]
    fig = plot_interactive_rolling_timeseries_dash(
        data,
        LMP_VOLATILITY_FOLDER,
        "($/MWh)",
        rolling_window,
    )

    return fig


@app.callback(
    Output("lmp-scatterplot-plot", "figure"),
    [
        Input("generate-lmp-scatterplot-button", "n_clicks"),
        State("lmp-dataset-selection", "value"),
        State("lmp-season-filter", "value"),
        State("lmp-region-filter", "value"),
        State("lmp-pnode-filter", "value"),
        State("lmp-scatterplot-outlier-method", "value"),
        State("lmp-scatterplot-iqr-multiplier-slider", "value"),
        State("lmp-scatterplot-dollar-threshold-slider", "value"),
    ],
    prevent_initial_call=True,
)
def update_lmp_scatterplot(
    _, dataset_path, season, region, pnode, method, iqr_multiplier, dollar_threshold
):
    if not dataset_path:
        return go.Figure().update_layout(title="No Dataset Selected")

    # Load and preprocess data
    data = preprocess_lmp_data(dataset_path)
    if region != "All":
        data = data[data["region"] == region]
    if pnode != "All":
        data = data[data["pnode_id"] == pnode]
    if season != "All":
        data = data[data["season"] == season]

    # Apply outlier detection method
    if method == "iqr":
        outliers = grid_analytics_helper.outlier_df_iqr(
            data, col_name=LMP_VOLATILITY_FOLDER, multiplier=iqr_multiplier
        )
    elif method == "dollar":
        outliers = grid_analytics_helper.outlier_df_by_dollar_amt(
            data, col_name=LMP_VOLATILITY_FOLDER, dollar_amt=dollar_threshold
        )
    else:
        return go.Figure().update_layout(title="Invalid Outlier Method Selected")
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=outliers["datetime_beginning_ept"],
            y=outliers[LMP_VOLATILITY_FOLDER],
            mode="markers",
            name="Outliers",
            marker=dict(color="red", size=8, opacity=0.8, symbol="x"),
        )
    )
    fig.update_layout(
        title="Outlier Scatterplot",
        xaxis_title="Datetime",
        yaxis_title="($/MWh)",
        template="plotly_white",
    )
    return fig


@app.callback(
    Output("lmp-pnode-outlier-table", "children"),
    [
        Input("generate-lmp-pnode-table-button", "n_clicks"),
        State("lmp-dataset-selection", "value"),
        State("lmp-season-filter", "value"),
        State("lmp-region-filter", "value"),
        State("lmp-pnode-filter", "value"),
        State("lmp-pnode-outlier-method", "value"),
        State("lmp-pnode-iqr-multiplier-slider", "value"),
        State("lmp-pnode-dollar-threshold-slider", "value"),
        State("lmp-pnode-top-n-slider", "value"),
    ],
    prevent_initial_call=True,
)
def generate_lmp_pnode_outlier_table(
    _,
    dataset_path,
    season,
    region,
    pnode,
    method,
    iqr_multiplier,
    dollar_threshold,
    top_n,
):
    if not dataset_path:
        return html.Div("No Dataset Selected", style={"color": "red"})

    # Load and preprocess data
    data = preprocess_lmp_data(dataset_path)
    if region != "All":
        data = data[data["region"] == region]
    if pnode != "All":
        data = data[data["pnode_id"] == pnode]
    if season != "All":
        data = data[data["season"] == season]

    # Apply outlier detection method
    if method == "iqr":
        outliers = grid_analytics_helper.outlier_df_iqr(
            data, col_name=LMP_VOLATILITY_FOLDER, multiplier=iqr_multiplier
        )
    elif method == "dollar":
        outliers = grid_analytics_helper.outlier_df_by_dollar_amt(
            data, col_name=LMP_VOLATILITY_FOLDER, dollar_amt=dollar_threshold
        )
    else:
        return html.Div("Invalid Outlier Method Selected", style={"color": "red"})

    if outliers.empty:
        return html.Div("No Outliers Detected", style={"color": "red"})

    # Generate top N nodes table
    top_outliers = (
        outliers[["pnode_id", "region"]]
        .value_counts()
        .reset_index(name="Outlier Count")
        .head(top_n)
    )
    table = dash.dash_table.DataTable(
        columns=[
            {"name": "PNode Name", "id": "pnode_id"},
            {"name": "Region", "id": "region"},
            {"name": "Outlier Count", "id": "Outlier Count"},
        ],
        data=top_outliers.to_dict("records"),
        style_table={"overflowX": "auto"},
        style_cell={"textAlign": "left"},
        style_header={"fontWeight": "bold"},
    )
    return table


@app.callback(
    [
        Output("lmp-scatterplot-iqr-slider-container", "style"),
        Output("lmp-scatterplot-dollar-slider-container", "style"),
    ],
    Input("lmp-scatterplot-outlier-method", "value"),
)
def lmp_toggle_outlier_method_sliders(method):
    if method == "iqr":
        return {"marginBottom": "20px"}, {"display": "none"}
    elif method == "dollar":
        return {"display": "none"}, {"marginBottom": "20px"}
    return {"display": "none"}, {"display": "none"}


@app.callback(
    [
        Output("lmp-pnode-iqr-slider-container", "style"),
        Output("lmp-pnode-dollar-slider-container", "style"),
    ],
    Input("lmp-pnode-outlier-method", "value"),
)
def lmp_toggle_pnode_outlier_method_sliders(method):
    if method == "iqr":
        return {"marginBottom": "20px"}, {"display": "none"}
    elif method == "dollar":
        return {"display": "none"}, {"marginBottom": "20px"}
    return {"display": "none"}, {"display": "none"}


################################################################################################################################
# Forced Outage Percentage EDA


@app.callback(
    Output("fo-dataset-selection", "options"),
    Input("refresh-dataset-button-fo", "n_clicks"),
    prevent_initial_call=True,
)
def update_forecast_datasets_fo(_):
    datasets = list_available_datasets(DATAFRAME_FOLDER)
    return datasets if datasets else [{"label": "No datasets available", "value": ""}]


@app.callback(
    [Output("fo-season-filter", "options"), Output("fo-season-filter", "value")],
    [Input("fo-dataset-selection", "value")],
    prevent_initial_call=True,
)
def update_season_filter_fo(dataset_path):
    if not dataset_path:
        return [], "All"
    try:
        data = preprocess_fo_data(dataset_path)

        seasons = [{"label": "All", "value": "All"}] + [
            {"label": season, "value": season} for season in data["season"].unique()
        ]
        return seasons, "All"
    except Exception as e:
        logging.error(f"Error updating season filter: {e}")
        return [], "All"


@app.callback(
    [Output("fo-region-filter", "options"), Output("fo-region-filter", "value")],
    [Input("fo-dataset-selection", "value")],
    prevent_initial_call=True,
)
def update_fo_region_filters(dataset_path):
    if not dataset_path:
        return [], "All"
    try:
        data = preprocess_fo_data(dataset_path)
        regions = [{"label": "All", "value": "All"}] + [
            {"label": region, "value": region} for region in data["region"].unique()
        ]
        return regions, "All"
    except Exception as e:
        logging.error(f"Error updating region filters: {e}")
        return [], "All"


def render_fo_histogram_tab():
    return html.Div(
        [
            html.Label("Bins:"),
            dcc.Slider(
                id="fo-histogram-bins-slider",
                min=10,
                max=100,
                step=5,
                value=30,
                marks={i: str(i) for i in range(10, 101, 10)},
            ),
            html.Div(
                [
                    html.Label("Log Transformation:"),
                    dcc.Checklist(
                        id="fo-histogram-log-transform",
                        options=[{"label": "Log Transform", "value": "log"}],
                        value=[],
                    ),
                ],
                style={"marginTop": "10px"},
            ),
            html.Button(
                "Generate Histogram",
                id="generate-fo-histogram-button",
                style={"marginTop": "10px"},
            ),
            dcc.Graph(id="fo-histogram-plot", style={"marginTop": "20px"}),
        ],
        style={"padding": "20px"},
    )


def render_fo_boxplot_tab():
    return html.Div(
        [
            html.Div(
                [
                    html.Label("Lower Outlier Threshold:"),
                    dcc.Slider(
                        id="fo-boxplot-lower-threshold-slider",
                        min=-100,
                        max=100,
                        step=10,
                        value=-50,
                        marks={i: str(i) for i in range(-100, 101, 10)},
                    ),
                ],
                style={"marginBottom": "20px"},
            ),
            html.Div(
                [
                    html.Label("Upper Outlier Threshold:"),
                    dcc.Slider(
                        id="fo-boxplot-upper-threshold-slider",
                        min=0,
                        max=200,
                        step=10,
                        value=100,
                        marks={i: str(i) for i in range(0, 201, 10)},
                    ),
                ],
                style={"marginBottom": "20px"},
            ),
            html.Button(
                "Generate Boxplot",
                id="generate-fo-boxplot-button",
                style={"marginTop": "10px"},
            ),
            dcc.Graph(id="fo-boxplot-plot", style={"marginTop": "20px"}),
        ],
        style={"padding": "20px"},
    )


def render_fo_timeseries():
    return html.Div(
        [
            html.Button(
                "Generate Timeseries",
                id="generate-fo-timeseries-button",
                style={"marginTop": "10px"},
            ),
            dcc.Graph(id="fo-timeseries-plot", style={"marginTop": "20px"}),
        ]
    )


def render_fo_rolling_timeseries():
    return html.Div(
        [
            html.Div(
                [
                    html.Label("Rolling Window (Days):"),
                    dcc.Slider(
                        id="fo-rolling-window-slider",
                        min=1,
                        max=35,
                        step=1,
                        value=7,
                        marks={i: str(i) for i in range(1, 36)},
                    ),
                ],
                style={"marginBottom": "20px"},
            ),
            html.Button(
                "Generate Rolling Timeseries",
                id="generate-fo-rolling-timeseries-button",
                style={"marginBottom": "20px"},
            ),
            dcc.Graph(id="fo-rolling-timeseries-plot", style={"marginTop": "20px"}),
        ],
        style={"padding": "20px"},
    )


def render_fo_calendar_heatmap():
    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.Label("Select Year:"),
                            dcc.Dropdown(
                                id="fo-calendar-year-filter",
                                placeholder="All Years",
                                value="All",
                            ),
                        ],
                        style={
                            "width": "33%",
                            "display": "inline-block",
                            "marginRight": "10px",
                        },
                    ),
                    html.Div(
                        [
                            html.Label("Select Month:"),
                            dcc.Dropdown(
                                id="fo-calendar-month-filter",
                                placeholder="All Months",
                                value="All",
                            ),
                        ],
                        style={
                            "width": "33%",
                            "display": "inline-block",
                            "marginRight": "10px",
                        },
                    ),
                    html.Div(
                        [
                            html.Label("Select Day:"),
                            dcc.Dropdown(
                                id="fo-calendar-day-filter",
                                placeholder="All Days",
                                value="All",
                            ),
                        ],
                        style={"width": "33%", "display": "inline-block"},
                    ),
                ],
                style={
                    "display": "flex",
                    "justifyContent": "space-between",
                    "marginBottom": "20px",
                },
            ),
            html.Button(
                "Generate Calendar Heatmap",
                id="generate-fo-calendar-button",
                style={"marginBottom": "20px"},
            ),
            dcc.Graph(id="fo-calendar-heatmap-plot", style={"marginTop": "20px"}),
        ],
        style={"padding": "20px"},
    )


@app.callback(
    Output("eda-tab-content-fo", "children"),
    Input("eda-tabs-fo", "value"),
)
def render_fo_tab_content(tab_value):
    if tab_value == "histogram-tab-fo":
        return render_fo_histogram_tab()
    elif tab_value == "boxplot-tab-fo":
        return render_fo_boxplot_tab()
    elif tab_value == "timeseries-tab-fo":
        return render_fo_timeseries()
    elif tab_value == "rolling-timeseries-tab-fo":
        return render_fo_rolling_timeseries()
    elif tab_value == "calendar-heatmap-tab-fo":
        return render_fo_calendar_heatmap()
    return html.Div("Other tabs under construction.")


@app.callback(
    Output("fo-histogram-plot", "figure"),
    [
        Input("generate-fo-histogram-button", "n_clicks"),
        State("fo-dataset-selection", "value"),
        State("fo-season-filter", "value"),
        State("fo-region-filter", "value"),
        State("fo-histogram-bins-slider", "value"),
        State("fo-histogram-log-transform", "value"),
    ],
    prevent_initial_call=True,
)
def update_fo_histogram(_, dataset_path, season, region, bins, log_transform):
    if not dataset_path:
        return go.Figure().update_layout(title="No Dataset Selected")

    data = preprocess_fo_data(dataset_path)
    if region != "All":
        data = data[data["region"] == region]
    if season != "All":
        data = data[data["season"] == season]

    if log_transform and "log" in log_transform:
        min_value = data["forced_outage_pct"].min()
        if min_value <= 0:
            data["forced_outage_pct_log"] = data["forced_outage_pct"] - min_value + 1
        else:
            data["forced_outage_pct_log"] = data["forced_outage_pct"] + 1

    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=data["forced_outage_pct_log"]
            if log_transform
            else data["forced_outage_pct"],
            nbinsx=bins,
            marker=dict(color="blue"),
            opacity=0.75,
        )
    )
    fig.update_layout(
        title="Forced Outage Percentage Distribution (Log Transformed)"
        if log_transform
        else "Forced Outage Percentage Distribution",
        xaxis_title="Forced Outage (%)",
        yaxis_title="Count",
        template="plotly_white",
    )
    return fig


@app.callback(
    Output("fo-boxplot-plot", "figure"),
    [
        Input("generate-fo-boxplot-button", "n_clicks"),
        State("fo-dataset-selection", "value"),
        State("fo-season-filter", "value"),
        State("fo-region-filter", "value"),
        State("fo-boxplot-lower-threshold-slider", "value"),
        State("fo-boxplot-upper-threshold-slider", "value"),
    ],
    prevent_initial_call=True,
)
def update_fo_boxplot(
    _, dataset_path, season, region, lower_threshold, upper_threshold
):
    if not dataset_path:
        return go.Figure().update_layout(title="No Dataset Selected")

    data = preprocess_fo_data(dataset_path)
    if region != "All":
        data = data[data["region"] == region]
    if season != "All":
        data = data[data["season"] == season]

    filtered_data = data[
        (data["forced_outage_pct"] >= lower_threshold)
        & (data["forced_outage_pct"] <= upper_threshold)
    ]

    fig = go.Figure()
    fig.add_trace(
        go.Box(
            y=filtered_data["forced_outage_pct"],
            x=filtered_data["region"] if "region" in filtered_data.columns else None,
            boxmean=True,
            name="Boxplot",
        )
    )
    fig.update_layout(
        title="Forced Outage Percentage Boxplot",
        xaxis_title="Region" if "region" in filtered_data.columns else None,
        yaxis_title="(%)",
        template="plotly_white",
    )
    return fig


@app.callback(
    Output("fo-timeseries-plot", "figure"),
    [
        Input("generate-fo-timeseries-button", "n_clicks"),
        State("fo-dataset-selection", "value"),
        State("fo-season-filter", "value"),
        State("fo-region-filter", "value"),
    ],
    prevent_initial_call=True,
)
def update_fo_timeseries(_, dataset_path, season, region):
    if not dataset_path:
        return go.Figure().update_layout(title="No Dataset Selected")
    data = preprocess_fo_data(dataset_path)
    if season != "All":
        data = data[data["season"] == season]

    fig = plot_interactive_timeseries_dash(
        data, "forced_outage_pct", "(%)", region=region, pnode_filter=False
    )

    return fig


@app.callback(
    Output("fo-rolling-timeseries-plot", "figure"),
    [
        Input("generate-fo-rolling-timeseries-button", "n_clicks"),
        State("fo-dataset-selection", "value"),
        State("fo-rolling-window-slider", "value"),
        State("fo-season-filter", "value"),
        State("fo-region-filter", "value"),
    ],
    prevent_initial_call=True,
)
def update_fo_rolling_timeseries(_, dataset_path, rolling_window, season, region):
    if not dataset_path:
        return go.Figure().update_layout(title="No Dataset Selected")
    data = preprocess_fo_data(dataset_path)
    if season != "All":
        data = data[data["season"] == season]
    fig = plot_interactive_rolling_timeseries_dash(
        data,
        "forced_outage_pct",
        "(%)",
        rolling_window,
        region=region,
        pnode_filter=False,
    )

    return fig


@app.callback(
    Output("fo-calendar-heatmap-plot", "figure"),
    [
        Input("generate-fo-calendar-button", "n_clicks"),
        State("fo-dataset-selection", "value"),
        State("fo-season-filter", "value"),
        State("fo-region-filter", "value"),
        State("fo-calendar-year-filter", "value"),
        State("fo-calendar-month-filter", "value"),
        State("fo-calendar-day-filter", "value"),
    ],
    prevent_initial_call=True,
)
def update_calendar_heatmap(_, dataset_path, season, region, year, month, day):
    """Generates the calendar heatmap."""
    if not dataset_path:
        return go.Figure().update_layout(title="No Dataset Selected")

    data = preprocess_fo_data(dataset_path)
    data = preprocess_calendar_data(data)

    unique_days = data["datetime_beginning_ept"].dt.day.unique()
    min_day = unique_days.min()
    max_day = unique_days.max()
    day_range = list(range(min_day, max_day + 1))

    if season != "All":
        data = data[data["season"] == season]
    if region != "All":
        data = data[data["region"] == region]
    if year != "All":
        data = data[data["year"] == year]
    if month != "All":
        data = data[data["month"] == month]
    if day != "All":
        data = data[data["day"] == day]

    heatmap_data = data.pivot_table(
        index="day",
        columns=["year", "month"],
        values="forced_outage_pct",
        aggfunc="mean",
    ).fillna(0)

    fig = px.imshow(
        heatmap_data.values,
        color_continuous_scale="YlGnBu",
        labels={"color": "Forced Outage (%)"},
        x=list(range(heatmap_data.shape[1])),
        y=day_range,
    )
    x_labels = [
        f"{pd.Timestamp(year=col[0], month=col[1], day=1).strftime('%b-%Y')}"
        for col in heatmap_data.columns
    ]
    fig.update_layout(
        title="Calendar Heatmap of Forced Outage Percentage",
        xaxis=dict(
            title="Year-Month",
            tickmode="array",
            tickvals=list(range(heatmap_data.shape[1])),
            ticktext=x_labels,
            tickangle=90,
        ),
        yaxis=dict(
            title="Day of Month",
            tickvals=day_range,
            ticktext=day_range,
        ),
        width=1200,
        height=800,
        margin=dict(t=100, b=200, l=100, r=100),
    )
    return fig


@app.callback(
    [
        Output("fo-calendar-year-filter", "options"),
        Output("fo-calendar-month-filter", "options"),
        Output("fo-calendar-day-filter", "options"),
    ],
    Input("fo-dataset-selection", "value"),
)
def update_fo_calendar_dropdowns(dataset_path):
    if not dataset_path:
        return [], [], []
    data = preprocess_fo_data(dataset_path)
    data["datetime_beginning_ept"] = pd.to_datetime(data["datetime_beginning_ept"])
    years = [{"label": "All", "value": "All"}] + [
        {"label": str(year), "value": year}
        for year in data["datetime_beginning_ept"].dt.year.unique()
    ]
    months = [{"label": "All", "value": "All"}] + [
        {"label": str(month), "value": month}
        for month in data["datetime_beginning_ept"].dt.month.unique()
    ]
    days = [{"label": "All", "value": "All"}] + [
        {"label": str(day), "value": day}
        for day in data["datetime_beginning_ept"].dt.day.unique()
    ]
    return years, months, days


################################################################################################################################
# Emergency Trigger EDA


@app.callback(
    Output("et-dataset-selection", "options"),
    Input("refresh-dataset-button-et", "n_clicks"),
    prevent_initial_call=True,
)
def update_forecast_datasets_et(_):
    datasets = list_available_datasets(DATAFRAME_FOLDER)
    return datasets if datasets else [{"label": "No datasets available", "value": ""}]


@app.callback(
    [Output("et-season-filter", "options"), Output("et-season-filter", "value")],
    [Input("et-dataset-selection", "value")],
    prevent_initial_call=True,
)
def update_season_filter_et(dataset_path):
    if not dataset_path:
        return [], "All"
    try:
        # Load and preprocess the dataset
        data = preprocess_trigger_data(dataset_path)

        seasons = [{"label": "All", "value": "All"}] + [
            {"label": season, "value": season} for season in data["season"].unique()
        ]
        return seasons, "All"
    except Exception as e:
        logging.error(f"Error updating season filter: {e}")
        return [], "All"


def render_et_barchart_tab():
    return html.Div(
        [
            html.Button(
                "Generate Barchart",
                id="generate-et-barchart-button",
                style={"marginTop": "10px"},
            ),
            dcc.Graph(id="et-barchart-plot", style={"marginTop": "20px"}),
        ]
    )


def render_et_freq_ts_tab():
    return html.Div(
        [
            html.Div(
                [
                    html.Label("Frequency Aggregation:"),
                    dcc.Dropdown(
                        id="et-freq-dropdown",
                        options=[
                            {"label": "Monthly (M)", "value": "M"},
                            {
                                "label": "By Month (All Years Combined)",
                                "value": "month",
                            },
                            {
                                "label": "By Day (All Years Combined)",
                                "value": "day",
                            },
                            {
                                "label": "By Hour (All Years Combined)",
                                "value": "hour)",
                            },
                            {"label": "By Year", "value": "year"},
                        ],
                        value="year",
                        style={"width": "100%"},
                    ),
                ],
                style={"width": "30%", "marginRight": "10px"},
            ),
            html.Button(
                "Generate Frequency Timeseries",
                id="generate-et-freq-ts-button",
                style={"marginTop": "10px"},
            ),
            dcc.Graph(id="et-freq-ts-plot", style={"marginTop": "20px"}),
        ],
        style={"padding": "20px"},
    )


@app.callback(
    Output("eda-tab-content-et", "children"),
    Input("eda-tabs-et", "value"),
)
def render_et_tab_content(tab_value):
    if tab_value == "barchart-tab-et":
        return render_et_barchart_tab()
    if tab_value == "freq-ts-tab-et":
        return render_et_freq_ts_tab()
    return html.Div("Other tabs under construction.")


@app.callback(
    Output("et-barchart-plot", "figure"),
    [
        Input("generate-et-barchart-button", "n_clicks"),
        State("et-dataset-selection", "value"),
        State("et-season-filter", "value"),
    ],
    prevent_initial_call=True,
)
def update_seasonal_barchart(_, dataset_path, season):
    if not dataset_path:
        return go.Figure().update_layout(title="No Dataset Selected")

    data = preprocess_trigger_data(dataset_path)

    if season != "All":
        data = data[data["season"] == season]

    grouped_data = (
        data.groupby(["season"])
        .agg({EMERGENCY_TRIGGER_FOLDER: "sum", NEAR_EMERGENCY: "sum"})
        .reset_index()
    )

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=grouped_data["season"],
            y=grouped_data[EMERGENCY_TRIGGER_FOLDER],
            name="Emergency Triggered",
            marker=dict(color="blue"),
        )
    )
    fig.add_trace(
        go.Bar(
            x=grouped_data["season"],
            y=grouped_data[NEAR_EMERGENCY],
            name="Near Emergency",
            marker=dict(color="red"),
        )
    )
    fig.update_layout(
        title="Seasonal Breakdown Barchart",
        xaxis_title="Season(s)",
        yaxis_title="Count",
        barmode="group",
        template="plotly_white",
        legend_title="Features",
    )
    return fig


@app.callback(
    Output("et-freq-ts-plot", "figure"),
    [
        Input("generate-et-freq-ts-button", "n_clicks"),
        State("et-dataset-selection", "value"),
        State("et-season-filter", "value"),
        State("et-freq-dropdown", "value"),
    ],
    prevent_initial_call=True,
)
def update_seasonal_freq_ts(_, dataset_path, season, frequency):
    if not dataset_path:
        return go.Figure().update_layout(title="No Dataset Selected")

    data = preprocess_trigger_data(dataset_path)

    if season != "All":
        data = data[data["season"] == season]

    if frequency in ["month", "day", "hour", "year"]:
        data["period"] = data["datetime_beginning_ept"].dt.__getattribute__(frequency)
    else:  # Ex. "M" for Year-Month
        data["period"] = data["datetime_beginning_ept"].dt.to_period(frequency)
    data["period"] = data["period"].astype(str)

    frequency_df = (
        data.groupby("period")[[EMERGENCY_TRIGGER_FOLDER, NEAR_EMERGENCY]]
        .sum()
        .reset_index()
    )
    fig = go.Figure()
    for feature in [EMERGENCY_TRIGGER_FOLDER, NEAR_EMERGENCY]:
        fig.add_trace(
            go.Scatter(
                x=frequency_df["period"],
                y=frequency_df[feature],
                mode="lines+markers",
                name=feature.replace("_", " ").title(),
            )
        )
    _title = f"Frequency of Features Over Time ({frequency.capitalize()}-Aggregated)"
    fig.update_layout(
        title=_title,
        xaxis_title="Time Period",
        yaxis_title="Frequency/Count",
        legend_title="Features",
        template="plotly_white",
    )

    return fig


################################################################################################################################
if __name__ == "__main__":
    app.run_server(debug=True)
