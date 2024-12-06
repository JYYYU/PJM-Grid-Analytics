import pandas as pd
import numpy as np
import plotly.express as px
from plotly import graph_objects as go
import plotly.io as pio
import ipywidgets as widgets
from IPython.display import display
import re
import os
import joblib
import json
from datetime import datetime
import logging
from scipy.stats import linregress
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.metrics import classification_report, mean_squared_error
from sklearn.preprocessing import OneHotEncoder

#######################################################################################################################################
### Merge Data ###
#######################################################################################################################################


def merge_lmp_gen_data(lmp_df, gen_cap_df, zone_to_region_df):
    """Merges day-ahead hourly LMPs (mnt_ftr_zonal_lmps) and daily generation capacity (day_gen_capacity) together.

    Args:
        lmp_df (pd.DataFrame): Dataframe containing day-ahead hourly LMPs data from PJM. See: https://dataminer2.pjm.com/feed/mnt_ftr_zonal_lmps/definition
        gen_cap_df (pd.DataFrame): Dataframe containing daily generation capacity data from PJM. See: https://dataminer2.pjm.com/feed/day_gen_capacity/definition
        zone_to_region_df (pd.DataFrame): Dataframe containing a custom mapping between zones and region.

    Returns:
        pd.DataFrame: A merged dataframe.
    """
    merged_df = pd.merge(
        lmp_df,
        gen_cap_df,
        left_on="datetime_beginning_ept",
        right_on="bid_datetime_beginning_ept",
        how="left",
    )
    merged_df.drop(
        columns=[
            "datetime_beginning_utc",
            "datetime_ending_utc",
            "bid_datetime_beginning_utc",
            "bid_datetime_beginning_ept",
        ],
        inplace=True,
    )
    # Handle any potential missing data from gen_cap_df
    for col in ["eco_max", "emerg_max", "total_committed"]:
        merged_df[col] = merged_df[col].interpolate(method="linear")
    # Add region by pricing node
    zone_pattern = "|".join(zone_to_region_df["zone"].apply(re.escape))
    zone_to_region_dict = zone_to_region_df.set_index("zone")["region"].to_dict()
    zone_arr = merged_df["pnode_name"].apply(lambda x: x.split("_")[0])
    parent_zone_arr = zone_arr.str.extract("(" + zone_pattern + ")")
    merged_df["region"] = parent_zone_arr.iloc[:, 0].map(zone_to_region_dict)
    merged_df.reset_index(drop=True, inplace=True)
    logging.info("LMP and Generation Capacity data merged together.")
    return merged_df


def merge_on_gen_outage_data(merged_df, outage_df):
    """Performs the second merge by merging on the closest forecasted generation outages to the forecast date (gen_outages_by_type).

    Args:
        merged_df (pd.DataFrame): Dataframe obtained from merge_lmp_gen_data.
        outage_df (pd.DataFrame): Dataframe containing generation outage for seven days by type from PJM. See: https://dataminer2.pjm.com/feed/gen_outages_by_type/definition

    Returns:
        pd.DataFrame: A merged dataframe.
    """
    outage_df = outage_df.sort_values(
        by=["region", "forecast_date", "forecast_execution_date_ept"],
        ascending=[True, True, False],
    )
    closest_forecasts = outage_df.drop_duplicates(
        subset=["forecast_date", "region"], keep="first", ignore_index=True
    ).copy()
    merged_df.loc[:, "date_only"] = merged_df["datetime_beginning_ept"].dt.date
    closest_forecasts.loc[:, "date_only"] = closest_forecasts["forecast_date"].dt.date
    merged_df = pd.merge(
        merged_df, closest_forecasts, on=["region", "date_only"], how="left"
    )
    # Add system-wide metrics (PJM RTO)
    pjm_rto_data = closest_forecasts[
        [
            "forecast_date",
            "total_outages_mw",
            "planned_outages_mw",
            "maintenance_outages_mw",
            "forced_outages_mw",
            "region",
        ]
    ].copy()
    pjm_rto_data = pjm_rto_data[pjm_rto_data["region"] == "PJM RTO"].drop(
        columns=["region"]
    )
    pjm_rto_data = pjm_rto_data.add_prefix("rto_").rename(
        columns={"rto_forecast_date": "forecast_date"}
    )
    merged_df = pd.merge(merged_df, pjm_rto_data, on="forecast_date", how="left")
    # Forward fill any missing data in gen_cap_df
    merged_df = merged_df.sort_values(by=["region", "datetime_beginning_ept"])
    merged_df[
        [
            "total_outages_mw",
            "planned_outages_mw",
            "maintenance_outages_mw",
            "forced_outages_mw",
            "rto_total_outages_mw",
            "rto_planned_outages_mw",
            "rto_maintenance_outages_mw",
            "rto_forced_outages_mw",
        ]
    ] = merged_df.groupby("region")[
        [
            "total_outages_mw",
            "planned_outages_mw",
            "maintenance_outages_mw",
            "forced_outages_mw",
            "rto_total_outages_mw",
            "rto_planned_outages_mw",
            "rto_maintenance_outages_mw",
            "rto_forced_outages_mw",
        ]
    ].ffill()
    merged_df.drop(columns=["forecast_execution_date_ept", "date_only"], inplace=True)
    merged_df.reset_index(drop=True, inplace=True)
    logging.info(
        "LMP, Generation Capacity, and Forecasted Generation Outages data all merged together."
    )
    return merged_df


def merge_historical_data(lmp_df, gen_cap_df, outage_df, zone_to_region_df):
    """Wrapper to merge all the necessary historical data.

    Args:
        lmp_df (pd.DataFrame): Dataframe containing day-ahead hourly LMPs data from PJM. See: https://dataminer2.pjm.com/feed/mnt_ftr_zonal_lmps/definition
        gen_cap_df (pd.DataFrame): Dataframe containing daily generation capacity data from PJM. See: https://dataminer2.pjm.com/feed/day_gen_capacity/definition
        outage_df (pd.DataFrame): Dataframe containing generation outage for seven days by type from PJM. See: https://dataminer2.pjm.com/feed/gen_outages_by_type/definition
        zone_to_region_df (pd.DataFrame): Dataframe containing a custom mapping between zones and region.

    Returns:
        pd.DataFrame: Final merged dataframe.
    """
    first_merge = merge_lmp_gen_data(lmp_df, gen_cap_df, zone_to_region_df)
    second_merge = merge_on_gen_outage_data(first_merge, outage_df)
    return second_merge


#######################################################################################################################################
### Feature Generation ###
#######################################################################################################################################


def create_capacity_margin(df):
    """Takes economic max and total committed of daily PJM available generation capacity and generates the new feature "capacity_margin".

    Args:
        df (pd.DataFrame): Dataframe containing Daily Generation Capacity data from PJM.

    Raises:
        ValueError: If the required columns to generate the feature is not found in the inputted dataframe.

    Returns:
        pd.DataFrame: A dataframe containing the daily hourly capacity margin.
    """
    req_features = ["eco_max", "total_committed"]
    if all(col in df.columns for col in req_features):
        df["capacity_margin"] = (
            ((df["eco_max"] - df["total_committed"]) / df["eco_max"]) * 100
        ).fillna(0)
        return df
    else:
        raise ValueError("'eco_max' or 'total_committed' are not in the dataframe.")


def create_emergency_triggered(df):
    """Takes economic max and total committed of daily PJM available generation capacity and generates the new feature "emergency_triggered".

    Args:
        df (pd.DataFrame): Dataframe containing Daily Generation Capacity data from PJM.

    Raises:
        ValueError: If the required columns to generate the feature is not found in the inputted dataframe.

    Returns:
        pd.DataFrame: A dataframe containing the daily hourly emergency trigger indicators.
    """
    req_features = ["eco_max", "total_committed"]
    if all(col in df.columns for col in req_features):
        df["emergency_triggered"] = (df["total_committed"] > df["eco_max"]).astype(int)
        return df
    else:
        raise ValueError("'eco_max' or 'total_committed' are not in the dataframe.")


def create_near_emergency(df, threshold=0.95):
    """Takes economic max and total committed of daily PJM available generation capacity and generates the new feature "near_emergency".

    Args:
        df (pd.DataFrame): Dataframe containing Daily Generation Capacity data from PJM.
        threshold (float, optional): Percentage of the economic max used to determine when to signal an early warning. Defaults to 0.95.

    Raises:
        ValueError: If the required columns to generate the feature is not found in the inputted dataframe.

    Returns:
        pd.DataFrame: A dataframe containing the daily hourly outage intensity.
    """
    req_features = ["eco_max", "total_committed"]
    if all(col in df.columns for col in req_features):
        df["near_emergency"] = (
            df["total_committed"] > threshold * df["eco_max"]
        ).astype(int)
        return df
    else:
        raise ValueError("'eco_max' or 'total_committed' are not in the dataframe.")


def create_forced_outage_pct(df):
    """Generates the forced outage percentage for Mid-Atlantic and Western regions, and the whole PJM RTO, labelled as "forced_outages_mw" and "rto_forced_outages_mw" respectively.

    Args:
        df (pd.DataFrame): Dataframe containing generation outage data.

    Raises:
        ValueError: If the required columns to generate the feature is not found in the inputted dataframe.

    Returns:
        pd.DataFrame: A dataframe containing the daily forced_outage_pct for Mid-Atlantic and Western regions, and the whole PJM RTO.
    """
    req_features = [
        "forced_outages_mw",
        "rto_forced_outages_mw",
        "total_outages_mw",
        "rto_total_outages_mw",
    ]
    if all(col in df.columns for col in req_features):
        df["forced_outage_pct_by_region"] = (
            (df["forced_outages_mw"] / df["total_outages_mw"]) * 100
        ).fillna(0)
        df["forced_outage_pct_PJM"] = (
            (df["rto_forced_outages_mw"] / df["rto_total_outages_mw"]) * 100
        ).fillna(0)
        return df
    else:
        raise ValueError(
            "'forced_outages_mw', 'rto_forced_outages_mw', 'total_outages_mw', or 'rto_total_outages_mw' are not in the dataframe."
        )


def create_outage_intensity(df):
    """Converts the hourly economic max to a daily rate and generates the new feature "outage_intensity".

    Args:
        df (pd.DataFrame): Dataframe containing Daily Generation Capacity and Generation Outage data from PJM.

    Raises:
        ValueError: If the required columns to generate the feature is not found in the inputted dataframe.

    Returns:
        pd.DataFrame: A dataframe containing the daily outage intensity.
    """
    req_features = ["datetime_beginning_ept", "total_outages_mw", "eco_max"]
    if all(col in df.columns for col in req_features):
        df.loc[:, "date_only"] = df["datetime_beginning_ept"].dt.date
        df["daily_eco_max"] = df.groupby("date_only")["eco_max"].transform("mean")
        df["outage_intensity"] = (
            (df["total_outages_mw"] / df["daily_eco_max"]) * 100
        ).fillna(0)
        df.drop(columns=["date_only"], inplace=True)
        return df
    else:
        raise ValueError(
            "'datetime_beginning_ept', 'total_outages_mw' or 'eco_max' are not in the dataframe."
        )


def create_region_stress_ratio(df):
    """Generates the ratios between the total outages from Western and Mid Atlantic regions with the entire PJM RTO outages.

    Args:
        df (pd.DataFrame): Dataframe containing generation outage data.

    Raises:
        ValueError: If the required columns to generate the feature is not found in the inputted dataframe.

    Returns:
        pd.DataFrmae: A dataframe containing the daily region stress ratios.
    """
    req_features = ["total_outages_mw", "rto_total_outages_mw"]
    if all(col in df.columns for col in req_features):
        df["region_stress_ratio"] = (
            (df["total_outages_mw"] / df["rto_total_outages_mw"]) * 100
        ).fillna(0)
        return df
    else:
        raise ValueError(
            "'total_outages_mw', or 'rto_total_outages_mw' are not in the dataframe."
        )


def create_lmp_delta(df):
    """Generates the hourly LMP delta and absolute LMP delta per pricing node, labelled "lmp_delta" and "lmp_abs_delta" respectively.

    Args:
        df (pd.DataFrame): Dataframe containing daily hourly LMP data for all available pricing node data from PJM.

    Raises:
        ValueError: If the required columns to generate the feature is not found in the inputted dataframe.

    Returns:
        pd.DataFrame: A dataframe containing the lmp delta and absolute lmp delta.
    """
    req_features = ["pnode_name", "datetime_beginning_ept", "lmp"]
    if all(col in df.columns for col in req_features):
        df = df.sort_values(by=["pnode_name", "datetime_beginning_ept"]).reset_index(
            drop=True
        )
        df["lmp_delta"] = df.groupby("pnode_name")["lmp"].diff()
        df["lmp_delta"] = df["lmp_delta"].fillna(0)
        df["lmp_abs_delta"] = df["lmp_delta"].abs()
        return df
    else:
        raise ValueError(
            "'pnode_name', 'datetime_beginning_ept', or 'lmp' are not in the dataframe."
        )


def create_lmp_volatility(df, n_hours_window=24):
    """Generates the rolling n_hours_window LMP volatility, labelled "lmp_volatility".

    Args:
        df (pd.DataFrame): Dataframe containing daily hourly LMP data for all available pricing node data from PJM and LMP delta data from create_lmp_delta.
        n_hours_window (int, optional): Number of hours for the rolling window. Defaults to 24.

    Raises:
        ValueError: If the required columns to generate the feature is not found in the inputted dataframe.

    Returns:
        pd.DataFrame: A dataframe containing the rolling n_hours_window lmp volatility.
    """
    req_features = ["pnode_name", "datetime_beginning_ept", "lmp_delta"]
    if all(col in df.columns for col in req_features):
        df = df.set_index("datetime_beginning_ept")
        df["lmp_volatility"] = (
            df.groupby("pnode_name")["lmp_delta"]
            .rolling(window=n_hours_window, min_periods=1)
            .std()
            .reset_index(level=0, drop=True)
        ).fillna(0)
        df = df.reset_index()
        return df
    else:
        raise ValueError(
            "'pnode_name', 'datetime_beginning_ept', or 'lmp_delta' are not in the dataframe."
        )


#######################################################################################################################################
### Data Handle ###
#######################################################################################################################################


def add_season_column(data):
    """Adds a 'season' column to the dataset based on the specified month mapping.

    Args:
        data (pd.DataFrame): The input dataset containing a 'datetime_beginning_ept' column.

    Returns:
        pd.DataFrame: Dataset with an additional 'season' column.
    """
    if "datetime_beginning_ept" not in data.columns:
        raise ValueError(
            "Column 'datetime_beginning_ept' is required for adding seasons."
        )
    data["datetime_beginning_ept"] = pd.to_datetime(data["datetime_beginning_ept"])
    data["month"] = data["datetime_beginning_ept"].dt.month
    month_to_season = {
        12: "Winter",
        1: "Winter",
        2: "Winter",
        6: "Summer",
        7: "Summer",
        8: "Summer",
    }
    data["season"] = data["month"].map(month_to_season).fillna("Spring & Fall")
    data.drop(columns=["month"], inplace=True)
    return data.copy()


def outlier_df_iqr(df, col_name, multiplier=1.5):
    """Outlier detection based on IQR x multiplier.

    Args:
        df (pd.DataFrame): Dataframe containings data for outlier detection.
        col_name (str): Column to be used in outlier detection.
        multiplier (float, optional): _description_. Defaults to 1.5.

    Returns:
        pd.Dataframe: A dataframe containing the outlier points.
    """
    if (not isinstance(multiplier, int)) or (not isinstance(multiplier, float)):
        multiplier = 1.5
    Q1 = df[col_name].quantile(0.25)
    Q3 = df[col_name].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    outlier_df = df[(df[col_name] > upper_bound) | (df[col_name] < lower_bound)].copy()
    return outlier_df.reset_index(drop=True)


def outlier_df_by_dollar_amt(df, col_name, dollar_amt=50):
    """Outlier detection based on dollar amount. This is specifically used for LMP features.

    Args:
        df (pd.DataFrame): Dataframe containings data for outlier detection.
        col_name (str): Column to be used in outlier detection.
        dollar_amt (int, optional): Dollar amount threshold to determine outliers. Defaults to $50.

    Returns:
        pd.Dataframe: A dataframe containing the outlier points.
    """
    if (not isinstance(dollar_amt, int)) or (not isinstance(dollar_amt, float)):
        dollar_amt = 50
    outlier_df = df[np.abs(df[col_name]) > dollar_amt]
    return outlier_df.reset_index(drop=True)


def interactive_pnode_lmp_outliers(
    data,
    lmp_feature,
    default_top_n_nodes=10,
    region_col="region",
):
    """Interactive visualization for top n outliers in a dataframe for lmp delta or lmp volatility.

    Args:
        data (pd.DataFrame): Dataframe containing data for outlier detection.
        lmp_feature (str): LMP feature name to consider ["lmp_delta", "lmp_volatility"]
        default_top_n_nodes (int, optional): Default value for top n nodes. Defaults to 10.
        region_col (str, optional): Column representing regions or categories. Defaults to "region".
    """
    if lmp_feature not in data.columns:
        raise ValueError(f"The column {lmp_feature} must exist in the dataframe.")
    if region_col not in data.columns:
        raise ValueError(f"The column '{region_col}' must exist in the dataframe.")
    if "season" not in data.columns:
        data = add_season_column(data)

    df = data[
        [
            "season",
            "pnode_name",
            region_col,
            lmp_feature,
        ]
    ].copy()
    df[lmp_feature] = df[lmp_feature].abs()

    region_dropdown = widgets.Dropdown(
        options=["All"] + sorted(df[region_col].unique().tolist()),
        value="All",
        description="Region:",
    )
    season_dropdown = widgets.Dropdown(
        options=["All"] + sorted(df["season"].unique().tolist()),
        value="All",
        description="Season:",
    )
    outlier_method_dropdown = widgets.Dropdown(
        options=["IQR-Based", "Dollar Threshold"],
        value="IQR-Based",
        description="Outlier Method:",
        style={"description_width": "initial"},
    )
    multiplier_slider = widgets.FloatSlider(
        value=1.5,
        min=1.0,
        max=5.0,
        step=0.1,
        description="IQR Multiplier:",
        layout=widgets.Layout(width="400px"),
        style={"description_width": "initial"},
    )
    dollar_amt_slider = widgets.IntSlider(
        value=50,
        min=10,
        max=500,
        step=10,
        description="Dollar Threshold:",
        layout=widgets.Layout(width="400px"),
        style={"description_width": "initial"},
    )
    top_n_slider = widgets.IntSlider(
        value=default_top_n_nodes,
        min=1,
        max=50,
        step=1,
        description="Top N Nodes:",
        layout=widgets.Layout(width="400px"),
        style={"description_width": "initial"},
    )

    output_table = widgets.Output()

    def update_table(change=None):
        filtered_data = df.copy()
        if region_dropdown.value != "All":
            filtered_data = filtered_data[
                filtered_data[region_col] == region_dropdown.value
            ]
        if season_dropdown.value != "All":
            filtered_data = filtered_data[
                filtered_data["season"] == season_dropdown.value
            ]
        if filtered_data.empty:
            with output_table:
                output_table.clear_output(wait=True)
                print(f"No data found for the selected region: {region_dropdown.value}")
            return
        if outlier_method_dropdown.value == "IQR-Based":
            outliers = outlier_df_iqr(
                df=filtered_data,
                col_name=lmp_feature,
                multiplier=multiplier_slider.value,
            )
        elif outlier_method_dropdown.value == "Dollar Threshold":
            outliers = outlier_df_by_dollar_amt(
                df=filtered_data,
                col_name=lmp_feature,
                dollar_amt=dollar_amt_slider.value,
            )
        else:
            outliers = pd.DataFrame()

        if outliers.empty:
            with output_table:
                output_table.clear_output(wait=True)
                print("No outliers detected with the selected method.")
            return
        top_outliers = (
            outliers[["pnode_name", region_col]]
            .value_counts()
            .head(top_n_slider.value)
            .reset_index()
            .rename(columns={0: "Outlier Count"})
        )
        with output_table:
            output_table.clear_output(wait=True)
            _title = f"Top {top_n_slider.value} Pricing Nodes with the Most Outliers by {lmp_feature.replace('_', ' ').title()}: \n*Using {outlier_method_dropdown.value}*"
            if season_dropdown.value != "All" and region_dropdown.value != "All":
                _title += f"\nRegion Filter: {region_dropdown.value} \nSeason Filter: {season_dropdown.value} "
            elif season_dropdown.value != "All":
                _title += f"\nSeason Filter: {season_dropdown.value}"
            elif region_dropdown.value != "All":
                _title += f"\nRegion Filter: {region_dropdown.value}"
            print(_title)
            top_outliers["Order"] = np.arange(1, top_n_slider.value + 1, 1)
            top_outliers.set_index("Order", inplace=True)
            display(top_outliers)

    region_dropdown.observe(update_table, names="value")
    season_dropdown.observe(update_table, names="value")
    outlier_method_dropdown.observe(update_table, names="value")
    multiplier_slider.observe(update_table, names="value")
    dollar_amt_slider.observe(update_table, names="value")
    top_n_slider.observe(update_table, names="value")
    update_table()
    controls = widgets.VBox(
        [
            region_dropdown,
            season_dropdown,
            outlier_method_dropdown,
            multiplier_slider,
            dollar_amt_slider,
            top_n_slider,
        ]
    )
    display(widgets.HBox([controls, output_table]))


def handle_forced_outage_pct_data(df):
    """Generates a new dataframe containing the daily forced outage percentage data for all regions and PJM RTO over time.

    Args:
        df (pd.DataFrame): Dataframe containing the generated features from create_forced_outage_pct.

    Raises:
        ValueError: If no data exists in the dataframe after sorting by zone.
        ValueError: If the required columns to generate the feature is not found in the inputted dataframe.

    Returns:
        pd.DataFrame: A dataframe containing a specific column for each region and PJM RTO over time for daily forced outage percentage data.
    """
    req_features = [
        "datetime_beginning_ept",
        "forced_outage_pct_by_region",
        "forced_outage_pct_PJM",
    ]
    if all(col in df.columns for col in req_features):
        merge_data = pd.DataFrame()
        for region, group in df.groupby("region"):
            sub_df = group.copy()
            sub_df["datetime_beginning_ept"] = sub_df["datetime_beginning_ept"].dt.date
            grouped_df = (
                sub_df.groupby("datetime_beginning_ept")["forced_outage_pct_by_region"]
                .mean()
                .reset_index()
            )
            grouped_df = grouped_df.rename(
                columns={"forced_outage_pct_by_region": "forced_outage_pct"}
            )
            grouped_df["region"] = np.repeat(region, grouped_df.shape[0])
            grouped_df.sort_values(
                by="datetime_beginning_ept", inplace=True, ignore_index=True
            )
            grouped_df.ffill(inplace=True)  # missing data here is rare
            merge_data = pd.concat([merge_data, grouped_df], axis=0, ignore_index=True)
        # Add on PJM RTO
        pjm = df.copy()
        pjm["datetime_beginning_ept"] = pjm["datetime_beginning_ept"].dt.date
        grouped_pjm = (
            pjm.groupby("datetime_beginning_ept")["forced_outage_pct_PJM"]
            .mean()
            .reset_index()
        )
        grouped_pjm = grouped_pjm.rename(
            columns={"forced_outage_pct_PJM": "forced_outage_pct"}
        )
        grouped_pjm.sort_values(
            by="datetime_beginning_ept", inplace=True, ignore_index=True
        )
        grouped_pjm["region"] = np.repeat("PJM RTO", grouped_pjm.shape[0])
        grouped_pjm.ffill(inplace=True)  # missing data here is rare
        merge_data = pd.concat([merge_data, grouped_pjm], axis=0, ignore_index=True)
        if len(merge_data.columns) > 1:
            return merge_data
        else:
            raise ValueError("The inputted dataframe does not have regional data.")
    else:
        raise ValueError(
            "'datetime_beginning_ept','forced_outage_pct_by_region', or 'forced_outage_pct_PJM' are not in the dataframe."
        )


def handle_outage_intensity_data(df):
    """Generates a new dataframe containing the daily outage intensity percentage data for all regions over time.

    Args:
        df (pd.DataFrame): Dataframe containing the generated features from create_outage_intensity.

    Raises:
        ValueError: If no data exists in the dataframe after sorting by zone.
        ValueError: If the required columns to generate the feature is not found in the inputted dataframe.

    Returns:
        pd.DataFrame: A dataframe containing a specific column for each region over time for daily outage intensity percentage data.
    """
    req_features = [
        "datetime_beginning_ept",
        "outage_intensity",
    ]
    if all(col in df.columns for col in req_features):
        merge_data = pd.DataFrame()
        for region, group in df.groupby("region"):
            sub_df = group.copy()
            sub_df["datetime_beginning_ept"] = sub_df["datetime_beginning_ept"].dt.date
            grouped_df = (
                sub_df.groupby("datetime_beginning_ept")["outage_intensity"]
                .mean()
                .reset_index()
            )
            grouped_df["region"] = np.repeat(region, grouped_df.shape[0])
            grouped_df.sort_values(
                by="datetime_beginning_ept", inplace=True, ignore_index=True
            )
            grouped_df.ffill(inplace=True)
            merge_data = pd.concat([merge_data, grouped_df], axis=0, ignore_index=True)
        if len(merge_data.columns) > 1:
            return merge_data
        else:
            raise ValueError("The inputted dataframe does not have regional data.")
    else:
        raise ValueError(
            "'datetime_beginning_ept' or 'outage_intensity' are not in the dataframe."
        )


def handle_capacity_margin_data(df):
    """Generates a new dataframe containing the hourly capacity margin data aggregated by hours.

    Args:
        df (pd.DataFrame): Dataframe containing the generated features from create_capacity_margin.

    Raises:
        ValueError: If the required columns to generate the feature is not found in the inputted dataframe.

    Returns:
        pd.DataFrame: A dataframe containing a the aggregated capacity margin data for each hour over all regions.
    """
    req_features = [
        "datetime_beginning_ept",
        "capacity_margin",
    ]
    if all(col in df.columns for col in req_features):
        agg_df = df.copy()
        grouped_df = (
            agg_df.groupby("datetime_beginning_ept")["capacity_margin"]
            .mean()
            .reset_index()
        )
        return grouped_df

    else:
        raise ValueError(
            "'datetime_beginning_ept' or 'capacity_margin' are not in the dataframe."
        )


def handle_region_stress_ratio_data(df):
    """Generates a new dataframe containing the daily region stress ratio data for all regions.

    Args:
        df (pd.DataFrame): Dataframe containing the generated features from create_region_stress_ratio.

    Raises:
        ValueError: If no data exists in the dataframe after sorting by zone.
        ValueError: If the required columns to generate the feature is not found in the inputted dataframe.

    Returns:
        pd.DataFrame: A dataframe containing a specific column for each region over time for daily region stress ratio data.
    """
    req_features = [
        "datetime_beginning_ept",
        "region_stress_ratio",
    ]
    if all(col in df.columns for col in req_features):
        merge_data = pd.DataFrame()
        for region, group in df.groupby("region"):
            sub_df = group.copy()
            sub_df["datetime_beginning_ept"] = sub_df["datetime_beginning_ept"].dt.date
            grouped_df = (
                sub_df.groupby("datetime_beginning_ept")["region_stress_ratio"]
                .mean()
                .reset_index()
            )
            grouped_df["region"] = np.repeat(region, grouped_df.shape[0])
            grouped_df.sort_values(
                by="datetime_beginning_ept", inplace=True, ignore_index=True
            )
            grouped_df.ffill(inplace=True)
            merge_data = pd.concat([merge_data, grouped_df], axis=0, ignore_index=True)

        if len(merge_data.columns) > 1:
            return merge_data
        else:
            raise ValueError("The inputted dataframe does not have regional data.")
    else:
        raise ValueError(
            "'datetime_beginning_ept' or 'region_stress_ratio' are not in the dataframe."
        )


def handle_daily_gen_capacity_data(df):
    """Generates a new dataframe containing the hourly daily generation capacity data with the trigger indicator features.
    - Note: As the generation capacity data was original daily granular and the indicator features, there are the same across all regions for the specific day. Thus applying an average will result in the same data.

    Args:
        df (pd.DataFrame): Dataframe containing the generated features from day_gen_capacity. Reference: https://dataminer2.pjm.com/feed/day_gen_capacity

    Raises:
        ValueError: If the required columns to generate the feature is not found in the inputted dataframe.

    Returns:
        pd.DataFrame: A dataframe containing the hourly daily generation capacity data with the trigger indicator features.
    """
    req_features = [
        "datetime_beginning_ept",
        "eco_max",
        "emerg_max",
        "total_committed",
        "emergency_triggered",
        "near_emergency",
    ]
    if all(col in df.columns for col in req_features):
        agg_data = df.copy()
        hourly_data = (
            agg_data.groupby(["datetime_beginning_ept"])[
                [
                    "eco_max",
                    "emerg_max",
                    "total_committed",
                    "emergency_triggered",
                    "near_emergency",
                ]
            ]
            .mean()
            .reset_index()
        )
        return hourly_data
    else:
        raise ValueError(
            "'datetime_beginning_ept',  'eco_max', 'emerg_max', 'total_committed', 'emergency_triggered', or 'near_emergency' are not in the dataframe."
        )


def handle_region_stress_ratio_lmp_vol_data(df):
    """Generates a new dataframe containing the daily specific data for lmp volatility and region stress ratio data across all regions.

    Args:
        df (pd.DataFrame): Dataframe containing the generated features from create_region_stress_ratio and create_lmp_volatility.

    Raises:
        ValueError: If the required columns to generate the feature is not found in the inputted dataframe.

    Returns:
        pd.DataFrame: A dataframe containing a daily region specific data for lmp volatility and region stress ratio data.
    """
    req_features = [
        "datetime_beginning_ept",
        "region",
        "region_stress_ratio",
        "lmp_volatility",
    ]
    if all(col in df.columns for col in req_features):
        agg_data = df.copy()
        agg_data["datetime_beginning_ept"] = agg_data["datetime_beginning_ept"].dt.date
        daily_lmp_volatility = (
            agg_data.groupby(["datetime_beginning_ept", "region"])["lmp_volatility"]
            .mean()
            .reset_index()
            .rename(columns={"lmp_volatility": "daily_lmp_volatility"})
        )
        region_sr = (
            agg_data.groupby(["datetime_beginning_ept", "region"])[
                "region_stress_ratio"
            ]
            .mean()
            .reset_index()
            .rename(columns={"region_stress_ratio": "daily_region_stress_ratio"})
        )
        final_df = pd.merge(
            daily_lmp_volatility,
            region_sr,
            on=["datetime_beginning_ept", "region"],
            how="inner",
        )
        return final_df
    else:
        raise ValueError(
            "'datetime_beginning_ept', 'region', 'region_stress_ratio', or 'lmp_volatility' are not in the dataframe."
        )

    req_features = [
        "datetime_beginning_ept",
        "region_stress_ratio",
    ]


def handle_capacity_margin_outage_intensity_data(df):
    """Generates a new dataframe containing the daily specific data for capacit margin and outage intensity data.

    Args:
        df (pd.DataFrame): Dataframe containing the generated features from create_region_stress_ratio and create_outage_intensity.

    Raises:
        ValueError: If the required columns to generate the feature is not found in the inputted dataframe.

    Returns:
        pd.DataFrame: A dataframe containing a daily region specific data for lmp volatility and region stress ratio data.
    """
    req_features = [
        "datetime_beginning_ept",
        "capacity_margin",
        "outage_intensity",
    ]
    if all(col in df.columns for col in req_features):
        agg_data = df.copy()
        agg_data["datetime_beginning_ept"] = agg_data["datetime_beginning_ept"].dt.date
        daily_capacity_margin = (
            agg_data.groupby(["datetime_beginning_ept"])["capacity_margin"]
            .mean()
            .reset_index()
            .rename(columns={"capacity_margin": "daily_capacity_margin"})
        )
        daily_outage_intensity = (
            agg_data.groupby(["datetime_beginning_ept"])["outage_intensity"]
            .mean()
            .reset_index()
            .rename(columns={"outage_intensity": "daily_outage_intensity"})
        )
        final_df = pd.merge(
            daily_capacity_margin,
            daily_outage_intensity,
            on=["datetime_beginning_ept"],
            how="inner",
        )
        return final_df
    else:
        raise ValueError(
            "'datetime_beginning_ept', 'capacity_margin', or 'outage_intensity' are not in the dataframe."
        )


def handle_region_stress_ratio_capacity_margin_data(df):
    """Generates a new dataframe containing the daily specific data for capacity margin and region stress ratio data.

    Args:
        df (pd.DataFrame): Dataframe containing the generated features from create_region_stress_ratio and create_capacity_margin.

    Raises:
        ValueError: If the required columns to generate the feature is not found in the inputted dataframe.

    Returns:
        pd.DataFrame: A dataframe containing a daily converted data for capacity margin and region stress ratio data.
    """
    req_features = ["datetime_beginning_ept", "region_stress_ratio", "capacity_margin"]
    if all(col in df.columns for col in req_features):
        agg_data = df.copy()
        agg_data["datetime_beginning_ept"] = agg_data["datetime_beginning_ept"].dt.date
        daily_capacity_margin = (
            agg_data.groupby(["datetime_beginning_ept"])["capacity_margin"]
            .mean()
            .reset_index()
            .rename(columns={"capacity_margin": "daily_capacity_margin"})
        )
        region_sr = (
            agg_data.groupby(["datetime_beginning_ept"])["region_stress_ratio"]
            .mean()
            .reset_index()
            .rename(columns={"region_stress_ratio": "daily_region_stress_ratio"})
        )
        final_df = pd.merge(
            daily_capacity_margin,
            region_sr,
            on=["datetime_beginning_ept"],
            how="inner",
        )
        return final_df
    else:
        raise ValueError(
            "'datetime_beginning_ept', 'region_stress_ratio', or 'capacity_margin' are not in the dataframe."
        )


def handle_region_stress_ratio_outage_intensity_data(df):
    """Generates a new dataframe containing the daily specific data for outage intensity and region stress ratio data.

    Args:
        df (pd.DataFrame): Dataframe containing the generated features from create_region_stress_ratio and create_outage_intensity.

    Raises:
        ValueError: If the required columns to generate the feature is not found in the inputted dataframe.

    Returns:
        pd.DataFrame: A dataframe containing a daily converted data for outage intensity and region stress ratio data.
    """
    req_features = ["datetime_beginning_ept", "region_stress_ratio", "outage_intensity"]
    if all(col in df.columns for col in req_features):
        agg_data = df.copy()
        agg_data["datetime_beginning_ept"] = agg_data["datetime_beginning_ept"].dt.date
        daily_outage_intensity = (
            agg_data.groupby(["datetime_beginning_ept"])["outage_intensity"]
            .mean()
            .reset_index()
            .rename(columns={"outage_intensity": "daily_outage_intensity"})
        )
        region_sr = (
            agg_data.groupby(["datetime_beginning_ept"])["region_stress_ratio"]
            .mean()
            .reset_index()
            .rename(columns={"region_stress_ratio": "daily_region_stress_ratio"})
        )
        final_df = pd.merge(
            daily_outage_intensity,
            region_sr,
            on=["datetime_beginning_ept"],
            how="inner",
        )
        return final_df
    else:
        raise ValueError(
            "'datetime_beginning_ept', 'region_stress_ratio', or 'outage_intensity' are not in the dataframe."
        )


def handle_forced_outages_capacity_margin_data(df):
    """Generates a new dataframe containing the daily specific data for forced outages and capacity margin data.

    Args:
        df (pd.DataFrame): Dataframe containing the features from PJM gen_outages_by_type and feature engineered function create_capacity_margin. Refer to: https://dataminer2.pjm.com/feed/gen_outages_by_type/definition

    Raises:
        ValueError: If the required columns to generate the feature is not found in the inputted dataframe.

    Returns:
        pd.DataFrame: A dataframe containing a daily specific data for forced outages and capacity margin data.
    """
    req_features = ["datetime_beginning_ept", "forced_outages_mw", "capacity_margin"]
    if all(col in df.columns for col in req_features):
        agg_data = df.copy()
        agg_data["datetime_beginning_ept"] = agg_data["datetime_beginning_ept"].dt.date
        daily_capacity_margin = (
            agg_data.groupby(["datetime_beginning_ept"])["capacity_margin"]
            .mean()
            .reset_index()
            .rename(columns={"capacity_margin": "daily_capacity_margin"})
        )
        daily_forced_outages = (
            agg_data.groupby(["datetime_beginning_ept"])["forced_outages_mw"]
            .mean()
            .reset_index()
            .rename(columns={"forced_outages_mw": "daily_forced_outages_mw"})
        )
        final_df = pd.merge(
            daily_capacity_margin,
            daily_forced_outages,
            on=["datetime_beginning_ept"],
            how="inner",
        )
        return final_df
    else:
        raise ValueError(
            "'datetime_beginning_ept', 'forced_outages_mw', or 'capacity_margin' are not in the dataframe."
        )


def handle_forced_outages_lmp_vol_data(df):
    """Generates a new dataframe containing the daily specific data for forced outages and lmp volatility data across all regions.

    Args:
        df (pd.DataFrame): Dataframe containing the features from PJM gen_outages_by_type and feature engineered function create_lmp_volatility. Refer to: https://dataminer2.pjm.com/feed/gen_outages_by_type/definition
    Raises:
        ValueError: If the required columns to generate the feature is not found in the inputted dataframe.

    Returns:
        pd.DataFrame: A dataframe containing a daily specific data for forced outages and lmp volatility data.
    """
    req_features = [
        "datetime_beginning_ept",
        "region",
        "forced_outages_mw",
        "lmp_volatility",
    ]
    if all(col in df.columns for col in req_features):
        agg_data = df.copy()
        agg_data["datetime_beginning_ept"] = agg_data["datetime_beginning_ept"].dt.date
        daily_lmp_volatility = (
            agg_data.groupby(["datetime_beginning_ept", "region"])["lmp_volatility"]
            .mean()
            .reset_index()
            .rename(columns={"lmp_volatility": "daily_lmp_volatility"})
        )
        daily_forced_outages = (
            agg_data.groupby(["datetime_beginning_ept", "region"])["forced_outages_mw"]
            .mean()
            .reset_index()
            .rename(columns={"forced_outages_mw": "daily_forced_outages_mw"})
        )
        final_df = pd.merge(
            daily_lmp_volatility,
            daily_forced_outages,
            on=["datetime_beginning_ept", "region"],
            how="inner",
        )
        return final_df
    else:
        raise ValueError(
            "'datetime_beginning_ept','region', 'forced_outages_mw', or 'lmp_volatility' are not in the dataframe."
        )


def handle_gen_outage_data(df):
    """Generates a new dataframe containing all the daily generation outage data (gen_outages_by_type) for all regions and PJM RTO over time.

    Args:
        df (pd.DataFrame): Dataframe containing the generated features from https://dataminer2.pjm.com/feed/gen_outages_by_type/definition.

    Raises:
        ValueError: If no data exists in the dataframe after sorting by zone.
        ValueError: If the required columns to generate the feature is not found in the inputted dataframe.

    Returns:
        pd.DataFrame: A dataframe containing a specific column for each region and PJM RTO over time for all generation outage data.
    """
    req_features = [
        "datetime_beginning_ept",
        "region",
        "total_outages_mw",
        "planned_outages_mw",
        "maintenance_outages_mw",
        "forced_outages_mw",
        "rto_total_outages_mw",
        "rto_planned_outages_mw",
        "rto_maintenance_outages_mw",
        "rto_forced_outages_mw",
    ]
    if all(col in df.columns for col in req_features):
        merge_data = pd.DataFrame()
        for region, group in df.groupby("region"):
            sub_df = group.copy()
            sub_df["datetime_beginning_ept"] = sub_df["datetime_beginning_ept"].dt.date
            grouped_df = (
                sub_df.groupby("datetime_beginning_ept")[
                    [
                        "total_outages_mw",
                        "planned_outages_mw",
                        "maintenance_outages_mw",
                        "forced_outages_mw",
                    ]
                ]
                .mean()
                .reset_index()
            )
            grouped_df["region"] = np.repeat(region, grouped_df.shape[0])
            grouped_df.sort_values(
                by="datetime_beginning_ept", inplace=True, ignore_index=True
            )
            grouped_df.ffill(inplace=True)  # missing data here is rare
            merge_data = pd.concat([merge_data, grouped_df], axis=0, ignore_index=True)
        # Add on PJM RTO
        pjm = df.copy()
        pjm["datetime_beginning_ept"] = pjm["datetime_beginning_ept"].dt.date
        grouped_pjm = (
            pjm.groupby("datetime_beginning_ept")[
                [
                    "rto_total_outages_mw",
                    "rto_planned_outages_mw",
                    "rto_maintenance_outages_mw",
                    "rto_forced_outages_mw",
                ]
            ]
            .mean()
            .reset_index()
        )
        grouped_pjm = grouped_pjm.rename(
            columns={
                "rto_total_outages_mw": "total_outages_mw",
                "rto_planned_outages_mw": "planned_outages_mw",
                "rto_maintenance_outages_mw": "maintenance_outages_mw",
                "rto_forced_outages_mw": "forced_outages_mw",
            }
        )
        grouped_pjm.sort_values(
            by="datetime_beginning_ept", inplace=True, ignore_index=True
        )
        grouped_pjm["region"] = np.repeat("PJM RTO", grouped_pjm.shape[0])
        grouped_pjm.ffill(inplace=True)  # missing data here is rare
        merge_data = pd.concat([merge_data, grouped_pjm], axis=0, ignore_index=True)
        if len(merge_data.columns) > 1:
            return merge_data
        else:
            raise ValueError(
                "The inputted dataframe does not have regional data nor PJM data."
            )
    else:
        raise ValueError(
            "'datetime_beginning_ept', 'total_outages_mw','planned_outages_mw','maintenance_outages_mw','forced_outages_mw','rto_total_outages_mw','rto_planned_outages_mw','rto_maintenance_outages_mw','rto_forced_outages_mw' are not in the dataframe."
        )


#######################################################################################################################################
### Plottings ###
#######################################################################################################################################


def plot_interactive_histogram(
    data,
    feature,
    feature_units,
    bins=30,
    log_transform=False,
    region_filter=True,
    region_col="region",
):
    """Generates an interactive histogram with optional log transformation and adjustable bins.

    Args:
        data (pd.DataFrame): Input dataset.
        feature (str): Column name for the feature to plot.
        feature_units (str): Units in which the plotted feature is in.
        bins (int, optional): Number of bins for the histogram. Defaults to 30.
        log_transform (bool, optional): Whether to initally apply log transformation to the feature values. Defaults to False.
        region_filter (bool, optional): Whether to include region-specific filtering. Defaults to True.
        region_col (str, optional): Column representing regions or categories. Defaults to "region".

    Raises:
        ValueError: Region feature name is not in the dataset.
        ValueError: The feature to plot is not in the dataset.
    """
    if region_filter and region_col not in data.columns:
        raise ValueError(f"'{region_col}' column must exist in the data.")
    if feature not in data.columns:
        raise ValueError(f"'{feature}' column must exist in the data.")
    if "season" not in data.columns:
        data = add_season_column(data)

    df = data.copy()

    region_dropdown = (
        widgets.Dropdown(
            options=["All"] + sorted(df[region_col].unique().tolist()),
            value="All",
            description="Region:",
        )
        if region_filter
        else None
    )
    season_dropdown = widgets.Dropdown(
        options=["All"] + sorted(df["season"].unique().tolist()),
        value="All",
        description="Season:",
    )
    bin_slider = widgets.IntSlider(
        value=bins, min=5, max=500, step=1, description="Bins:"
    )
    log_toggle = widgets.Checkbox(value=log_transform, description="Log Transform")

    output_plot = widgets.Output()
    output_summary = widgets.Output()

    fig = go.Figure()

    def update_histogram(change=None):
        # if change is not None:
        #     print("Triggered by:", change)
        filtered_data = df.copy()
        if season_dropdown.value != "All":
            filtered_data = filtered_data[
                filtered_data["season"] == season_dropdown.value
            ]
        if region_filter and region_dropdown and region_dropdown.value != "All":
            filtered_data = filtered_data[
                filtered_data["region"] == region_dropdown.value
            ]
        if log_toggle.value:
            min_value = filtered_data[feature].min()
            if min_value <= 0:
                filtered_data[f"{feature}_log"] = filtered_data[feature] - min_value + 1
            else:
                filtered_data[f"{feature}_log"] = filtered_data[feature] + 1
            feature_to_plot = f"{feature}_log"
        else:
            feature_to_plot = feature
        if not filtered_data.empty:
            fig.data = []
            fig.add_trace(
                go.Histogram(
                    x=filtered_data[feature_to_plot],
                    nbinsx=bin_slider.value,
                    marker=dict(color="blue"),
                    opacity=0.75,
                )
            )

            _title = f"{feature.replace('_', ' ').title()} Distribution "
            if (
                season_dropdown.value != "All"
                and region_dropdown is not None
                and region_dropdown.value != "All"
            ):
                _title += f"({region_dropdown.value};{season_dropdown.value})"
            elif season_dropdown.value != "All":
                _title += f"({season_dropdown.value})"
            elif (
                region_filter
                and region_dropdown is not None
                and region_dropdown.value != "All"
            ):
                if region_dropdown.value != "All":
                    _title += f"({region_dropdown.value})"

            fig.update_layout(
                title=_title,
                xaxis_title=f"{feature.replace('_', ' ').title()} {feature_units} (Log Transformed)"
                if log_toggle.value
                else feature.replace("_", " ").title() + " " + feature_units,
                yaxis_title="Count",
                template="plotly_white",
            )
        else:
            with output_plot:
                output_plot.clear_output(wait=True)
                print("No data available with the selected filters.")
            return
        with output_plot:
            output_plot.clear_output(wait=True)
            pio.show(fig, renderer="notebook")
        with output_summary:
            output_summary.clear_output(wait=True)
            if not filtered_data.empty:
                mean = filtered_data[feature].mean()
                std_dev = filtered_data[feature].std()
                percentiles = filtered_data[feature].quantile([0.25, 0.5, 0.75])
                stats_df = pd.DataFrame(
                    {
                        "Metric": [
                            "Mean",
                            "Standard Deviation",
                            "25th Percentile",
                            "Median",
                            "75th Percentile",
                        ],
                        "Value": [
                            mean,
                            std_dev,
                            percentiles[0.25],
                            percentiles[0.5],
                            percentiles[0.75],
                        ],
                    }
                )
                stats_df.set_index("Metric", inplace=True)
                display(stats_df)
            else:
                fig.data = []  # delete later if not good
                print("No data available for the selected filters.")

    if region_filter and region_dropdown:
        region_dropdown.observe(update_histogram, names="value")
    season_dropdown.observe(update_histogram, names="value")
    bin_slider.observe(update_histogram, names="value")
    log_toggle.observe(update_histogram, names="value")
    update_histogram()
    controls = (
        widgets.VBox([season_dropdown, region_dropdown, bin_slider, log_toggle])
        if region_filter and region_dropdown
        else widgets.VBox([season_dropdown, bin_slider, log_toggle])
    )
    display(widgets.HBox([controls, output_summary]))
    display(output_plot)


def plot_interactive_boxplot(
    data, feature, feature_units, region_filter=True, region_col="region"
):
    """Interactive Boxplot with filtering options, including season and outlier filtering.

    Args:
        data (pd.DataFrame): Input data containing the column to plot.
        feature (str): Feature/Column name of the specific data for visuals.
        feature_units (str): Units in which the plotted feature is in.
        region_filter (bool, optional): Whether to include region-specific filtering. Defaults to True.
        region_col (str, optional): Column representing regions or categories. Defaults to "region".

    Raises:
        ValueError: Region feature name is not in the dataset.
        ValueError: The feature to plot is not in the dataset.
    """
    if region_filter and region_col not in data.columns:
        raise ValueError(f"'{region_col}' column must exist in the data.")
    if feature not in data.columns:
        raise ValueError(f"'{feature}' column must exist in the data.")
    if "season" not in data.columns:
        data = add_season_column(data)

    df = data.copy()

    region_dropdown = (
        widgets.Dropdown(
            options=["All"] + sorted(df[region_col].unique().tolist()),
            value="All",
            description="Region:",
        )
        if region_filter
        else None
    )
    season_dropdown = widgets.Dropdown(
        options=["All"] + sorted(df["season"].unique().tolist()),
        value="All",
        description="Season:",
    )
    upper_outlier_slider = widgets.FloatSlider(
        value=df[feature].max(),
        min=df[feature].min() - 1,
        max=df[feature].max() + 1,
        step=0.1,
        description="Upper Outlier Threshold:",
        readout_format=".1f",
        style={"description_width": "initial"},
        layout=widgets.Layout(width="500px"),
    )
    lower_outlier_slider = widgets.FloatSlider(
        value=df[feature].min(),
        min=df[feature].min() - 1,
        max=df[feature].max() + 1,
        step=0.1,
        description="Lower Outlier Threshold:",
        readout_format=".1f",
        style={"description_width": "initial"},
        layout=widgets.Layout(width="500px"),
    )

    output_plot = widgets.Output()
    fig = go.Figure()

    def update_boxplot(change=None):
        # if change is not None:
        #     print("Triggered by:", change)
        filtered_data = df.copy()
        if region_filter and region_dropdown and region_dropdown.value != "All":
            filtered_data = filtered_data[
                filtered_data[region_col] == region_dropdown.value
            ]
        if season_dropdown.value != "All":
            filtered_data = filtered_data[
                filtered_data["season"] == season_dropdown.value
            ]
        filtered_data = filtered_data[
            (filtered_data[feature] >= lower_outlier_slider.value)
            & (filtered_data[feature] <= upper_outlier_slider.value)
        ]
        if not filtered_data.empty:
            fig.data = []
            if region_filter and region_dropdown:
                fig.add_trace(
                    go.Box(
                        y=filtered_data[feature],
                        x=filtered_data[region_col],
                        boxmean=True,
                        name="Boxplot",
                        marker=dict(color="blue"),
                    )
                )
            else:
                fig.add_trace(
                    go.Box(
                        y=filtered_data[feature],
                        boxmean=True,
                        name="PJM",
                        marker=dict(color="blue"),
                    )
                )
            _title = f"{feature.replace('_', ' ').title()} Boxplot(s) "
            if (
                season_dropdown.value != "All"
                and region_dropdown is not None
                and region_dropdown.value != "All"
            ):
                _title += f"({region_dropdown.value};{season_dropdown.value})"
            elif season_dropdown.value != "All":
                _title += f"({season_dropdown.value})"
            elif (
                region_filter
                and region_dropdown is not None
                and region_dropdown.value != "All"
            ):
                if region_dropdown.value != "All":
                    _title += f"({region_dropdown.value})"

            fig.update_layout(
                title=_title,
                xaxis_title="Region" if region_filter else "",
                yaxis_title=f"{feature_units}",
                template="plotly_white",
            )
        else:
            fig.data = []
            with output_plot:
                output_plot.clear_output(wait=True)
                print("No data available with the selected filters.")
            return
        with output_plot:
            output_plot.clear_output(wait=True)
            pio.show(fig, renderer="notebook")

    if region_filter and region_dropdown:
        region_dropdown.observe(update_boxplot, names="value")
    season_dropdown.observe(update_boxplot, names="value")
    upper_outlier_slider.observe(update_boxplot, names="value")
    lower_outlier_slider.observe(update_boxplot, names="value")
    update_boxplot()
    controls = (
        widgets.VBox([season_dropdown, region_dropdown])
        if region_filter and region_dropdown
        else widgets.VBox([season_dropdown])
    )
    outlier_controls = widgets.HBox([lower_outlier_slider, upper_outlier_slider])
    display(widgets.HBox([controls, outlier_controls]))
    display(output_plot)


def add_summer_winter_bands_to_plot(fig, data, datetime_col):
    """Adds vertical bands for summer and winter months to a Plotly figure.

    Args:
        fig (plotly.graph_objects.Figure): Plotly figure to add bands to.
        data (pd.DataFrame): Dataframe containing the date range for the plot.
        datetime_col (str): Column name of the datetime column in the data.
    """
    data_dte = pd.to_datetime(data[datetime_col].copy())
    first_day = data_dte.min()
    last_day = data_dte.max()
    start_year = data_dte.dt.year.min()
    end_year = data_dte.dt.year.max()

    for year in range(start_year, end_year + 1):
        summer_start = pd.Timestamp(f"{year}-06-01")
        summer_end = pd.Timestamp(f"{year}-08-31")
        if summer_start <= last_day and summer_end >= first_day:
            fig.add_vrect(
                x0=summer_start,
                x1=summer_end,
                fillcolor="orange",
                opacity=0.15,
                layer="below",
                line_width=0,
            )
        winter_start = pd.Timestamp(f"{year}-12-01")
        winter_end = pd.Timestamp(f"{year+1}-02-28")
        if winter_start <= last_day and winter_end >= first_day:
            fig.add_vrect(
                x0=winter_start,
                x1=winter_end,
                fillcolor="blue",
                opacity=0.15,
                layer="below",
                line_width=0,
            )
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(size=10, color="orange"),
            name="Summer (June-Aug)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(size=10, color="blue"),
            name="Winter (Dec-Feb)",
        )
    )


def plot_interactive_timeseries(
    data,
    feature,
    feature_units,
    seasonal_bands=True,
    region_filter=True,
    region_col="region",
    datetime_col="datetime_beginning_ept",
):
    """Interactive Time Series Plot with date range and region filtering.

    Args:
        data (pd.DataFrame): Input data containing the time series.
        feature (str): Column name for the feature to plot.
        feature_units (str): Units in which the plotted feature is in.
        seasonal_bands (bool, optional): Whether to include seasonal bands. Defaults to True.
        region_filter (bool, optional): Whether to include region-specific filtering. Defaults to True.
        region_col (str): Column representing regions or categories. Defaults to "region".
        datetime_col (str): Column containing datetime values. Defaults to "datetime_beginning_ept".

    Raises:
        ValueError: Region feature name is not in the dataset.
        ValueError: The feature to plot is not in the dataset.
    """
    if region_filter and region_col not in data.columns:
        raise ValueError(f"'{region_col}' column must exist in the data.")
    if feature not in data.columns:
        raise ValueError(f"'{feature}' column must exist in the data.")
    if "season" not in data.columns:
        data = add_season_column(data)

    df = data.copy()
    df[datetime_col] = pd.to_datetime(df[datetime_col])
    region_dropdown = (
        widgets.Dropdown(
            options=["All"] + sorted(df[region_col].unique().tolist()),
            value="All",
            description="Region:",
        )
        if region_filter
        else None
    )
    start_date_input = widgets.Text(
        value=df[datetime_col].min().strftime("%Y-%m-%d"),
        description="Start Date:",
        placeholder="YYYY-MM-DD",
        layout=widgets.Layout(width="300px"),
    )
    end_date_input = widgets.Text(
        value=df[datetime_col].max().strftime("%Y-%m-%d"),
        description="End Date:",
        placeholder="YYYY-MM-DD",
        layout=widgets.Layout(width="300px"),
    )
    season_toggle = widgets.Checkbox(
        value=seasonal_bands, description="Add Seasonal Bands"
    )

    output_plot = widgets.Output()
    fig = go.Figure()

    def update_timeseries(change=None):
        # if change is not None:
        #     print("Triggered by:", change)
        filtered_data = df.copy()
        if region_filter and region_dropdown and region_dropdown.value != "All":
            filtered_data = filtered_data[
                filtered_data[region_col] == region_dropdown.value
            ]
        try:
            start_date = pd.to_datetime(start_date_input.value)
            end_date = pd.to_datetime(end_date_input.value)
        except ValueError:
            with output_plot:
                output_plot.clear_output(wait=True)
                print("Invalid date format. Please use YYYY-MM-DD.")
            return
        filtered_data = filtered_data[
            (filtered_data[datetime_col] >= start_date)
            & (filtered_data[datetime_col] <= end_date)
        ]
        if not filtered_data.empty:
            fig.data = []
            if region_filter and region_dropdown:
                for region, group in filtered_data.groupby(region_col):
                    fig.add_trace(
                        go.Scatter(
                            x=group[datetime_col],
                            y=group[feature],
                            mode="lines",
                            name=region,
                        )
                    )
            else:
                fig.add_trace(
                    go.Scatter(
                        x=filtered_data[datetime_col],
                        y=filtered_data[feature],
                        mode="lines",
                    )
                )

            _title = f"{feature.replace('_', ' ').title()} Time Series "
            if region_filter and region_dropdown.value != "All":
                _title += f"({region_dropdown.value})"

            fig.update_layout(
                title=_title,
                xaxis_title="Datetime",
                yaxis_title=f"{feature_units}",
                xaxis=dict(range=[start_date, end_date]),
                template="plotly_white",
            )
            if season_toggle.value:
                add_summer_winter_bands_to_plot(fig, filtered_data, datetime_col)
            else:
                fig.layout.shapes = []
        else:
            with output_plot:
                output_plot.clear_output(wait=True)
                print("No data available with the selected filters.")
            return
        with output_plot:
            output_plot.clear_output(wait=True)
            pio.show(fig, renderer="notebook")

    if region_filter and region_dropdown:
        region_dropdown.observe(update_timeseries, names="value")
    start_date_input.observe(update_timeseries, names="value")
    end_date_input.observe(update_timeseries, names="value")
    season_toggle.observe(update_timeseries, names="value")
    update_timeseries()
    date_range_inputs = widgets.HBox([start_date_input, end_date_input])
    controls = (
        widgets.HBox([region_dropdown, season_toggle, date_range_inputs])
        if region_filter and region_dropdown
        else widgets.HBox([season_toggle, date_range_inputs])
    )
    display(controls, output_plot)


def plot_interactive_scatter_two_features(
    data,
    feature_x,
    feature_y,
    x_units="",
    y_units="",
    region_filter=True,
    region_col="region",
    add_best_fit=False,
):
    """
    Interactive Scatter Plot of Two Features with Region Filtering.

    Args:
        data (pd.DataFrame): Input dataset.
        feature_x (str): Column name for the feature to plot on the x-axis.
        feature_y (str): Column name for the feature to plot on the y-axis.
        x_units (str): Units for the x-axis feature.
        y_units (str): Units for the y-axis feature.
        region_filter (bool, optional): Whether to allow region-specific filtering. Defaults to True.
        region_col (str, optional): Column for regions or categories. Defaults to "region".
        add_best_fit (bool, optional): Whether to add a line of best-fit. Defaults to False.

    Raises:
        ValueError: Region feature name is not in the dataset.
        ValueError: The two features to compare are not in the dataset.
    """
    if region_filter and region_col not in data.columns:
        raise ValueError(f"'{region_col}' column must exist in the data.")
    if feature_x not in data.columns or feature_y not in data.columns:
        raise ValueError(
            f"'{feature_x}' and '{feature_y}' columns must exist in the data."
        )
    if "season" not in data.columns:
        data = add_season_column(data)

    df = data.copy()

    region_dropdown = (
        widgets.Dropdown(
            options=["All"] + sorted(df[region_col].unique().tolist()),
            value="All",
            description="Region:",
        )
        if region_filter
        else None
    )
    season_dropdown = widgets.Dropdown(
        options=["All"] + sorted(df["season"].unique().tolist()),
        value="All",
        description="Season:",
    )
    trendline_checkbox = widgets.Checkbox(
        value=add_best_fit, description="Add Trendline"
    )
    output_plot = widgets.Output()
    fig = go.Figure()

    def update_scatter(change=None):
        filtered_data = df.copy()
        if region_filter and region_dropdown and region_dropdown.value != "All":
            filtered_data = filtered_data[
                filtered_data[region_col] == region_dropdown.value
            ]
        if season_dropdown.value != "All":
            filtered_data = filtered_data[
                filtered_data["season"] == season_dropdown.value
            ]
        if not filtered_data.empty:
            fig.data = []
            if region_filter and region_dropdown:
                for region, region_group in filtered_data.groupby(region_col):
                    fig.add_trace(
                        go.Scatter(
                            x=region_group[feature_x],
                            y=region_group[feature_y],
                            mode="markers",
                            name=region,
                            marker=dict(size=8, opacity=0.6),
                        )
                    )
            else:
                fig.add_trace(
                    go.Scatter(
                        x=filtered_data[feature_x],
                        y=filtered_data[feature_y],
                        mode="markers",
                        name="Data Points",
                        marker=dict(size=6, color="blue", opacity=0.7),
                    )
                )
            if trendline_checkbox.value:
                x_values = filtered_data[feature_x]
                y_values = filtered_data[feature_y]
                slope, intercept, r_value, p_value, std_err = linregress(
                    x_values, y_values
                )
                trendline_x = np.linspace(x_values.min(), x_values.max(), 100)
                trendline_y = slope * trendline_x + intercept
                fig.add_trace(
                    go.Scatter(
                        x=trendline_x,
                        y=trendline_y,
                        mode="lines",
                        name="Trendline",
                        line=dict(color="red", dash="dash"),
                    )
                )
            _title = f"Scatter Plot of {feature_x.replace('_', ' ').title()} vs {feature_y.replace('_', ' ').title()} \n"
            if region_filter and region_dropdown and region_dropdown.value != "All":
                _title += f" (Region: {region_dropdown.value})"
            fig.update_layout(
                title=_title,
                xaxis_title=f"{feature_x.replace('_', ' ').title()} {x_units}",
                yaxis_title=f"{feature_y.replace('_', ' ').title()} {y_units}",
                template="plotly_white",
                width=1000,
                height=800,
            )
        else:
            with output_plot:
                output_plot.clear_output(wait=True)
                print("No data available with the selected filters.")
            return

        with output_plot:
            output_plot.clear_output(wait=True)
            pio.show(fig, renderer="notebook")

    if region_filter and region_dropdown:
        region_dropdown.observe(update_scatter, names="value")
    season_dropdown.observe(update_scatter, names="value")
    trendline_checkbox.observe(update_scatter, names="value")
    update_scatter()
    controls = (
        widgets.HBox([region_dropdown, season_dropdown, trendline_checkbox])
        if region_filter
        else widgets.HBox([season_dropdown, trendline_checkbox])
    )
    display(controls)
    display(output_plot)


def plot_interactive_timeseries_scatter_outliers(
    data,
    feature,
    feature_units,
    seasonal_bands=True,
    region_filter=True,
    region_col="region",
    datetime_col="datetime_beginning_ept",
):
    """Interactive Scatterplot Time Series with Outlier Detection Only.

    Args:
        data (pd.DataFrame): Input dataset containing the time series.
        feature (str): Column name for the feature to plot.
        feature_units (str): Units for the feature.
        seasonal_bands (bool, optional): Whether to include summer and winter bands in the plot. Defaults to True.
        region_filter (bool, optional): Whether to allow region-specific filtering. Defaults to True.
        region_col (str, optional): Column for regions or categories. Defaults to "region".
        datetime_col (str, optional): Column containing datetime values. Defaults to "datetime_beginning_ept".

    Raises:
        ValueError: Region feature name is not in the dataset.
        ValueError: The feature to plot is not in the dataset.
    """
    if region_filter and region_col not in data.columns:
        raise ValueError(f"'{region_col}' column must exist in the data.")
    if feature not in data.columns:
        raise ValueError(f"'{feature}' column must exist in the data.")
    if "season" not in data.columns:
        data = add_season_column(data)

    df = data.copy()
    df[datetime_col] = pd.to_datetime(df[datetime_col])

    region_dropdown = (
        widgets.Dropdown(
            options=["All"] + sorted(df[region_col].unique().tolist()),
            value="All",
            description="Region:",
        )
        if region_filter
        else None
    )
    start_date_input = widgets.Text(
        value=df[datetime_col].min().strftime("%Y-%m-%d"),
        description="Start Date:",
        placeholder="YYYY-MM-DD",
        layout=widgets.Layout(width="300px"),
    )
    end_date_input = widgets.Text(
        value=df[datetime_col].max().strftime("%Y-%m-%d"),
        description="End Date:",
        placeholder="YYYY-MM-DD",
        layout=widgets.Layout(width="300px"),
    )
    outlier_method_dropdown = widgets.Dropdown(
        options=["IQR-Based", "Dollar Threshold"],
        value="IQR-Based",
        description="Outlier Method:",
        style={"description_width": "initial"},
    )
    multiplier_slider = widgets.FloatSlider(
        value=1.5,
        min=1.0,
        max=5.0,
        step=0.1,
        description="IQR Multiplier:",
        layout=widgets.Layout(width="400px"),
    )
    dollar_amt_slider = widgets.IntSlider(
        value=50,
        min=10,
        max=500,
        step=10,
        description="Dollar Threshold:",
        layout=widgets.Layout(width="400px"),
    )
    seasonal_toggle = widgets.Checkbox(
        value=seasonal_bands, description="Add Seasonal Bands"
    )
    output_plot = widgets.Output()
    fig = go.Figure()

    def update_scatter(change=None):
        # if change is not None:
        #     print("Triggered by:", change)
        filtered_data = df.copy()
        if region_filter and region_dropdown and region_dropdown.value != "All":
            filtered_data = filtered_data[
                filtered_data[region_col] == region_dropdown.value
            ]
        try:
            start_date = pd.to_datetime(start_date_input.value)
            end_date = pd.to_datetime(end_date_input.value)
        except ValueError:
            with output_plot:
                output_plot.clear_output(wait=True)
                print("Invalid date format. Please use YYYY-MM-DD.")
            return

        filtered_data = filtered_data[
            (filtered_data[datetime_col] >= start_date)
            & (filtered_data[datetime_col] <= end_date)
        ]
        if outlier_method_dropdown.value == "IQR-Based":
            outliers = outlier_df_iqr(
                filtered_data, col_name=feature, multiplier=multiplier_slider.value
            )
        elif outlier_method_dropdown.value == "Dollar Threshold":
            outliers = outlier_df_by_dollar_amt(
                filtered_data, col_name=feature, dollar_amt=dollar_amt_slider.value
            )
        else:
            outliers = pd.DataFrame()

        if not outliers.empty:
            fig.data = []
            fig.add_trace(
                go.Scatter(
                    x=outliers[datetime_col],
                    y=outliers[feature],
                    mode="markers",
                    name="Outliers",
                    marker=dict(color="red", size=8, opacity=0.8, symbol="x"),
                )
            )

            _title = f"{feature.replace('_', ' ').title()} Outliers Only "
            if region_filter and region_dropdown.value != "All":
                _title += f"({outlier_method_dropdown.value};{region_dropdown.value})"
            else:
                _title += f"({outlier_method_dropdown.value})"

            fig.update_layout(
                title=_title,
                xaxis_title="Datetime",
                yaxis_title=f"{feature_units}",
                xaxis=dict(range=[start_date, end_date]),
                template="plotly_white",
            )
            if seasonal_toggle.value:
                add_summer_winter_bands_to_plot(fig, outliers, datetime_col)
            else:
                fig.layout.shapes = []
        else:
            with output_plot:
                output_plot.clear_output(wait=True)
                print("No outliers detected within the selected parameters.")
            return

        with output_plot:
            output_plot.clear_output(wait=True)
            pio.show(fig, renderer="notebook")

    if region_filter and region_dropdown:
        region_dropdown.observe(update_scatter, names="value")
    start_date_input.observe(update_scatter, names="value")
    end_date_input.observe(update_scatter, names="value")
    outlier_method_dropdown.observe(update_scatter, names="value")
    multiplier_slider.observe(update_scatter, names="value")
    dollar_amt_slider.observe(update_scatter, names="value")
    seasonal_toggle.observe(update_scatter, names="value")
    update_scatter()
    date_range_inputs = widgets.HBox([start_date_input, end_date_input])
    controls = (
        widgets.HBox([region_dropdown, seasonal_toggle, date_range_inputs])
        if region_filter and region_dropdown
        else widgets.HBox([seasonal_toggle, date_range_inputs])
    )
    display(
        widgets.VBox(
            [
                controls,
                outlier_method_dropdown,
                multiplier_slider,
                dollar_amt_slider,
            ],
        )
    )
    display(output_plot)


def plot_interactive_rolling_timeseries(
    data,
    feature,
    feature_units,
    seasonal_bands=True,
    hourly_roll=True,
    region_filter=True,
    region_col="region",
    datetime_col="datetime_beginning_ept",
):
    """Interactive Rolling Hourly Timeseries Plot by Region.

    Args:
        data (pd.DataFrame): Input data containing the time series.
        feature (str): Column name for the feature to plot.
        feature_units (str): Units for the feature.
        seasonal_bands (bool, optional): Whether to include seasonal bands. Defaults to True.
        hourly_roll (bool, optional): If the date granularity is in hours. Defaults to True.
        region_filter (bool, optional): Whether to include region-specific filtering. Defaults to True.
        region_col (str): Column representing regions or categories. Defaults to "region".
        datetime_col (str): Column containing datetime values. Defaults to "datetime_beginning_ept".

    Raises:
        ValueError: Region feature name is not in the dataset.
        ValueError: The feature to plot is not in the dataset.
    """
    if region_filter and region_col not in data.columns:
        raise ValueError(f"'{region_col}' column must exist in the data.")
    if feature not in data.columns:
        raise ValueError(f"'{feature}' column must exist in the data.")
    if "season" not in data.columns:
        data = add_season_column(data)

    df = data.copy()
    df[datetime_col] = pd.to_datetime(df[datetime_col])
    region_dropdown = (
        widgets.Dropdown(
            options=["All"] + sorted(df[region_col].unique().tolist()),
            value="All",
            description="Region:",
        )
        if region_filter
        else None
    )
    start_date_input = widgets.Text(
        value=df[datetime_col].min().strftime("%Y-%m-%d"),
        description="Start Date:",
        placeholder="YYYY-MM-DD",
        layout=widgets.Layout(width="300px"),
    )
    end_date_input = widgets.Text(
        value=df[datetime_col].max().strftime("%Y-%m-%d"),
        description="End Date:",
        placeholder="YYYY-MM-DD",
        layout=widgets.Layout(width="300px"),
    )
    rolling_window_slider = widgets.IntSlider(
        value=1,
        min=1,
        max=14 * 24,
        step=1,
        description="Rolling Window (Hours):"
        if hourly_roll
        else "Rolling Window (Days):",
        layout=widgets.Layout(width="400px"),
        style={"description_width": "initial"},
    )
    seasonal_toggle = widgets.Checkbox(
        value=seasonal_bands, description="Add Seasonal Bands"
    )

    output_plot = widgets.Output()
    fig = go.Figure()

    def update_rolling_plot(change=None):
        filtered_data = df.copy()
        if region_filter and region_dropdown and region_dropdown.value != "All":
            filtered_data = filtered_data[
                filtered_data[region_col] == region_dropdown.value
            ]
        try:
            start_date = pd.to_datetime(start_date_input.value)
            end_date = pd.to_datetime(end_date_input.value)
        except ValueError:
            with output_plot:
                output_plot.clear_output(wait=True)
                print("Invalid date format. Please use YYYY-MM-DD.")
            return
        filtered_data = filtered_data[
            (filtered_data[datetime_col] >= start_date)
            & (filtered_data[datetime_col] <= end_date)
        ]
        if not filtered_data.empty:
            fig.data = []
            agg_data = (
                filtered_data.groupby([region_col, datetime_col])[feature]
                .mean()
                .reset_index()
            )
            for region, group in agg_data.groupby(region_col):
                group = group.sort_values(datetime_col)
                group["rolling_mean"] = (
                    group[feature].rolling(window=rolling_window_slider.value).mean()
                )
                fig.add_trace(
                    go.Scatter(
                        x=group[datetime_col],
                        y=group["rolling_mean"],
                        mode="lines",
                        name=region,
                    )
                )

                _title = (
                    f"{feature.replace('_', ' ').title()} Rolling {rolling_window_slider.value} Window (Hourly) Average Time Series "
                    if hourly_roll
                    else f"{feature.replace('_', ' ').title()} Rolling {rolling_window_slider.value} Window (Daily) Average Time Series "
                )
                if region_filter and region_dropdown.value != "All":
                    _title += f"({region_dropdown.value})"

                fig.update_layout(
                    title=_title,
                    xaxis_title="Datetime",
                    yaxis_title=f"{feature_units}",
                    xaxis=dict(range=[start_date, end_date]),
                    template="plotly_white",
                )
            if seasonal_toggle.value:
                add_summer_winter_bands_to_plot(fig, agg_data, datetime_col)
            else:
                fig.layout.shapes = []
        else:
            with output_plot:
                output_plot.clear_output(wait=True)
                print("No data available with the selected filters.")
            return

        with output_plot:
            output_plot.clear_output(wait=True)
            pio.show(fig, renderer="notebook")

    if region_filter and region_dropdown:
        region_dropdown.observe(update_rolling_plot, names="value")
    start_date_input.observe(update_rolling_plot, names="value")
    end_date_input.observe(update_rolling_plot, names="value")
    rolling_window_slider.observe(update_rolling_plot, names="value")
    seasonal_toggle.observe(update_rolling_plot, names="value")
    update_rolling_plot()
    date_range_inputs = widgets.HBox([start_date_input, end_date_input])
    controls = (
        widgets.VBox(
            [
                region_dropdown,
                date_range_inputs,
                rolling_window_slider,
                seasonal_toggle,
            ]
        )
        if region_filter and region_dropdown
        else widgets.VBox(
            [
                date_range_inputs,
                rolling_window_slider,
                seasonal_toggle,
            ]
        )
    )
    display(controls, output_plot)


def plot_interactive_calendar_heatmap(
    data,
    feature,
    feature_units,
    region_filter=True,
    region_col="region",
    datetime_col="datetime_beginning_ept",
):
    """Generates an Interactive Calendar Heatmap.

    Args:
        data (pd.DataFrame): Input dataset.
        feature (str): Column name for the feature to plot.
        feature_units (str): Units in which the plotted feature is in.
        region_filter (bool, optional): Whether to include region-specific filtering. Defaults to True.
        region_col (str): Column representing regions or categories. Defaults to "region".
        datetime_col (str): Column containing datetime values. Defaults to "datetime_beginning_ept".

    Raises:
        ValueError: Region feature name is not in the dataset.
        ValueError: The feature to plot is not in the dataset.
    """
    if region_filter and region_col not in data.columns:
        raise ValueError(f"'{region_col}' column must exist in the data.")
    if feature not in data.columns:
        raise ValueError(f"'{feature}' column must exist in the data.")
    if "season" not in data.columns:
        data = add_season_column(data)

    df = data.copy()
    df[datetime_col] = pd.to_datetime(df[datetime_col])
    df["day"] = df[datetime_col].dt.day
    df["month"] = df[datetime_col].dt.month
    df["year"] = df[datetime_col].dt.year

    region_dropdown = (
        widgets.Dropdown(
            options=["All"] + sorted(df[region_col].unique().tolist()),
            value="All",
            description="Region:",
        )
        if region_filter
        else None
    )

    season_dropdown = widgets.Dropdown(
        options=["All"] + sorted(df["season"].unique().tolist()),
        value="All",
        description="Season:",
    )
    year_dropdown = widgets.Dropdown(
        options=["All"] + sorted(df["year"].unique().tolist()),
        value="All",
        description="Year:",
    )
    month_dropdown = widgets.Dropdown(
        options=["All"] + sorted(df["month"].unique().tolist()),
        value="All",
        description="Day:",
    )
    day_dropdown = widgets.Dropdown(
        options=["All"] + sorted(df["day"].unique().tolist()),
        value="All",
        description="Day:",
    )
    output_plot = widgets.Output()
    # fig = go.Figure()

    def update_calendar_heatmap(change=None):
        filtered_data = df.copy()
        if region_filter and region_dropdown and region_dropdown.value != "All":
            filtered_data = filtered_data[
                filtered_data[region_col] == region_dropdown.value
            ]
        if season_dropdown.value != "All":
            filtered_data = filtered_data[
                filtered_data["season"] == season_dropdown.value
            ]
        if (
            year_dropdown.value != "All"
            and month_dropdown.value != "All"
            and day_dropdown.value != "All"
        ):
            filtered_data = filtered_data[
                (filtered_data["year"] == int(year_dropdown.value))
                & (filtered_data["month"] == int(month_dropdown.value))
                & (filtered_data["day"] == int(day_dropdown.value))
            ]
        elif year_dropdown.value != "All" and month_dropdown.value != "All":
            filtered_data = filtered_data[
                (filtered_data["year"] == int(year_dropdown.value))
                & (filtered_data["month"] == int(month_dropdown.value))
            ]
        elif year_dropdown.value != "All" and day_dropdown.value != "All":
            filtered_data = filtered_data[
                (filtered_data["year"] == int(year_dropdown.value))
                & (filtered_data["day"] == int(day_dropdown.value))
            ]
        elif month_dropdown.value != "All" and day_dropdown.value != "All":
            filtered_data = filtered_data[
                (filtered_data["month"] == int(month_dropdown.value))
                & (filtered_data["day"] == int(day_dropdown.value))
            ]
        elif year_dropdown.value != "All":
            filtered_data = filtered_data[
                filtered_data["year"] == int(year_dropdown.value)
            ]
        elif month_dropdown.value != "All":
            filtered_data = filtered_data[
                filtered_data["month"] == int(month_dropdown.value)
            ]
        elif day_dropdown.value != "All":
            filtered_data = filtered_data[
                filtered_data["day"] == int(day_dropdown.value)
            ]

        if not filtered_data.empty:
            fig = []
            heatmap_data = filtered_data.pivot_table(
                index="day",
                columns=["year", "month"],
                values=feature,
            ).fillna(0)
            x_labels = [f"{col[0]}-{col[1]}" for col in heatmap_data.columns]
            tickvals = list(range(len(x_labels)))
            tickvals = list(range(len(heatmap_data.columns)))
            fig = px.imshow(
                heatmap_data,
                color_continuous_scale="YlGnBu",
                labels={"color": f"{feature.replace("_"," ").title()} {feature_units}"},
                x=list(range(heatmap_data.shape[1])),
            )
            _title = f"Calendar Heatmap of {feature.replace("_"," ").title()} "
            if (
                season_dropdown.value != "All"
                and region_dropdown is not None
                and region_dropdown.value != "All"
            ):
                _title += f"({region_dropdown.value};{season_dropdown.value})"
            elif season_dropdown.value != "All":
                _title += f"({season_dropdown.value})"
            elif (
                region_filter
                and region_dropdown is not None
                and region_dropdown.value != "All"
            ):
                if region_dropdown.value != "All":
                    _title += f"({region_dropdown.value})"

            fig.update_layout(
                title=_title,
                xaxis=dict(
                    title="Year-Month",
                    tickmode="array",
                    tickvals=tickvals,
                    ticktext=x_labels,
                    automargin=True,
                ),
                yaxis=dict(
                    title="Day of Month",
                    tickmode="array",
                    tickvals=list(range(1, 32)),
                    ticktext=list(range(1, 32)),
                ),
                width=1200,
                height=800,
            )
        else:
            fig.data = []
            with output_plot:
                output_plot.clear_output(wait=True)
                print("No data available with the selected filters.")
            return
        with output_plot:
            output_plot.clear_output(wait=True)
            pio.show(fig, renderer="notebook")

    if region_filter and region_dropdown:
        region_dropdown.observe(update_calendar_heatmap, names="value")
    season_dropdown.observe(update_calendar_heatmap, names="value")
    year_dropdown.observe(update_calendar_heatmap, names="value")
    month_dropdown.observe(update_calendar_heatmap, names="value")
    day_dropdown.observe(update_calendar_heatmap, names="value")
    update_calendar_heatmap()
    controls = (
        widgets.HBox(
            [
                season_dropdown,
                region_dropdown,
                year_dropdown,
                month_dropdown,
                day_dropdown,
            ]
        )
        if region_filter and region_dropdown
        else widgets.VBox(
            [season_dropdown, year_dropdown, month_dropdown, day_dropdown]
        )
    )
    display(controls)
    display(output_plot)


def plot_interactive_feature_count_breakdown_by_season(
    data,
    features,
    region_filter=True,
    region_col="region",
):
    """Interactive barplots comparing two features across seasons with optional region-based filtering.

    Args:
        data (pd.DataFrame): Input dataset.
        features (list of str): List of features from the dataset to compare.
        region_filter (bool, optional): _description_. Defaults to True.
        region_col (str, optional): _description_. Defaults to "region".

    Raises:
        ValueError: Region feature name is not in the dataset.
        ValueError: The features to plot are not in the dataset.
    """
    if region_filter and region_col not in data.columns:
        raise ValueError(f"'{region_col}' column must exist in the data.")
    if not all(pd.Series(features).isin(data.columns)):
        raise ValueError(f"'{",".join(features)}' column must exist in the data.")
    if "season" not in data.columns:
        data = add_season_column(data)

    df = data.copy()
    region_dropdown = (
        widgets.Dropdown(
            options=["All"] + sorted(df[region_col].unique().tolist()),
            value="All",
            description="Region:",
        )
        if region_filter
        else None
    )
    season_dropdown = widgets.Dropdown(
        options=["All"] + sorted(df["season"].unique().tolist()),
        value="All",
        description="Season:",
    )
    output_plot = widgets.Output()
    fig = go.Figure()

    def update_plot(change=None):
        filtered_data = df.copy()
        if region_filter and region_dropdown and region_dropdown.value != "All":
            filtered_data = filtered_data[
                filtered_data[region_col] == region_dropdown.value
            ]
        if season_dropdown.value != "All":
            filtered_data = filtered_data[
                filtered_data["season"] == season_dropdown.value
            ]
        if not filtered_data.empty:
            fig.data = []
            bar_width = 0.35
            seasonal_summary = (
                filtered_data.groupby("season")[features].sum().reset_index()
            )
            for feature in features:
                fig.add_trace(
                    go.Bar(
                        x=seasonal_summary["season"],
                        y=seasonal_summary[feature],
                        name=f"{feature.replace('_', ' ').title()}",
                        width=bar_width,
                    )
                )

            _title = "Seasonal Breakdown Barchart "
            if region_filter and region_dropdown and region_dropdown.value != "All":
                _title += f" (Region: {region_dropdown.value})"
            fig.update_layout(
                title=_title,
                xaxis=dict(title="Season(s)"),
                yaxis=dict(title="Count"),
                legend=dict(title="Features"),
                barmode="group",
                template="plotly_white",
                width=800,
                height=600,
            )
        else:
            with output_plot:
                output_plot.clear_output(wait=True)
                print("No data available with the selected filters.")
            return

        with output_plot:
            output_plot.clear_output(wait=True)
            pio.show(fig, renderer="notebook")

    if region_filter and region_dropdown:
        region_dropdown.observe(update_plot, names="value")
    season_dropdown.observe(update_plot, names="value")
    update_plot()
    controls = (
        widgets.HBox([region_dropdown, season_dropdown])
        if region_filter
        else widgets.HBox([season_dropdown])
    )
    display(controls)
    display(output_plot)


def plot_interactive_frequency_timeseries(
    data,
    features,
    region_filter=True,
    region_col="region",
    datetime_col="datetime_beginning_ept",
):
    """Generates a interactive timeseries analyzing the frequency of multiple features with optional region-based filtering and customizable time aggregation granularity.

    Args:
        data (pd.DataFrame): Input dataset.
        features (list of str): List of features from the dataset to compare.
        region_filter (bool, optional): Whether to include region-specific filtering. Defaults to True.
        region_col (str): Column representing regions or categories. Defaults to "region".
        datetime_col (str): Column containing datetime values. Defaults to "datetime_beginning_ept".

    Raises:
        ValueError: Region feature name is not in the dataset.
        ValueError: The features to plot are not in the dataset.
    """
    if region_filter and region_col not in data.columns:
        raise ValueError(f"'{region_col}' column must exist in the data.")
    if not all(pd.Series(features).isin(data.columns)):
        raise ValueError(f"'{",".join(features)}' column must exist in the data.")
    if "season" not in data.columns:
        data = add_season_column(data)

    df = data.copy()
    df[datetime_col] = pd.to_datetime(df[datetime_col])

    freq_label_map = {
        "Monthly (M)": "M",
        "By Month (All Years Combined)": "month",
        "By Day (All Years Combined)": "day",
        "By Hour (All Years Combined)": "hour",
        "By Year": "year",
    }
    freq_dropdown = widgets.Dropdown(
        options=list(freq_label_map.keys()),
        value="Monthly (M)",
        description="Frequency Aggregation:",
        style={"description_width": "initial"},
    )
    region_dropdown = (
        widgets.Dropdown(
            options=["All"] + sorted(df[region_col].unique().tolist()),
            value="All",
            description="Region:",
        )
        if region_filter
        else None
    )

    output_plot = widgets.Output()
    fig = go.Figure()

    def update_freq_plot(change=None):
        freq = freq_label_map[freq_dropdown.value]
        filtered_data = data.copy()
        if region_filter and region_dropdown and region_dropdown.value != "All":
            filtered_data = filtered_data[
                filtered_data[region_col] == region_dropdown.value
            ]

        time_data = filtered_data.copy()
        if freq in ["month", "day", "hour", "year"]:
            time_data["period"] = pd.to_datetime(
                time_data[datetime_col]
            ).dt.__getattribute__(freq)
        else:  # if Year-Month (ie. M) is selected
            time_data["period"] = pd.to_datetime(time_data[datetime_col]).dt.to_period(
                freq
            )

        frequency = time_data.groupby("period")[features].sum().reset_index()
        frequency["period"] = frequency["period"].astype(str)

        if not frequency.empty:
            fig.data = []
            for feature in features:
                fig.add_trace(
                    go.Scatter(
                        x=frequency["period"],
                        y=frequency[feature],
                        mode="lines+markers",
                        name=feature.replace("_", " ").title(),
                    )
                )
            _title = (
                f"Frequency of Features Over Time ({freq.capitalize()}-Aggregated) "
            )
            if region_filter and region_dropdown and region_dropdown.value != "All":
                _title += f"(Region: {region_dropdown.value})"
            fig.update_layout(
                title=_title,
                xaxis=dict(title="Time Period", tickangle=45),
                yaxis=dict(title="Frequency/Count"),
                legend=dict(title="Features"),
                template="plotly_white",
                width=900,
                height=500,
            )
        else:
            with output_plot:
                output_plot.clear_output(wait=True)
                print("No data available with the selected filters.")
            return

        with output_plot:
            output_plot.clear_output(wait=True)
            pio.show(fig, renderer="notebook")

    if region_filter and region_dropdown:
        region_dropdown.observe(update_freq_plot, names="value")
    freq_dropdown.observe(update_freq_plot, names="value")
    update_freq_plot()
    controls = (
        widgets.VBox(
            [
                region_dropdown,
                freq_dropdown,
            ]
        )
        if region_filter and region_dropdown
        else widgets.VBox(
            [
                freq_dropdown,
            ]
        )
    )
    display(controls, output_plot)


def plot_interactive_timeseries_two_features(
    data,
    feature1,
    feature2,
    feature1_units,
    feature2_units,
    seasonal_bands=True,
    region_filter=True,
    region_col="region",
    datetime_col="datetime_beginning_ept",
):
    """
    Interactive Time Series Plot for Two Features with Dual Y-Axes, date range, and region filtering.

    Args:
        data (pd.DataFrame): Input data containing the time series.
        feature1 (str): Column name for the first feature to plot in comparison.
        feature2 (str): Column name for the second feature to plot in comparison.
        feature1_units (str): Units for the first feature (displayed on the left y-axis).
        feature2_units (str): Units for the second feature (displayed on the right y-axis).
        seasonal_bands (bool, optional): Whether to include seasonal bands. Defaults to True.
        region_filter (bool, optional): Whether to include region-specific filtering. Defaults to True.
        region_col (str): Column representing regions or categories. Defaults to "region".
        datetime_col (str): Column containing datetime values. Defaults to "datetime_beginning_ept".

    Raises:
        ValueError: Region feature name is not in the dataset.
        ValueError: The feature to plot is not in the dataset.
    """
    if region_filter and region_col not in data.columns:
        raise ValueError(f"'{region_col}' column must exist in the data.")
    if feature1 not in data.columns or feature2 not in data.columns:
        raise ValueError(
            f"'{feature1}' and '{feature2}' columns must exist in the data."
        )
    if "season" not in data.columns:
        data = add_season_column(data)

    df = data.copy()
    df[datetime_col] = pd.to_datetime(df[datetime_col])
    region_dropdown = (
        widgets.Dropdown(
            options=["All"] + sorted(df[region_col].unique().tolist()),
            value="All",
            description="Region:",
        )
        if region_filter
        else None
    )
    start_date_input = widgets.Text(
        value=df[datetime_col].min().strftime("%Y-%m-%d"),
        description="Start Date:",
        placeholder="YYYY-MM-DD",
        layout=widgets.Layout(width="300px"),
    )
    end_date_input = widgets.Text(
        value=df[datetime_col].max().strftime("%Y-%m-%d"),
        description="End Date:",
        placeholder="YYYY-MM-DD",
        layout=widgets.Layout(width="300px"),
    )
    season_toggle = widgets.Checkbox(
        value=seasonal_bands, description="Add Seasonal Bands"
    )

    output_plot = widgets.Output()
    fig = go.Figure()

    def update_timeseries(change=None):
        filtered_data = df.copy()
        if region_filter and region_dropdown and region_dropdown.value != "All":
            filtered_data = filtered_data[
                filtered_data[region_col] == region_dropdown.value
            ]
        try:
            start_date = pd.to_datetime(start_date_input.value)
            end_date = pd.to_datetime(end_date_input.value)
        except ValueError:
            with output_plot:
                output_plot.clear_output(wait=True)
                print("Invalid date format. Please use YYYY-MM-DD.")
            return
        filtered_data = filtered_data[
            (filtered_data[datetime_col] >= start_date)
            & (filtered_data[datetime_col] <= end_date)
        ]
        if not filtered_data.empty:
            fig.data = []
            fig.add_trace(
                go.Scatter(
                    x=filtered_data[datetime_col],
                    y=filtered_data[feature1],
                    mode="lines",
                    name=feature1.replace("_", " ").title(),
                    yaxis="y",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=filtered_data[datetime_col],
                    y=filtered_data[feature2],
                    mode="lines",
                    name=feature2.replace("_", " ").title(),
                    yaxis="y2",
                )
            )

            _title = f"{feature1.replace('_', ' ').title()} and {feature2.replace('_', ' ').title()} Time Series "
            if region_filter and region_dropdown.value != "All":
                _title += f"({region_dropdown.value})"

            fig.update_layout(
                title=_title,
                xaxis_title="Datetime",
                yaxis=dict(
                    title=f"{feature1.replace("_", " ").title()} {feature1_units}",
                    showgrid=True,
                ),
                yaxis2=dict(
                    title=f"{feature2.replace("_", " ").title()} {feature2_units}",
                    overlaying="y",
                    side="right",
                    showgrid=False,
                ),
                xaxis=dict(range=[start_date, end_date]),
                legend=dict(title="Features", x=1.05, y=1),
                template="plotly_white",
                width=1200,
                height=600,
            )
            if season_toggle.value:
                add_summer_winter_bands_to_plot(fig, filtered_data, datetime_col)
            else:
                fig.layout.shapes = []
        else:
            with output_plot:
                output_plot.clear_output(wait=True)
                print("No data available with the selected filters.")
            return
        with output_plot:
            output_plot.clear_output(wait=True)
            pio.show(fig, renderer="notebook")

    if region_filter and region_dropdown:
        region_dropdown.observe(update_timeseries, names="value")
    start_date_input.observe(update_timeseries, names="value")
    end_date_input.observe(update_timeseries, names="value")
    season_toggle.observe(update_timeseries, names="value")
    update_timeseries()
    date_range_inputs = widgets.HBox([start_date_input, end_date_input])
    controls = (
        widgets.HBox([region_dropdown, season_toggle, date_range_inputs])
        if region_filter and region_dropdown
        else widgets.HBox([season_toggle, date_range_inputs])
    )
    display(controls, output_plot)


#######################################################################################################################################
### Modelling ###
#######################################################################################################################################


def emergency_trigger_set_up(df):
    """Prepares the dataset for emergency trigger prediction by engineering features, applying one-hot encoding for categorical variables, and setting up temporal structure.

    Args:
        df (pd.DataFrame): The input dataset. Must include the following columns:
            - datetime_beginning_ept: Timestamp column (datetime64).
            - near_emergency: Binary indicator (0/1).
            - capacity_margin, lmp_volatility, region_stress_ratio: Relevant features.
            - region: Categorical variable representing geographic regions.

    Returns:
        pd.DataFrame: Preprocessed dataset for time-series modelling with engineered features.
    """
    # Grouping by datetime only (no region grouping)
    et_setup_df = (
        df.groupby(["datetime_beginning_ept"]).mean(numeric_only=True).reset_index()
    ).copy()

    et_setup_df = et_setup_df.sort_values(by=["datetime_beginning_ept"])

    # add temporal features
    et_setup_df["hour"] = et_setup_df["datetime_beginning_ept"].dt.hour
    et_setup_df["day_of_week"] = et_setup_df[
        "datetime_beginning_ept"
    ].dt.dayofweek  # 0=Monday, 6=Sunday
    et_setup_df["month"] = et_setup_df["datetime_beginning_ept"].dt.month
    et_setup_df["is_weekend"] = (et_setup_df["day_of_week"] >= 5).astype(
        int
    )  # Weekend indicator
    et_setup_df["season"] = et_setup_df["month"] % 12 // 3 + 1

    # add lagged features (hourly features)
    lags = [1, 3, 6, 24]
    for lag in lags:
        et_setup_df[f"near_emergency_lag{lag}"] = et_setup_df["near_emergency"].shift(
            lag
        )
        et_setup_df[f"capacity_margin_lag{lag}"] = et_setup_df["capacity_margin"].shift(
            lag
        )
        et_setup_df[f"lmp_volatility_lag{lag}"] = et_setup_df["lmp_volatility"].shift(
            lag
        )

    # region stress ratio is daily granular so 24 minimum lag is necessary to prevent data leakage.
    lags = [24, 48]
    for lag in lags:
        et_setup_df[f"region_stress_ratio_lag{lag}"] = et_setup_df[
            "region_stress_ratio"
        ].shift(lag)

    # add rolling averages features
    rolling_windows = [3, 6, 24]
    for window in rolling_windows:
        et_setup_df[f"lmp_volatility_roll{window}"] = (
            et_setup_df["lmp_volatility"].rolling(window, min_periods=1).mean()
        )

    rolling_windows = [24, 48, 168]
    for window in rolling_windows:
        et_setup_df[f"region_stress_ratio_roll{window}"] = (
            et_setup_df["region_stress_ratio"].rolling(window, min_periods=1).mean()
        )

    et_setup_df = et_setup_df.dropna()
    selected_features = [
        "emergency_triggered",
        "datetime_beginning_ept",
        "hour",
        "day_of_week",
        "season",
        "near_emergency_lag1",
        "capacity_margin_lag24",
        "lmp_volatility_lag6",
        "region_stress_ratio_lag24",
        "region_stress_ratio_lag48",
        "lmp_volatility_roll24",
        "region_stress_ratio_roll48",
    ]
    et_setup_df = et_setup_df[selected_features]

    et_setup_df.set_index("datetime_beginning_ept", inplace=True)

    return et_setup_df


def lmp_volatility_set_up(df):
    """Prepares the dataset for LMP volatility prediction by engineering features, adding lagged and rolling features, and creating interaction terms.

    Args:
        df (pd.DataFrame): The input dataset. Includes: datetime_beginning_ept, region, pnode_id, lmp_abs_delta, near_emergency, capacity_margin, lmp_volatility, region_stress_ratio, outage_intensity.

    Returns:
        pd.DataFrame: Preprocessed dataset for time-series modelling with engineered features.
    """

    lmp_vol_setup_df = df.copy()
    lmp_vol_setup_df = lmp_vol_setup_df.sort_values(
        by=["datetime_beginning_ept", "region", "pnode_id"]
    )

    # add temporal features
    lmp_vol_setup_df["hour"] = lmp_vol_setup_df["datetime_beginning_ept"].dt.hour
    lmp_vol_setup_df["day_of_week"] = lmp_vol_setup_df[
        "datetime_beginning_ept"
    ].dt.dayofweek
    lmp_vol_setup_df["month"] = lmp_vol_setup_df["datetime_beginning_ept"].dt.month
    lmp_vol_setup_df["is_weekend"] = (lmp_vol_setup_df["day_of_week"] >= 5).astype(int)
    lmp_vol_setup_df["season"] = lmp_vol_setup_df["month"] % 12 // 3 + 1

    lags = [1, 3, 6, 24]
    for lag in lags:
        lmp_vol_setup_df[f"lmp_abs_delta_lag{lag}"] = lmp_vol_setup_df.groupby(
            ["region"]
        )["lmp_abs_delta"].shift(lag)
        lmp_vol_setup_df[f"near_emergency_lag{lag}"] = lmp_vol_setup_df.groupby(
            ["region"]
        )["near_emergency"].shift(lag)
        lmp_vol_setup_df[f"capacity_margin_lag{lag}"] = lmp_vol_setup_df.groupby(
            ["region"]
        )["capacity_margin"].shift(lag)
        lmp_vol_setup_df[f"lmp_volatility_lag{lag}"] = lmp_vol_setup_df.groupby(
            ["region"]
        )["lmp_volatility"].shift(lag)

    rolling_windows_hourly = [3, 6, 24]
    for window in rolling_windows_hourly:
        lmp_vol_setup_df[f"lmp_volatility_roll{window}"] = (
            lmp_vol_setup_df.groupby(["region"])["lmp_volatility"]
            .rolling(window, min_periods=1)
            .mean()
            .reset_index(0, drop=True)
        )
        lmp_vol_setup_df[f"capacity_margin_roll{window}"] = (
            lmp_vol_setup_df.groupby(["region"])["capacity_margin"]
            .rolling(window, min_periods=1)
            .mean()
            .reset_index(0, drop=True)
        )

    rolling_windows = [24, 48, 168]
    for window in rolling_windows:
        lmp_vol_setup_df[f"region_stress_ratio_roll{window}"] = (
            lmp_vol_setup_df.groupby(["region"])["region_stress_ratio"]
            .rolling(window, min_periods=1)
            .mean()
            .reset_index(0, drop=True)
        )
        lmp_vol_setup_df[f"outage_intensity_roll{window}"] = (
            lmp_vol_setup_df.groupby(["region"])["outage_intensity"]
            .rolling(window, min_periods=1)
            .mean()
            .reset_index(0, drop=True)
        )

    lmp_vol_setup_df["region_stress_x_capacity_margin"] = (
        lmp_vol_setup_df["region_stress_ratio_roll24"]
        * lmp_vol_setup_df["capacity_margin_roll24"]
    )
    lmp_vol_setup_df["outage_intensity_x_capacity_margin"] = (
        lmp_vol_setup_df["outage_intensity_roll24"]
        * lmp_vol_setup_df["capacity_margin_roll24"]
    )
    lmp_vol_setup_df["outage_intensity_x_region_stress"] = (
        lmp_vol_setup_df["outage_intensity_roll24"]
        * lmp_vol_setup_df["region_stress_ratio_roll24"]
    )
    lmp_vol_setup_df["near_emergency_x_capacity_margin"] = (
        lmp_vol_setup_df["near_emergency_lag1"]
        * lmp_vol_setup_df["capacity_margin_lag1"]
    )

    lmp_vol_setup_df = lmp_vol_setup_df.dropna()
    onehot_encoder = OneHotEncoder(drop="first", sparse_output=False)
    onehot_encoded = onehot_encoder.fit_transform(lmp_vol_setup_df[["region"]])
    region_columns = [f"region_cat{cat}" for cat in onehot_encoder.categories_[0][1:]]
    lmp_vol_setup_df[region_columns] = onehot_encoded
    lmp_vol_setup_df.drop(columns=["region"], inplace=True)

    selected_features = [
        "lmp_volatility",
        "datetime_beginning_ept",
        "pnode_id",
        "hour",
        "day_of_week",
        "season",
        "lmp_abs_delta_lag1",
        "lmp_abs_delta_lag24",
        "near_emergency_lag1",
        "capacity_margin_lag24",
        "lmp_volatility_lag6",
        "region_stress_ratio_roll24",
        "outage_intensity_roll24",
        "capacity_margin_roll24",
        "region_stress_x_capacity_margin",
        "outage_intensity_x_capacity_margin",
        "outage_intensity_x_region_stress",
        "near_emergency_x_capacity_margin",
    ] + region_columns
    lmp_vol_setup_df = lmp_vol_setup_df[selected_features]

    lmp_vol_setup_df.set_index("datetime_beginning_ept", inplace=True)
    lmp_vol_setup_df = lmp_vol_setup_df.iloc[
        24:
    ]  # remove 0 vol given it is the first 24 hour rolling std
    return lmp_vol_setup_df


def forced_outages_set_up(df):
    """
    Prepares the dataset for forced outage predictions by engineering features, adding lagged and rolling features, and creating interaction terms.

    Args:
        df (pd.DataFrame): The input dataset. Includes: datetime_beginning_ept, region,
                           forced_outage_mw, region_stress_ratio, capacity_margin, outage_intensity.

    Returns:
        pd.DataFrame: Preprocessed dataset for time-series modelling with engineered features.
    """
    forced_outages_setup_df = (
        df.groupby([df["datetime_beginning_ept"].dt.date, "region"])
        .mean(numeric_only=True)
        .reset_index()
    ).copy()
    forced_outages_setup_df["datetime_beginning_ept"] = pd.to_datetime(
        forced_outages_setup_df["datetime_beginning_ept"]
    )
    forced_outages_setup_df = forced_outages_setup_df.sort_values(
        by=["datetime_beginning_ept", "region"]
    )

    # Add temporal features
    forced_outages_setup_df["day_of_week"] = forced_outages_setup_df[
        "datetime_beginning_ept"
    ].dt.dayofweek
    forced_outages_setup_df["month"] = forced_outages_setup_df[
        "datetime_beginning_ept"
    ].dt.month
    forced_outages_setup_df["is_weekend"] = (
        forced_outages_setup_df["day_of_week"] >= 5
    ).astype(int)
    forced_outages_setup_df["season"] = forced_outages_setup_df["month"] % 12 // 3 + 1

    lags = [1, 2, 7]
    for lag in lags:
        forced_outages_setup_df[f"forced_outages_lag{lag}"] = (
            forced_outages_setup_df.groupby("region")["forced_outages_mw"].shift(lag)
        )
        forced_outages_setup_df[f"outage_intensity_lag{lag}"] = (
            forced_outages_setup_df.groupby("region")["outage_intensity"].shift(lag)
        )
        forced_outages_setup_df[f"region_stress_ratio_lag{lag}"] = (
            forced_outages_setup_df.groupby("region")["region_stress_ratio"].shift(lag)
        )

    # Add rolling averages
    rolling_windows = [
        2,
        7,
        14,
    ]  # in days now as forced outages is a daily granular feature
    for window in rolling_windows:
        forced_outages_setup_df[f"forced_outages_roll{window}"] = (
            forced_outages_setup_df.groupby("region")["forced_outages_mw"]
            .rolling(window, min_periods=1)
            .mean()
            .reset_index(0, drop=True)
        )
        forced_outages_setup_df[f"outage_intensity_roll{window}"] = (
            forced_outages_setup_df.groupby("region")["outage_intensity"]
            .rolling(window, min_periods=1)
            .mean()
            .reset_index(0, drop=True)
        )
        forced_outages_setup_df[f"region_stress_ratio_roll{window}"] = (
            forced_outages_setup_df.groupby("region")["region_stress_ratio"]
            .rolling(window, min_periods=1)
            .mean()
            .reset_index(0, drop=True)
        )
        forced_outages_setup_df[f"capacity_margin_roll{window}"] = (
            forced_outages_setup_df.groupby("region")["capacity_margin"]
            .rolling(window, min_periods=1)
            .mean()
            .reset_index(0, drop=True)
        )
    # Add interaction terms
    forced_outages_setup_df["outage_intensity_x_region_stress"] = (
        forced_outages_setup_df["outage_intensity_roll7"]
        * forced_outages_setup_df["region_stress_ratio_roll7"]
    )
    forced_outages_setup_df["capacity_margin_x_region_stress"] = (
        forced_outages_setup_df["capacity_margin_roll7"]
        * forced_outages_setup_df["region_stress_ratio_roll7"]
    )

    forced_outages_setup_df = forced_outages_setup_df.dropna()
    onehot_encoder = OneHotEncoder(drop="first", sparse_output=False)
    onehot_encoded = onehot_encoder.fit_transform(forced_outages_setup_df[["region"]])
    region_columns = [f"region_cat{cat}" for cat in onehot_encoder.categories_[0][1:]]
    forced_outages_setup_df[region_columns] = onehot_encoded
    forced_outages_setup_df.drop(columns=["region"], inplace=True)

    selected_features = [
        "forced_outages_mw",
        "datetime_beginning_ept",
        "month",
        "day_of_week",
        "is_weekend",
        "season",
        "forced_outages_lag1",
        "forced_outages_lag7",
        "outage_intensity_lag1",
        "region_stress_ratio_roll7",
        "outage_intensity_roll7",
        "forced_outages_roll7",
        "capacity_margin_roll7",
        "outage_intensity_x_region_stress",
        "capacity_margin_x_region_stress",
    ] + region_columns
    forced_outages_setup_df = forced_outages_setup_df[selected_features]
    forced_outages_setup_df.set_index("datetime_beginning_ept", inplace=True)

    return forced_outages_setup_df


def optimize_hyperparameters_classification(model_name, X, y):
    """Perform hyperparameter optimization for a given model.

    Args:
        model_name (str): Name of the model ("decision_tree", "random_forest", "lightgbm").
        X (pd.DataFrame): Feature data.
        y (pd.Series): Target variable.

    Returns:
        model: Best-tuned model.
    """
    if model_name not in ["decision_tree", "random_forest", "lightgbm"]:
        raise ValueError(
            "Models available are: 'decision_tree', 'random_forest', 'lightgbm'"
        )

    if model_name == "decision_tree":
        param_dist = {
            "max_depth": [None, 5, 10, 20],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 5],
        }
        model = RandomizedSearchCV(
            DecisionTreeClassifier(random_state=42),
            param_dist,
            n_iter=10,
            scoring="f1",
            cv=3,
            random_state=42,
        )

    elif model_name == "random_forest":
        param_dist = {
            "n_estimators": [50, 100, 200],
            "max_depth": [None, 10, 20, 30],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
        }
        model = RandomizedSearchCV(
            RandomForestClassifier(random_state=42, n_jobs=-1),
            param_dist,
            n_iter=20,
            scoring="f1",
            cv=3,
            random_state=42,
        )

    elif model_name == "lightgbm":
        param_dist = {
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "num_leaves": [15, 31, 63],
            "max_depth": [-1, 10, 20],
        }
        model = RandomizedSearchCV(
            LGBMClassifier(random_state=42),
            param_dist,
            n_iter=20,
            scoring="f1",
            cv=3,
            random_state=42,
        )

    model.fit(X, y)
    print(f"Best Parameters for {model_name}:", model.best_params_)
    return model.best_estimator_


def optimize_hyperparameters_regression(model_name, X, y):
    """Perform hyperparameter optimization for a given regression model.

    Args:
        model_name (str): Name of the model ("decision_tree", "random_forest", "lightgbm").
        X (pd.DataFrame): Feature data.
        y (pd.Series): Target variable.

    Returns:
        model: Best-tuned model.
    """
    if model_name not in ["decision_tree", "random_forest", "lightgbm"]:
        raise ValueError(
            "Models available are: 'decision_tree', 'random_forest', 'lightgbm'"
        )

    if model_name == "decision_tree":
        param_dist = {
            "max_depth": [None, 5, 10, 20],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 5],
        }
        model = RandomizedSearchCV(
            DecisionTreeRegressor(random_state=42),
            param_dist,
            n_iter=10,
            scoring="neg_mean_squared_error",
            cv=3,
            random_state=42,
        )

    elif model_name == "random_forest":
        param_dist = {
            "n_estimators": [50, 100, 200],
            "max_depth": [None, 10, 20, 30],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
        }
        model = RandomizedSearchCV(
            RandomForestRegressor(random_state=42, n_jobs=-1),
            param_dist,
            n_iter=20,
            scoring="neg_mean_squared_error",
            cv=3,
            random_state=42,
        )

    elif model_name == "lightgbm":
        param_dist = {
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "num_leaves": [15, 31, 63],
            "max_depth": [-1, 10, 20],
        }
        model = RandomizedSearchCV(
            LGBMRegressor(random_state=42),
            param_dist,
            n_iter=20,
            scoring="neg_mean_squared_error",
            cv=3,
            random_state=42,
        )

    model.fit(X, y)
    print(f"Best Parameters for {model_name}:", model.best_params_)
    return model.best_estimator_


def walk_forward_validation_classification(
    data,
    target_column,
    model_save_path,
    models_to_use=None,
    n_splits=5,
    save_best_model=True,
    train_test_split_ratio=0.8,
    gap=24,
):
    """Perform walk-forward validation with hyperparameter optimization for classification models, and saves the best models and evaluation metrics.

    Args:
        data (pd.DataFrame): Time-series data with datetime index.
        target_column (str): Name of the target column (Ex. "emergency_triggered").
        model_save_path (str): Path to save models.
        models_to_use (list of str): List of model names to run, specifically ["decision_tree", "random_forest", "lightgbm"]. Defaults to None.
        n_splits (int, optional): Number of splits for TimeSeriesSplit. Default to 5.
        save_best_model (bool, optional): Whether to save the best model for each type. Defaults to True.
        train_test_split_ratio (float, optional): The percentage of data allocated as the train set. Defaults to 0.8.
        gap (int, optional): Number of unique time points (in hours) to skip between training and testing sets. Defaults to 24 hours.s
    """

    data = data.sort_index()
    unique_times = data.index.unique()
    split_idx = int(len(unique_times) * train_test_split_ratio)
    train_times = unique_times[:split_idx]
    test_times = unique_times[split_idx:]

    train_data = data.loc[data.index.isin(train_times)]
    test_data = data.loc[data.index.isin(test_times)]

    X_train = train_data.drop(columns=[target_column])
    y_train = train_data[target_column]
    X_test = test_data.drop(columns=[target_column])
    y_test = test_data[target_column]

    model_names = models_to_use
    for model_name in model_names:
        best_f1_score = -1
        best_model = None
        splitter = custom_time_series_split_no_overlap(train_data, n_splits, gap=gap)
        for fold, (train_idx, test_idx) in enumerate(splitter, start=1):
            print(f"\n____ Fold {fold} - {model_name} ____")
            X_fold_train, X_fold_test = X_train.loc[train_idx], X_train.loc[test_idx]
            y_fold_train, y_fold_test = y_train.loc[train_idx], y_train.loc[test_idx]
            candidate_model = optimize_hyperparameters_classification(
                model_name, X_fold_train, y_fold_train
            )
            y_pred = candidate_model.predict(X_fold_test)
            report = classification_report(y_fold_test, y_pred, output_dict=True)
            f1_score = report["weighted avg"]["f1-score"]
            if f1_score > best_f1_score:
                best_f1_score = f1_score
                best_model = candidate_model
            print(f"Fold {fold} - {model_name}: F1-Score = {f1_score:.2f}")

        if best_model:  # eval best model on hold-out test set
            y_test_pred = best_model.predict(X_test)
            test_report = classification_report(y_test, y_test_pred, output_dict=True)
            test_f1_score = test_report["weighted avg"]["f1-score"]
            print(f"Hold-out Test - {model_name}: F1-Score = {test_f1_score:.2f}")

        if save_best_model and best_model:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_name = f"{model_name}_{target_column}_{timestamp}.pkl"
            save_path = os.path.join(model_save_path, target_column, file_name)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            joblib.dump(best_model, save_path)

            metadata = {
                "best_f1_score": best_f1_score,
                "test_f1_score": test_f1_score,
                "save_path": save_path,
                "training_time": timestamp,
                "features": X_train.columns.tolist(),
                "date_range": {
                    "train_start": train_times.min().strftime("%Y-%m-%d"),
                    "train_end": train_times.max().strftime("%Y-%m-%d"),
                    "test_start": test_times.min().strftime("%Y-%m-%d"),
                    "test_end": test_times.max().strftime("%Y-%m-%d"),
                },
            }
            metadata_path = os.path.join(
                model_save_path,
                target_column,
                f"{model_name}_metadata_{timestamp}.json",
            )
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=4)
            print(f"Metadata for {model_name} saved at: {metadata_path}")


def walk_forward_validation_regression(
    data,
    target_column,
    model_save_path,
    models_to_use=None,
    n_splits=5,
    save_best_model=True,
    train_test_split_ratio=0.8,
    gap=24,
):
    """
    Perform walk-forward validation with hyperparameter optimization for regression models, and saves the best models and evaluation metrics.

    Args:
        data (pd.DataFrame): Time-series data with datetime index.
        target_column (str): Name of the target column (e.g., "lmp_volatility").
        model_save_path (str): Path to save models.
        models_to_use (list of str): List of model names to run (e.g., ["decision_tree", "random_forest", "lightgbm"]).
        n_splits (int, optional): Number of splits for TimeSeriesSplit. Default to 5.
        save_best_model (bool, optional): Whether to save the best model for each type. Defaults to True.
        train_test_split_ratio (float, optional): The percentage of data allocated as the train set. Defaults to 0.8.
        gap (int, optional): Number of unique time points (in hours) to skip between training and testing sets. Defaults to 24 hours.
    """
    # Ensure data is sorted
    data = data.sort_index()
    unique_times = data.index.unique()
    split_idx = int(len(unique_times) * train_test_split_ratio)
    train_times = unique_times[:split_idx]
    test_times = unique_times[split_idx:]

    train_data = data.loc[data.index.isin(train_times)]
    test_data = data.loc[data.index.isin(test_times)]

    X_train = train_data.drop(columns=[target_column])
    y_train = train_data[target_column]
    X_test = test_data.drop(columns=[target_column])
    y_test = test_data[target_column]

    model_names = models_to_use or ["decision_tree", "random_forest", "lightgbm"]

    for model_name in model_names:
        best_model = None
        best_rmse = float("inf")
        splitter = custom_time_series_split_no_overlap(train_data, n_splits, gap=gap)
        for fold, (train_idx, test_idx) in enumerate(splitter, start=1):
            print(f"\n____ Fold {fold} - {model_name} ____")
            X_fold_train, X_fold_test = X_train.loc[train_idx], X_train.loc[test_idx]
            y_fold_train, y_fold_test = y_train.loc[train_idx], y_train.loc[test_idx]
            candidate_model = optimize_hyperparameters_regression(
                model_name, X_fold_train, y_fold_train
            )
            y_pred = candidate_model.predict(X_fold_test)
            rmse = np.sqrt(mean_squared_error(y_fold_test, y_pred))
            print(f"Fold {fold} - RMSE: {rmse:.2f}")
            if rmse < best_rmse:
                best_rmse = rmse
                best_model = candidate_model
        if best_model:
            y_test_pred = best_model.predict(X_test)
            test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
            print(f"Hold-out Test - {model_name}: RMSE = {test_rmse:.2f}")
        if save_best_model and best_model:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_name = f"{model_name}_{target_column}_{timestamp}.pkl"
            save_path = os.path.join(model_save_path, target_column, file_name)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            joblib.dump(best_model, save_path)
            print(f"Best {model_name} saved at: {save_path}")
            metadata = {
                "best_rmse": best_rmse,
                "test_rmse": test_rmse,
                "save_path": save_path,
                "training_time": timestamp,
                "features": X_train.columns.tolist(),
                "date_range": {
                    "train_start": train_times.min().strftime("%Y-%m-%d"),
                    "train_end": train_times.max().strftime("%Y-%m-%d"),
                    "test_start": test_times.min().strftime("%Y-%m-%d"),
                    "test_end": test_times.max().strftime("%Y-%m-%d"),
                },
            }
            metadata_path = os.path.join(
                model_save_path,
                target_column,
                f"{model_name}_metadata_{timestamp}.json",
            )
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=4)
            print(f"Metadata for {model_name} saved at: {metadata_path}")


def custom_time_series_split_no_overlap(data, n_splits, gap=24):
    """
    Custom Time Series Split that prevents overlap of grouped data by unique time points.

    Args:
        data (pd.DataFrame): Input DataFrame.
        n_splits (int): Number of splits.
        gap (int, optional): Number of unique time points (in hours) to skip between training and testing sets. Defaults to 24 hours.

    Yields:
        train_indices (np.array): Indices for training data.
        test_indices (np.array): Indices for testing data.
    """
    data = data.sort_index()
    unique_time_points = data.index.unique()
    n_time_points = len(unique_time_points)
    split_size = n_time_points // (n_splits + 1)

    for i in range(n_splits):
        train_end_idx = split_size * (i + 1)
        test_start_idx = train_end_idx + gap
        test_end_idx = test_start_idx + split_size
        if test_end_idx > n_time_points:
            test_end_idx = n_time_points
        train_time_points = unique_time_points[:train_end_idx]
        test_time_points = unique_time_points[test_start_idx:test_end_idx]
        train_indices = data.loc[train_time_points].index
        test_indices = data.loc[test_time_points].index
        yield train_indices, test_indices
