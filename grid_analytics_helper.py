import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import logging
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from lightgbm import LGBMClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# Merge Data #
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


# Feature Generation #
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


########################################################################################################################################################################################################
# Data Handle #
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
            raise ValueError(
                "The inputted dataframe does not have regional data nor PJM data."
            )
    else:
        raise ValueError(
            "'datetime_beginning_ept','forced_outage_pct_by_region', or 'forced_outage_pct_PJM' are not in the dataframe."
        )


def handle_raw_outage_data(df):
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


def pnode_lmp_outliers(
    df,
    *args,
    method=outlier_df_iqr,
    region_filter=None,
    region_col="region",
    top_n_nodes=10,
    col_names=["pnode_name", "region"],
):
    """Produces the top_n_nodes based on a outlier detection method user selects.

    Args:
        df (pd.DataFrame): Dataframe containings data for outlier detection.
        method (callable, optional): Outlier detection method, which is either outlier_df_iqr or outlier_df_by_dollar_amt. Defaults to outlier_df_iqr.
        region_filter (_type_, optional): List of region(s) to filter for. Defaults to None.
        region_col (str, optional): Column representing regions or categories. Defaults to "region".
        top_n_nodes (int, optional): Top n nodes user wants to see. Defaults to 10.
        col_names (list, optional): Features to display in the final results. Defaults to ["pnode_name", "region"].

    Raises:
        ValueError: Region filter provided yielded no data.
        ValueError: Outlier detection method provided does not exist.
    """
    filtered_df = df.copy()
    if region_filter:
        if isinstance(region_filter, str):
            region_filter = re.split(r"[,\s;]+", region_filter)
        filtered_df = filtered_df[
            filtered_df[region_col].isin(region_filter)
        ].reset_index(drop=True)

        if filtered_df.empty:
            raise ValueError(
                f"No data found for the specified regions: {region_filter}"
            )
    if method == outlier_df_iqr:
        multiplier = args[0] if args else 1.5
        outlier_df = method(df=filtered_df, col_name="lmp_delta", multiplier=multiplier)
    elif method == outlier_df_by_dollar_amt:
        dollar_amt = args[0] if args else 50
        outlier_df = method(df=filtered_df, col_name="lmp_delta", dollar_amt=dollar_amt)
    else:
        raise ValueError(
            "Invalid method. Use '_outlier_df_iqr' or '_outlier_df_by_dollar_amt'."
        )
    print(f"Top {top_n_nodes} Pricing Nodes with the Most Outliers:")
    print(outlier_df[col_names].value_counts().head(top_n_nodes))


# Plot #
def plot_forced_outage_lmp_vol_ts(
    data,
    floor_outlier_threshold=None,
    ceil_outlier_threshold=None,
    season_filter=None,
    title=None,
    **plot_args,
):
    """LMP volatility has hourly granularity while Forced Outages has daily granularity data, so Capacity Margin is averaged into a daily granular data, then a timeseries used to display the relationship between the two features.

    Args:
        data (pd.DataFrame): Dataframe containings data for visuals.
        floor_outlier_threshold (float, optional): A floor value set on Forced Outages values to display.
        ceil_outlier_threshold (float, optional): A ceiling value on Forced Outages values to display.
        region_filter (list of str, optional): List of region(s) to filter for. Defaults to None.
        region_col (str, optional): Column representing regions or categories. Defaults to "region".
        season_filter (str, optional): To specify which season to view. The options are "Winter" or "Summer" and is case insensitive, other specifications will result in the Spring and Fall data displayed. Defaults to None.
        title (str, optional): Title of chart. Defaults to None.
    """
    filtered_data = data[
        ["datetime_beginning_ept", "lmp_volatility", "forced_outages_mw"]
    ].copy()
    filtered_data["date_only"] = filtered_data["datetime_beginning_ept"].dt.date
    filtered_data = (
        filtered_data.groupby("date_only")[["lmp_volatility", "forced_outages_mw"]]
        .mean()
        .reset_index()
    )
    filtered_data.sort_values(by="date_only", inplace=True)
    filtered_data["month"] = pd.to_datetime(filtered_data["date_only"]).dt.month
    month_to_season = {
        12: "Winter",
        1: "Winter",
        2: "Winter",
        6: "Summer",
        7: "Summer",
        8: "Summer",
    }
    filtered_data["season"] = (
        filtered_data["month"].map(month_to_season).fillna("Spring & Fall")
    )
    if season_filter is not None:
        if season_filter.lower() == "winter":
            filtered_data = filtered_data[filtered_data["season"] == "Winter"]
            season_title = "(Season: Winter)"
        elif season_filter.lower() == "summer":
            filtered_data = filtered_data[filtered_data["season"] == "Summer"]
            season_title = "(Season: Summer)"
        else:
            filtered_data = filtered_data[filtered_data["season"] == "Spring & Fall"]
            season_title = "(Season: Spring & Fall)"
    else:
        season_title = None
    # outlier threshold on forced outage feature:
    if floor_outlier_threshold is not None:
        filtered_data = filtered_data[
            filtered_data["forced_outages_mw"] > floor_outlier_threshold
        ]
    if ceil_outlier_threshold is not None:
        filtered_data = filtered_data[
            filtered_data["forced_outages_mw"] < ceil_outlier_threshold
        ]
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(
        filtered_data["date_only"],
        filtered_data["forced_outages_mw"],
        color="blue",
        label="Forced Outage (MW)",
        alpha=0.9,
        linestyle="dotted",
        **plot_args,
    )
    ax1.set_xlabel("Datetime")
    ax1.set_ylabel("Forced Outage (MW)", color="blue")
    ax1.tick_params(axis="y", labelcolor="blue")
    ax2 = ax1.twinx()
    ax2.plot(
        filtered_data["date_only"],
        filtered_data["lmp_volatility"],
        color="red",
        label="LMP Volatility ($/MWh)",
        alpha=0.75,
        **plot_args,
    )
    ax2.set_ylabel("LMP Volatility ($/MWh)", color="red")
    ax2.tick_params(axis="y", labelcolor="red")
    if title is None:
        plt.title(
            f"Forced Outages vs LMP Volatility - Daily Aggregated {f'{season_title}' if season_title else ''}",
            fontsize=14,
        )
    else:
        plt.title(title)
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_capacity_margin_v_forced_outages(
    data,
    high_forced_outage_threshold=5000,
    floor_outlier_threshold=None,
    ceil_outlier_threshold=None,
    season_filter=None,
    title=None,
    **plot_args,
):
    """Capacity Margin has hourly granularity while Forced Outages has daily granularity data, so Capacity Margin is averaged into a daily granular data, then a scatterplot used to display the relationship between the two features.

    Args:
        data (pd.DataFrame): Dataframe containings data for visuals.
        high_outage_threshold (int, optional): Threshold which determines in a data point is a consider a "high" outage intenstiy. Defaults to 15.
        floor_outlier_threshold (float, optional): A floor value set on Forced Outages values to display.
        ceil_outlier_threshold (float, optional): A ceiling value on Forced Outages values to display.
        season_filter (str, optional): To specify which season to view. The options are "Winter" or "Summer" and is case insensitive, other specifications will result in the Spring and Fall data displayed. Defaults to None.
        title (str, optional): Title of chart. Defaults to None.
    """
    filtered_data = data[
        ["datetime_beginning_ept", "capacity_margin", "forced_outages_mw"]
    ].copy()
    filtered_data["date_only"] = filtered_data["datetime_beginning_ept"].dt.date
    filtered_data = (
        filtered_data.groupby("date_only")[["capacity_margin", "forced_outages_mw"]]
        .mean()
        .reset_index()
    )
    filtered_data.sort_values(by="date_only", inplace=True)
    filtered_data["month"] = pd.to_datetime(filtered_data["date_only"]).dt.month
    month_to_season = {
        12: "Winter",
        1: "Winter",
        2: "Winter",
        6: "Summer",
        7: "Summer",
        8: "Summer",
    }
    filtered_data["season"] = (
        filtered_data["month"].map(month_to_season).fillna("Spring & Fall")
    )
    if season_filter is not None:
        if season_filter.lower() == "winter":
            filtered_data = filtered_data[filtered_data["season"] == "Winter"]
            season_title = "(Season: Winter)"
        elif season_filter.lower() == "summer":
            filtered_data = filtered_data[filtered_data["season"] == "Summer"]
            season_title = "(Season: Summer)"
        else:
            filtered_data = filtered_data[filtered_data["season"] == "Spring & Fall"]
            season_title = "(Season: Spring & Fall)"
    else:
        season_title = None
    # outlier threshold on forced outage feature:
    if floor_outlier_threshold is not None:
        filtered_data = filtered_data[
            filtered_data["forced_outages_mw"] > floor_outlier_threshold
        ]
    if ceil_outlier_threshold is not None:
        filtered_data = filtered_data[
            filtered_data["forced_outages_mw"] < ceil_outlier_threshold
        ]
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        x="capacity_margin",
        y="forced_outages_mw",
        data=filtered_data,
        hue=(
            filtered_data["forced_outages_mw"] > high_forced_outage_threshold
        ),  # Highlight high outage intensity
        palette={True: "red", False: "blue"},
        legend="brief",
        alpha=0.6,
        s=50,
        **plot_args,
    )
    x = filtered_data["capacity_margin"].values.reshape(-1, 1)
    y = filtered_data["forced_outages_mw"].values
    reg_model = LinearRegression().fit(x, y)
    plt.plot(
        filtered_data["capacity_margin"],
        reg_model.predict(x),
        color="green",
        label=f"Regression Line (R² = {reg_model.score(x, y):.2f})",
    )
    if title is None:
        plt.title(
            f"Forced Outages vs Capacity Margin - Daily Aggregated {f'{season_title}' if season_title else ''}",
            fontsize=14,
        )
    else:
        plt.title(title)
    plt.xlabel("Capacity Margin (%)", fontsize=12)
    plt.ylabel("Forced Outages (MW)", fontsize=12)
    plt.legend(
        title=f"High Forced Outages (>{high_forced_outage_threshold}MW)",
        loc="upper left",
    )
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_forced_outages_v_lmp_vol_by_region(
    data,
    high_forced_outage_threshold=5000,
    floor_outlier_threshold=None,
    ceil_outlier_threshold=None,
    region_filter=None,
    region_col="region",
    season_filter=None,
    title=None,
    **plot_args,
):
    """LMP volatility has hourly granularity while Forced Outages has daily granularity data, so Capacity Margin is averaged into a daily granular data, then a scatterplot used to display the relationship between the two features.

    Args:
        data (pd.DataFrame): Dataframe containings data for visuals.
        high_outage_threshold (int, optional): Threshold which determines in a data point is a consider a "high" outage intenstiy. Defaults to 15.
        floor_outlier_threshold (float, optional): A floor value set on Forced Outages values to display.
        ceil_outlier_threshold (float, optional): A ceiling value on Forced Outages values to display.
        region_filter (list of str, optional): List of region(s) to filter for. Defaults to None.
        region_col (str, optional): Column representing regions or categories. Defaults to "region".
        season_filter (str, optional): To specify which season to view. The options are "Winter" or "Summer" and is case insensitive, other specifications will result in the Spring and Fall data displayed. Defaults to None.
        title (str, optional): Title of chart. Defaults to None.
    """
    filtered_data = data.copy()
    if region_filter:
        if isinstance(region_filter, str):
            region_filter = re.split(r"[,\s;]+", region_filter)
        filtered_data = filtered_data[
            filtered_data[region_col].isin(region_filter)
        ].reset_index(drop=True)

        if filtered_data.empty:
            raise ValueError(
                f"No data found for the specified regions: {region_filter}"
            )

    filtered_data = data[
        ["datetime_beginning_ept", "region", "lmp_volatility", "forced_outages_mw"]
    ].copy()
    filtered_data["date_only"] = filtered_data["datetime_beginning_ept"].dt.date
    filtered_data = (
        filtered_data.groupby(["date_only", "region"])[
            ["lmp_volatility", "forced_outages_mw"]
        ]
        .mean()
        .reset_index()
    )
    filtered_data.sort_values(by="date_only", inplace=True)
    filtered_data["month"] = pd.to_datetime(filtered_data["date_only"]).dt.month
    month_to_season = {
        12: "Winter",
        1: "Winter",
        2: "Winter",
        6: "Summer",
        7: "Summer",
        8: "Summer",
    }
    filtered_data["season"] = (
        filtered_data["month"].map(month_to_season).fillna("Spring & Fall")
    )
    if season_filter is not None:
        if season_filter.lower() == "winter":
            filtered_data = filtered_data[filtered_data["season"] == "Winter"]
            season_title = "(Season: Winter)"
        elif season_filter.lower() == "summer":
            filtered_data = filtered_data[filtered_data["season"] == "Summer"]
            season_title = "(Season: Summer)"
        else:
            filtered_data = filtered_data[filtered_data["season"] == "Spring & Fall"]
            season_title = "(Season: Spring & Fall)"
    else:
        season_title = None
    # outlier threshold on forced outage feature:
    if floor_outlier_threshold is not None:
        filtered_data = filtered_data[
            filtered_data["forced_outages_mw"] > floor_outlier_threshold
        ]
    if ceil_outlier_threshold is not None:
        filtered_data = filtered_data[
            filtered_data["forced_outages_mw"] < ceil_outlier_threshold
        ]
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        x="lmp_volatility",
        y="forced_outages_mw",
        data=filtered_data,
        hue=(filtered_data["forced_outages_mw"] > high_forced_outage_threshold),
        palette={True: "red", False: "blue"},
        legend="brief",
        alpha=0.6,
        s=50,
        **plot_args,
    )
    x = filtered_data["lmp_volatility"].values.reshape(-1, 1)
    y = filtered_data["forced_outages_mw"].values
    reg_model = LinearRegression().fit(x, y)
    plt.plot(
        filtered_data["lmp_volatility"],
        reg_model.predict(x),
        color="green",
        label=f"Regression Line (R² = {reg_model.score(x, y):.2f})",
    )
    if title is None:
        title = f"Forced Outages vs LMP Volatility - Daily Aggregated {f'{season_title}' if season_title else ''}"
        if region_filter:
            title += f" - Region: {",".join(region_filter)}"
    plt.title(
        title,
        fontsize=14,
    )
    plt.xlabel("lmp_volatility ($/MWh)", fontsize=12)
    plt.ylabel("Forced Outages (MW)", fontsize=12)
    plt.legend(
        title=f"High Forced Outages (>{high_forced_outage_threshold}MW)",
        loc="upper left",
    )
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_capacity_margin_v_outage_intensity(
    data,
    high_outage_threshold=15,
    floor_outlier_threshold=None,
    ceil_outlier_threshold=None,
    season_filter=None,
    **plot_args,
):
    """Capacity Margin has hourly granularity while Outage Intensity has daily granularity data, so Capacity Margin is averaged into a daily granular data, then a scatterplot used to display the relationship between the two features.

    Args:
        data (pd.DataFrame): Dataframe containings data for visuals.
        high_outage_threshold (int, optional): Threshold which determines in a data point is a consider a "high" outage intenstiy. Defaults to 15.
        floor_outlier_threshold (float, optional): A floor value set on Outage Intensity values to display.
        ceil_outlier_threshold (float, optional): A ceiling value on Outage Intensity values to display.
        season_filter (str, optional): To specify which season to view. The options are "Winter" or "Summer" and is case insensitive, other specifications will result in the Spring and Fall data displayed. Defaults to None.
    """
    filtered_data = data[
        ["datetime_beginning_ept", "capacity_margin", "outage_intensity"]
    ].copy()
    filtered_data["date_only"] = filtered_data["datetime_beginning_ept"].dt.date
    filtered_data = (
        filtered_data.groupby("date_only")[["capacity_margin", "outage_intensity"]]
        .mean()
        .reset_index()
    )
    filtered_data.sort_values(by="date_only", inplace=True)
    # filtered_data["high_outage"] = (
    #     filtered_data["outage_intensity"] > high_outage_threshold
    # )

    filtered_data["month"] = pd.to_datetime(filtered_data["date_only"]).dt.month
    month_to_season = {
        12: "Winter",
        1: "Winter",
        2: "Winter",
        6: "Summer",
        7: "Summer",
        8: "Summer",
    }
    filtered_data["season"] = (
        filtered_data["month"].map(month_to_season).fillna("Spring & Fall")
    )
    if season_filter is not None:
        if season_filter.lower() == "winter":
            filtered_data = filtered_data[filtered_data["season"] == "Winter"]
            season_title = "(Season: Winter)"
        elif season_filter.lower() == "summer":
            filtered_data = filtered_data[filtered_data["season"] == "Summer"]
            season_title = "(Season: Summer)"
        else:
            filtered_data = filtered_data[filtered_data["season"] == "Spring & Fall"]
            season_title = "(Season: Spring & Fall)"
    else:
        season_title = None

    if floor_outlier_threshold is not None:
        filtered_data = filtered_data[
            filtered_data["outage_intensity"] > floor_outlier_threshold
        ]
    if ceil_outlier_threshold is not None:
        filtered_data = filtered_data[
            filtered_data["outage_intensity"] < ceil_outlier_threshold
        ]

    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        x="capacity_margin",
        y="outage_intensity",
        data=filtered_data,
        hue=(
            filtered_data["outage_intensity"] > high_outage_threshold
        ),  # Highlight high outage intensity
        palette={True: "red", False: "blue"},
        legend="brief",
        alpha=0.6,
        s=50,
        **plot_args,
    )
    x = filtered_data["capacity_margin"].values.reshape(-1, 1)
    y = filtered_data["outage_intensity"].values
    reg_model = LinearRegression().fit(x, y)
    plt.plot(
        filtered_data["capacity_margin"],
        reg_model.predict(x),
        color="green",
        label=f"Regression Line (R² = {reg_model.score(x, y):.2f})",
    )
    plt.title(
        f"Outage Intensity vs Capacity Margin - Daily Aggregated {f'{season_title}' if season_title else ''}",
        fontsize=14,
    )
    plt.xlabel("Capacity Margin (%)", fontsize=12)
    plt.ylabel("Outage Intensity (%)", fontsize=12)
    plt.legend(
        title=f"High Outage Intensity (>{high_outage_threshold}%)", loc="upper left"
    )
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_scatter_pnode_outliers(
    data,
    y_label,
    *args,
    col_name="lmp_delta",
    title=None,
    method=outlier_df_iqr,
    region_filter=None,
    region_col="region",
    seasonal_bands=True,
    **plot_args,
):
    """Creates a scatter plot of a time series.

    Args:
        data (pd.DataFrame): Dataframe containings data for outlier detection.
        y_label (str): Label of the y-axis.
        col_name (str, optional): Column name of the specific data for visuals. Defaults to "lmp_delta".
        title (str, optional): Title of chart. Defaults to None.
        method (callable, optional): Outlier detection method, which is either outlier_df_iqr or outlier_df_by_dollar_amt. Defaults to outlier_df_iqr.
        region_filter (list of str, optional): List of region(s) to filter for. Defaults to None.
        region_col (str, optional): Column representing regions or categories. Defaults to "region".
        seasonal_bands (bool, optional): Determine if vertical bands indicating winter and summers months will be added to the plot. Defaults to True.

    Raises:
        ValueError: Region filter provided yielded no data.
        ValueError: Outlier detection method provided does not exist.
    """
    filtered_data = data.copy()
    if region_filter:
        if isinstance(region_filter, str):
            region_filter = re.split(r"[,\s;]+", region_filter)
        filtered_data = filtered_data[
            filtered_data[region_col].isin(region_filter)
        ].reset_index(drop=True)

        if filtered_data.empty:
            raise ValueError(
                f"No data found for the specified regions: {region_filter}"
            )

    if method == outlier_df_iqr:
        multiplier = args[0] if args else 1.5
        outlier_df = method(df=filtered_data, col_name=col_name, multiplier=multiplier)
    elif method == outlier_df_by_dollar_amt:
        dollar_amt = args[0] if args else 50
        outlier_df = method(df=filtered_data, col_name=col_name, dollar_amt=dollar_amt)
    else:
        raise ValueError(
            "Invalid method. Use '_outlier_df_iqr' or '_outlier_df_by_dollar_amt'."
        )

    if method == outlier_df_by_dollar_amt:  # for LMP features
        if len(args) != 0:
            dollar_amt = args[0]
        else:
            dollar_amt = 50
        plt.scatter(
            outlier_df["datetime_beginning_ept"],
            outlier_df[col_name],
            color="red",
            label=f"Outliers > |${dollar_amt} /MWh|",
            alpha=0.6,
            **plot_args,
        )
    else:
        plt.scatter(
            outlier_df["datetime_beginning_ept"],
            outlier_df[col_name],
            color="red",
            label=f"Outliers based on IQR x {multiplier}",
            alpha=0.6,
            **plot_args,
        )
    if seasonal_bands:
        add_summer_winter_bands(outlier_df, "datetime_beginning_ept")
    if title is None:
        title = f"{col_name}. Over Time with Outliers"
        if region_filter:
            title += f" - Region: {",".join(region_filter)}"
    plt.title(title)
    plt.xlabel("Datetime")
    plt.ylabel(y_label)
    plt.legend()
    plt.grid()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_boxplot_with_outlier_filter_by_region(
    data,
    col_name,
    y_label,
    title=None,
    region_filter=None,
    region_col="region",
    outlier_threshold=None,
    cover_all_regions=False,
    **boxplot_args,
):
    """Plot a boxplot for a given column by region(s), with the ability to filter extreme outliers.

    Args:
        data (pd.DataFrame): Dataframe containings data for visuals.
        col_name (str): Column name of the specific data for visuals.
        y_label (str): Label of the y-axis.
        title (str, optional): Title of chart. Defaults to None.
        region_filter (list of str, optional): List of region(s) to filter for and MUST be provided if cover_all_regions is True. Defaults to None.
        region_col (str, optional): Column representing regions or categories. Defaults to "region".
        outlier_threshold (float, optional): If provided, exclude rows where abs(col_name) > outlier_threshold.. Defaults to None.
        cover_all_regions (bool, optional): Indicates if data presented is representative of the entire PJM RTO rather than individual regions. Defaults to False.

    Raises:
        ValueError: Region filter provided yielded no data.
        ValueError: Outlier threshold provided yielded no data.
    """
    filtered_data = data.copy()
    if region_filter:
        if isinstance(region_filter, str):
            region_filter = re.split(r"[,\s;]+", region_filter)
        filtered_data = filtered_data[
            filtered_data[region_col].isin(region_filter)
        ].reset_index(drop=True)

        if filtered_data.empty:
            raise ValueError(
                f"No data found for the specified regions: {region_filter}"
            )

    if outlier_threshold is not None:
        filtered_data = filtered_data[
            filtered_data[col_name].abs() <= outlier_threshold
        ].reset_index(drop=True)

        if filtered_data.empty:
            raise ValueError(
                f"No data left after applying the outlier threshold of {outlier_threshold}"
            )

    plt.figure(figsize=(10, 6))
    if cover_all_regions:
        filtered_data[region_col] = filtered_data[region_col].replace(
            {region_filter[0]: "All of PJM"}
        )
    sns.boxplot(
        x=region_col,
        y=col_name,
        data=filtered_data,
        hue=region_col,
        palette="Set2",
        showfliers=True,
        width=0.9,
        **boxplot_args,
    )
    if title:
        plt.title(title)
    else:
        plt.title(
            f"{col_name.replace("_", " ").upper()} by Region {f'(Outliers Restricted | Threshold = {outlier_threshold})' if outlier_threshold else ''}"
        )
    if region_filter is None and not cover_all_regions:
        plt.xlabel("Region")
    plt.ylabel(y_label)
    plt.grid(axis="y")
    plt.tight_layout()
    plt.show()


def plot_timeseries_plots_by_region(
    data,
    col_name,
    y_label,
    title=None,
    region_filter=None,
    region_col="region",
    remove_legend=False,
    seasonal_bands=True,
    **plot_args,
):
    """Creates a timeseries plot.

    Args:
        data (pd.DataFrame): Dataframe containings data for visuals.
        col_name (str): Column name of the specific data for visuals.
        y_label (str): Label of the y-axis.
        title (str, optional): Title of chart. Defaults to None.
        region_filter (list of str, optional): List of region(s) to filter for. Defaults to None.
        region_col (str, optional): Column representing regions or categories. Defaults to "region".
        remove_legend (bool, optional): Determines if a legend should be removed or not. Defaults to False.
        seasonal_bands (bool, optional): Determine if vertical bands indicating winter and summers months will be added to the plot. Defaults to True.

    Raises:
        ValueError: Region filter provided yielded no data.
    """
    filtered_data = data.copy()
    if region_filter:
        if isinstance(region_filter, str):
            region_filter = re.split(r"[,\s;]+", region_filter)
        filtered_data = filtered_data[
            filtered_data[region_col].isin(region_filter)
        ].reset_index(drop=True)

        if filtered_data.empty:
            raise ValueError(
                f"No data found for the specified regions: {region_filter}"
            )

    plt.figure(figsize=(12, 6))
    for region, group in filtered_data.groupby(region_col):
        group = group.sort_values("datetime_beginning_ept")
        group.set_index("datetime_beginning_ept")[col_name].plot(
            label=region, alpha=0.6, **plot_args
        )
    if seasonal_bands:
        add_summer_winter_bands(group, "datetime_beginning_ept")
    if title:
        plt.title(title)
    else:
        plt.title(f"{col_name.replace("_", " ").upper()} Over Time by Region")
    plt.xlabel("Datetime")
    plt.ylabel(y_label)
    if not remove_legend:
        plt.legend(loc="best")
    plt.grid()
    plt.tight_layout()
    plt.show()


def plot_timeseries_raw_outage_by_region(
    data,
    col_names,
    y_label,
    title=None,
    region_filter=None,
    region_col="region",
    remove_legend=False,
    seasonal_bands=True,
    **plot_args,
):
    """_summary_

    Args:
        data (_type_): _description_
        col_names (_type_): _description_
        y_label (_type_): _description_
        title (_type_, optional): _description_. Defaults to None.
        region_filter (_type_, optional): _description_. Defaults to None.
        region_col (str, optional): _description_. Defaults to "region".
        remove_legend (bool, optional): _description_. Defaults to False.
        seasonal_bands (bool, optional): _description_. Defaults to True.

    Raises:
        ValueError: _description_
    """
    filtered_data = data.copy()
    if region_filter:
        if isinstance(region_filter, str):
            region_filter = re.split(r"[,\s;]+", region_filter)
        filtered_data = filtered_data[
            filtered_data[region_col].isin(region_filter)
        ].reset_index(drop=True)

        if filtered_data.empty:
            raise ValueError(
                f"No data found for the specified regions: {region_filter}"
            )
    unique_regions = filtered_data[region_col].unique()
    fig, axes = plt.subplots(
        nrows=len(unique_regions), ncols=1, figsize=(12, 6 * len(unique_regions))
    )
    if len(unique_regions) == 1:
        axes = [axes]
    for ax, region in zip(axes, unique_regions):
        group = filtered_data[filtered_data[region_col] == region].sort_values(
            "datetime_beginning_ept"
        )
        group.set_index("datetime_beginning_ept")[col_names].plot(
            ax=ax, alpha=0.6, **plot_args
        )
        if seasonal_bands:
            add_summer_winter_bands(group, "datetime_beginning_ept", ax=ax)
        if title:
            ax.set_title(f"{title} - {region}")
        else:
            format_titles = [i.replace("_", " ").capitalize() for i in col_names]
            ax.set_title(f"{' v. '.join(format_titles)} Over Time in {region}")
        ax.set_xlabel("Datetime")
        ax.set_ylabel(y_label)
        ax.grid()
        if not remove_legend:
            ax.legend(loc="best")
    plt.tight_layout()
    plt.show()


def plot_rolling_plots_by_region(
    data,
    col_name,
    y_label,
    title=None,
    region_filter=None,
    region_col="region",
    n_day_rolls=1,
    remove_legend=False,
    seasonal_bands=True,
    **plot_args,
):
    """Creates a rolling window timeseries plot.

    Args:
        data (pd.DataFrame): Dataframe containings data for visuals.
        col_name (str): Column name of the specific data for visuals.
        y_label (str): Label of the y-axis.
        title (str, optional): Title of chart. Defaults to None.
        region_filter (list of str, optional): List of region(s) to filter for. Defaults to None.
        region_col (str, optional): Column representing regions or categories. Defaults to "region".
        n_day_rolls (int, optional): Number of days to calculate the rolling average. Defaults to 1.
        remove_legend (bool, optional): Determines if a legend should be removed or not. Defaults to False.
        seasonal_bands (bool, optional): Determine if vertical bands indicating winter and summers months will be added to the plot. Defaults to True.

    Raises:
        ValueError: Region filter provided yielded no data.
    """
    filtered_data = data.copy()
    if region_filter:
        if isinstance(region_filter, str):
            region_filter = re.split(r"[,\s;]+", region_filter)
        filtered_data = filtered_data[
            filtered_data[region_col].isin(region_filter)
        ].reset_index(drop=True)

        if filtered_data.empty:
            raise ValueError(
                f"No data found for the specified regions: {region_filter}"
            )
    # Average out all node points by each hour
    agg_data = (
        filtered_data.groupby([region_col, "datetime_beginning_ept"])[col_name]
        .mean()
        .reset_index()
    )
    window_roll = 24 * n_day_rolls
    plt.figure(figsize=(12, 6))
    for region, group in agg_data.groupby(region_col):
        group = group.sort_values("datetime_beginning_ept")
        group.set_index("datetime_beginning_ept")[col_name].rolling(
            window=window_roll
        ).mean().plot(label=region, alpha=0.6, **plot_args)
    if seasonal_bands:
        add_summer_winter_bands(group, "datetime_beginning_ept")
    if title:
        plt.title(title)
    else:
        plt.title(
            f"{col_name.replace("_", " ").upper()} Over Time by Region {f'(Rolling Window = {window_roll/24} day(s))' if n_day_rolls >= 1 else f'(Rolling Window = {window_roll} hours)'}"
        )
    plt.xlabel("Datetime")
    plt.ylabel(y_label)
    if not remove_legend:
        plt.legend(loc="best")
    plt.grid()
    plt.tight_layout()
    plt.show()


def plot_emergency_triggers(
    data, start_date=None, end_date=None, rolling_window=7, use_rolling=True
):
    """
    Plot Emergency Trigger and Near Emergency Threshold over time with user-defined zoom range.

    Args:
        data (pd.DataFrame): DataFrame containing 'datetime_beginning_ept', 'total_committed',
                             'eco_max', 'emergency_triggered', 'near_emergency'.
        start_date (datetime): Datetime of when data should start at. Defaults to None.
        end_date (datetime): Datetime of when data should end at. Defaults to None.
        rolling_window (int, optional): Number of days for the rolling average (only applied if use_rolling is True). Default is 7.
        use_rolling (bool, optional): Whether to apply rolling averages to the plot.
    """
    if start_date or end_date:
        mask = (
            (data["datetime_beginning_ept"] >= pd.to_datetime(start_date))
            if start_date
            else True
        )
        mask &= (
            (data["datetime_beginning_ept"] <= pd.to_datetime(end_date))
            if end_date
            else True
        )
        filtered_data = data[mask].copy()
    else:
        filtered_data = data.copy()
    filtered_data = (
        filtered_data.groupby("datetime_beginning_ept")[
            ["total_committed", "eco_max", "emergency_triggered", "near_emergency"]
        ]
        .mean()
        .reset_index()
    )
    if use_rolling:
        filtered_data = (
            filtered_data.set_index("datetime_beginning_ept")
            .rolling(f"{rolling_window}D")
            .mean()
            .reset_index()
        )
        filtered_data = create_emergency_triggered(filtered_data)
        filtered_data = create_near_emergency(filtered_data)
    plt.figure(figsize=(12, 6))
    plt.plot(
        filtered_data["datetime_beginning_ept"],
        filtered_data["total_committed"],
        label="Total Committed (MW)",
        alpha=0.7,
    )
    plt.plot(
        filtered_data["datetime_beginning_ept"],
        filtered_data["eco_max"],
        label="Eco Max (MW)",
        alpha=0.7,
        color="orange",
    )
    emergency_points = filtered_data[filtered_data["emergency_triggered"] == 1]
    plt.scatter(
        emergency_points["datetime_beginning_ept"],
        emergency_points["total_committed"],
        label="Emergency Triggered",
        color="red",
        marker="o",
        alpha=0.8,
        s=30,
    )
    near_emergency_points = filtered_data[filtered_data["near_emergency"] == 1]
    plt.scatter(
        near_emergency_points["datetime_beginning_ept"],
        near_emergency_points["total_committed"],
        label="Near Emergency",
        color="blue",
        marker="^",
        alpha=0.5,
        s=25,
    )
    plt.title("Emergency Trigger and Near Emergency Threshold Over Time")
    plt.xlabel("Datetime")
    plt.ylabel("Power (MW)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def add_summer_winter_bands(data, date_col, ax=None):
    """Adds vertical bands highlighting summer (June to August) and winter (December to February) on line plots.

    Args:
        df (pd.DataFrame): Dataframe containing the date range for the plot.
        date_col (str): Column name of the column containing the datetime objects covering the date range of the plot.
        ax (matplotlib.axes.Axes, optional): Specific axis to draw the bands on. Defaults to None.
    """
    data_dte = pd.to_datetime(data[date_col].copy())
    first_day = data_dte.min()
    last_day = data_dte.max()
    start_year = data_dte.dt.year.min()
    end_year = data_dte.dt.year.max()
    summer_label_flag = False
    winter_label_flag = False
    if ax is None:
        ax = plt.gca()

    for year in range(start_year, end_year + 1):
        if last_day >= pd.Timestamp(f"{year}-06-01") and first_day <= pd.Timestamp(
            f"{year}-08-31"
        ):
            ax.axvspan(
                pd.Timestamp(f"{year}-06-01"),
                pd.Timestamp(f"{year}-08-31"),
                color="orange",
                alpha=0.1,
                label="Summer" if not summer_label_flag else None,
            )
            summer_label_flag = True
        if last_day >= pd.Timestamp(f"{year}-12-01") and first_day <= pd.Timestamp(
            f"{year+1}-02-28"
        ):
            ax.axvspan(
                pd.Timestamp(f"{year}-12-01"),
                pd.Timestamp(f"{year+1}-02-28"),
                color="blue",
                alpha=0.1,
                label="Winter" if not winter_label_flag else None,
            )
            winter_label_flag = True


def plot_region_stress_ratio(
    data,
    region_filter=None,
    region_col="region",
    rolling_hourly_window=None,
    **plot_args,
):
    """
    Plot the Region Stress Ratio over time and visualize stress levels by region.

    Args:
        data (pd.DataFrame): DataFrame with region stress ratio calculated.
        region_filter (list): List of regions to include.
        rolling_hourly_window (int): Rolling window size () for smoothing, if any.
    """
    # Filter and apply rolling average if needed
    filtered_data = data.copy()
    if region_filter:
        if isinstance(region_filter, str):
            region_filter = re.split(r"[,\s;]+", region_filter)
        filtered_data = filtered_data[
            filtered_data[region_col].isin(region_filter)
        ].reset_index(drop=True)

        if filtered_data.empty:
            raise ValueError(
                f"No data found for the specified regions: {region_filter}"
            )

    filtered_data["datetime_beginning_ept"] = filtered_data[
        "datetime_beginning_ept"
    ].dt.date
    agg_data = (
        filtered_data.groupby(["datetime_beginning_ept", region_col])[
            "region_stress_ratio"
        ]
        .mean()
        .reset_index()
    )
    plt.figure(figsize=(12, 6))
    for region, group in agg_data.groupby([region_col]):
        group = group.sort_values("datetime_beginning_ept")
        group.set_index("datetime_beginning_ept")["region_stress_ratio"].rolling(
            window=rolling_hourly_window
        ).mean().plot(label=region, alpha=0.6, **plot_args)

    # plt.axhline(100, color="red", linestyle="--", label="100% Stress Threshold")
    plt.title("Region Stress Ratio Over Time")
    plt.xlabel("Datetime")
    plt.ylabel("Stress Ratio (%)")
    plt.legend(title="Region", loc="upper left")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


### Modelling ###


# Emergency Triggers#
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

    et_setup_df = df.copy()
    et_setup_df = et_setup_df.sort_values(
        by=["datetime_beginning_ept", "region", "pnode_id"]
    )

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

    # add lagged features
    lags = [1, 3, 6, 24]
    for lag in lags:
        et_setup_df[f"near_emergency_lag{lag}"] = et_setup_df.groupby(["region"])[
            "near_emergency"
        ].shift(lag)
        et_setup_df[f"capacity_margin_lag{lag}"] = et_setup_df.groupby(["region"])[
            "capacity_margin"
        ].shift(lag)
        et_setup_df[f"lmp_volatility_lag{lag}"] = et_setup_df.groupby(["region"])[
            "lmp_volatility"
        ].shift(lag)

    # add rolling averages features
    rolling_windows = [3, 6, 24]
    for window in rolling_windows:
        et_setup_df[f"volatility_roll{window}"] = (
            et_setup_df.groupby(["region"])["lmp_volatility"]
            .rolling(window, min_periods=1)
            .mean()
            .reset_index(0, drop=True)
        )
        et_setup_df[f"region_stress_roll{window}"] = (
            et_setup_df.groupby(["region"])["region_stress_ratio"]
            .rolling(window, min_periods=1)
            .mean()
            .reset_index(0, drop=True)
        )

    et_setup_df = et_setup_df.dropna()
    onehot_encoder = OneHotEncoder(drop="first", sparse_output=False)
    onehot_encoded = onehot_encoder.fit_transform(et_setup_df[["region"]])
    region_columns = [f"region_{cat}" for cat in onehot_encoder.categories_[0][1:]]
    et_setup_df[region_columns] = onehot_encoded
    et_setup_df.drop(columns=["region"], inplace=True)

    selected_features = [
        "emergency_triggered",
        "datetime_beginning_ept",
        "pnode_id",
        "hour",
        "day_of_week",
        "season",
        "capacity_margin",
        "lmp_volatility",
        "region_stress_ratio",
        # "near_emergency",
        "near_emergency_lag1",
        "capacity_margin_lag24",
        "lmp_volatility_lag6",
        "volatility_roll24",
        "region_stress_roll6",
    ] + region_columns
    et_setup_df = et_setup_df[selected_features]

    et_setup_df.set_index("datetime_beginning_ept", inplace=True)

    return et_setup_df


def optimize_hyperparameters_et(model_name, X, y):
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
            RandomForestClassifier(random_state=42),
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


def custom_time_series_split_no_overlap(data, n_splits, gap=0):
    """
    Custom Time Series Split that prevents overlap of grouped data by unique time points.

    Args:
        data (pd.DataFrame): Input DataFrame.
        n_splits (int): Number of splits.
        gap (int, optional): Number of unique time points (in hours) to skip between training and testing sets. Defaults to 0.

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


def walk_forward_validation_et(
    data,
    target_column,
    models_to_use=None,
    n_splits=5,
):
    """Perform walk-forward validation with hyperparameter optimization for multiple models.

    Args:
        data (pd.DataFrame): Time-series data with datetime index.
        target_column (str): Name of the target column (Ex. "emergency_triggered").
        models_to_use (list of str): List of model names to run, specifically ["decision_tree", "random_forest", "lightgbm"]. Defaults to None.
        n_splits (int, optional): Number of splits for TimeSeriesSplit. Default to 5.

    Returns:
        dict: Results for each model, including metrics for each fold and overall performance.
    """
    data = data.sort_index()
    X = data.drop(columns=[target_column])
    y = data[target_column]
    # tscv = TimeSeriesSplit(n_splits=n_splits, gap=24)
    splitter = custom_time_series_split_no_overlap(data, n_splits, gap=24)
    model_names = models_to_use
    results = {model_name: [] for model_name in model_names}

    for fold, (train_idx, test_idx) in enumerate(splitter, start=1):
        print(f"\n____ Fold {fold} ____")
        X_train, X_test = X.loc[train_idx], X.loc[test_idx]
        y_train, y_test = y.loc[train_idx], y.loc[test_idx]
        for model_name in model_names:
            print(f"Optimizing {model_name}...")
            best_model = optimize_hyperparameters_et(model_name, X_train, y_train)
            y_pred = best_model.predict(X_test)
            report = classification_report(y_test, y_pred, output_dict=True)
            results[model_name].append(
                {
                    "fold": fold,
                    "model": best_model,
                    "f1_score": report["weighted avg"]["f1-score"],
                    "y_test": y_test,
                    "y_pred": y_pred,
                }
            )

            print(
                f"Fold {fold} - {model_name}: F1-Score = {report['1']['f1-score']:.2f}"
            )

    return results


def display_results_et(results):
    """Displays the results gathered from walk_forward_validation_et.

    Args:
        results (dict): Results for each model, including metrics for each fold and overall performance.
    """
    folds = []
    f1_scores = []
    for dicts in results["decision_tree"]:
        folds.append(dicts["fold"])
        f1_scores.append(dicts["f1_score"])
    f1_scores

    plt.figure(figsize=(8, 5))
    plt.plot(folds, f1_scores, marker="o", linestyle="-", color="blue")
    plt.title("Decision Tree F1-Scores Across Folds")
    plt.xlabel("Fold")
    plt.ylabel("F1-Score")
    plt.ylim(0.9, 1.1)
    plt.grid()
    plt.show()
