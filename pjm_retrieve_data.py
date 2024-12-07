import requests
import pandas as pd
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from multiprocessing import Pool, cpu_count
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class PJMHeaderSetUp:
    """A class to manage and retrieve the subscription key and headers needed for requests to the PJM API."""

    def __init__(self, settings_url="https://dataminer2.pjm.com/config/settings.json"):
        """Initializes the PJMHeaderSetUp instance with a settings URL. Automatically fetches the subscription key and constructs the header.

        Args:
            settings_url (str, optional): The URL to fetch settings from. Defaults to "https://dataminer2.pjm.com/config/settings.json".
        """
        self._settings_url = settings_url
        self._key = None
        self._header = None
        self._fetch_subscription_key()

    def _fetch_subscription_key(self):
        """Private method to retrieve the subscription key from the settings URL. Sets the `_key` and `_header` attributes based on the response.

        Raises:
            requests.exceptions.RequestException: If the request to fetch the key fails.
        """
        response = requests.get(self._settings_url)
        if response.status_code == 200:
            data = response.json()
            self._key = data["subscriptionKey"]
            self._header = {"Ocp-Apim-Subscription-Key": self._key}
        else:
            raise requests.exceptions.RequestException(
                f"Failed to retrieve data from {self._settings_url}: {response.status_code} - {response.text}"
            )

    @property
    def settings_url(self):
        """Getter fir settings_url.

        Returns:
            str: Property to get the current settings URL.
        """
        return self._settings_url

    @settings_url.setter
    def settings_url(self, new_url):
        """Sets a new settings_url, which also triggers a refresh of key and header.

        Args:
            new_url (str): The new URL to set as the settings URL.

        Raises:
            Exception: If the request to fetch the key fails.
        """
        self._settings_url = new_url
        try:
            self._fetch_subscription_key()
        except requests.exceptions.RequestException as e:
            raise Exception(f"Error fetching subscription key: {e}")

    @settings_url.deleter
    def settings_url(self):
        """Deleter for settings_url. Sets `_settings_url`, `_key`, and `_header` attributes to None."""
        print("Deleting settings_url")
        self._settings_url = None
        self._key = None
        self._header = None

    # Getter/Setter for key
    @property
    def key(self):
        """Getter for key.

        Returns:
            str: Property to get the current subscription key.
        """
        return self._key

    @key.setter
    def key(self, new_key):
        """Sets a new subscription key and updates the header.

        Args:
            new_key (str): The new subscription key to set.
        """
        self._key = new_key
        self._header = {"Ocp-Apim-Subscription-Key": self._key}

    @key.deleter
    def key(self):
        """Deleter for subscription key. Sets `_key` and `_header` attributes to None."""
        print("Deleting subscription key")
        self._key = None
        self._header = None

    # Getter/Setter for header
    @property
    def header(self):
        """Getter for header.

        Returns:
            dict: Property to get the current header containing the subscription key.
        """
        return self._header

    @header.setter
    def header(self, new_header):
        """Sets a new header dictionary.

        Args:
            new_header (dict): The new header dictionary to set.
        """
        self._header = new_header

    @header.deleter
    def header(self):
        """Deletes the header. Sets `_header` attribute to None."""
        print("Deleting header")
        self._header = None

    def refresh_key(self):
        """Force refreshes the subscription key by re-fetching it from the settings URL.

        Raises:
            Exception: If the request to fetch the key fails.
        """
        try:
            self._fetch_subscription_key()
        except requests.exceptions.RequestException as e:
            raise Exception(f"Error fetching subscription key: {e}")


class PJMMonthlyFTRBidsFetcher(PJMHeaderSetUp):
    """A class to retrieve monthly FTR bids."""

    def __init__(
        self,
        market_name=None,
        start_datetime=None,
        end_datetime=None,
        is_range=True,
        row_count=50000,
        keep_true_ups=True,
    ):
        """Initialize the PJMMonthlyFTRBidsFetcher with options for date range or specific market name.

        Args:
            market_name (str, optional): A specific market name (e.g., "JAN 2022 Auction") to fetch data for.
          If provided, only data for this market name will be fetched.
            start_datetime(tuple of int, optional): ... Defaults to None.
            end_datetime(tuple of int, optional): ... Defaults to None.
            is_range (bool, optional): Determines if the retrieved data is from a date range (True) or a static date (False). Defaults to True.
          as integers, representing the start and end dates (e.g., ((2014, 7), (2022, 8))).
            row_count (int, optional): Number of rows to fetch per request. Defaults to 50000.
            keep_true_ups (bool, optional): Whether to keep true up bids.
        """

        super().__init__()

        if market_name:  # Single auction name (Target Specific "MMM YYYY Auction" - Ex. DEC 2022 Auction)
            self.market_name = market_name
            self.date_range_mode = False
        elif start_datetime is not None and end_datetime is None:
            self._catch_datetime_entry(start_datetime)
            if is_range:  # Want all data from start_datetime to whatever is available
                self.start_date = self.create_datetime(start_datetime)
                self.end_date = self._default_end_date()
                self.date_range_mode = True
            else:  # Similar to providing a market_name; however as a tuple of integers instead of string
                self.market_name = self.create_auction_label(start_datetime)
                self.date_range_mode = False
        elif start_datetime is None and end_datetime is not None:
            self._catch_datetime_entry(end_datetime)
            if is_range:  # Want all data from whatever is available to end_datetiem
                self.start_date = self.create_datetime((2006, 1))
                self.end_date = self.create_datetime(end_datetime)
                self.date_range_mode = True
            else:
                self.market_name = self.create_auction_label(end_datetime)
                self.date_range_mode = False
        elif start_datetime is not None and end_datetime is not None:
            self._catch_datetime_entry(start_datetime)
            self._catch_datetime_entry(end_datetime)
            self.start_date = self.create_datetime(start_datetime)
            self.end_date = self.create_datetime(end_datetime)
            self.date_range_mode = True
        else:  # Both start_datetime and end_datetime are None
            self.start_date = self.create_datetime((2006, 1))
            self.end_date = self._default_end_date()
            self.date_range_mode = True

        self.row_count = row_count
        self.base_url = "https://api.pjm.com/api/v1/ftr_bids_mnt"
        self.keep_true_ups = keep_true_ups

    def _catch_datetime_entry(self, dte):
        """Checks if the tuple representing the date has integer values.

        Args:
            dte (tuple of int): Tuple of length two containing the year and day respectively.

        Raises:
            ValueError: The dte provided is nto a tuple of integers.
        """
        if not (
            isinstance(dte, tuple)
            and len(dte) == 2
            and all(isinstance(entry, int) for entry in dte)
        ):
            raise ValueError(
                "start_datetime and end_datetime must be a tuple of integers (year, month), ex. (2017, 8) for August 2017."
            )

    def create_auction_label(self, year_month_tuple):
        """Takes a tuple of int (year, day) and converts it into a market_name specific auction.

        Args:
            year_month_tuple (tuple of int): Tuple of length two containing the year and day respectively.

        Returns:
            str: String formated as the market_name to access.
        """
        year, month = year_month_tuple
        dte = datetime(year, month, 1)
        return dte.strftime("%b %Y ").upper() + "Auction"

    def create_datetime(self, year_month_tuple):
        """Converts a tuple contain a year and month value into a datetime object.

        Args:
            year_month_tuple (tuple): A tuple contain (year, month) as integers.

        Returns:
            datetime: A datetime object.
        """
        year, month = year_month_tuple
        return datetime(year, month, 1)

    def _default_end_date(self):
        """Takes the current date and returns the tuple of int represent the current year and integer representing the past 4th month.

        Returns:
            tuple of int: Tuple of length two containing the year and day respectively.
        """
        four_months_ago = datetime.now() - relativedelta(months=4)
        return self.create_datetime((four_months_ago.year, four_months_ago.month))

    def _default_market_name(self):
        """Calculate the market name based on 4 months ago (Note: Monthly FTR bids are posted on a four month delay and are updated the first of every month).

        Returns:
            str: A market_name string.
        """
        four_months_ago = datetime.now() - relativedelta(months=4)
        return four_months_ago.strftime("%b %Y ").upper() + "Auction"

    def get_monthly_ftr_bids(
        self, market_name, start_row=1, row_count=None, download=True, **kwargs
    ):
        """Fetch a single batch of FTR bids data for a specific market name.

                Args:
                    market_name (str): An auction to access.
                    start_row (int, optional): The starting row number for the batch to fetch. Defaults to 1.
                    row_count (int, optional): The number of rows to fetch. Defaults to 50000.
                    download (bool, optional): Determines how results should be returned (If True, no
        links or search criteria will be echoed). Defaults to True.

                Raises:
                    requests.exceptions.RequestException: Request was unsucessful

                Returns:
                    pd.DataFrame: A dataframe of a single batch of FTR bids data for a specific market name.
        """
        headers = self.header
        params = {
            "market_name": market_name,
            "download": download,
            "rowCount": row_count if row_count is not None else self.row_count,
            "startRow": start_row,
        }
        params.update(kwargs)
        response = requests.get(self.base_url, headers=headers, params=params)
        if response.status_code == 200:
            data = response.json()
            return pd.DataFrame(data)
        else:
            raise requests.exceptions.RequestException(
                f"Failed to retrieve data from {self._settings_url}: {response.status_code} - {response.text}"
            )

    def _fetch_single_batch(self, args):
        """Helper function to fetch a single batch of data for multiprocessing.

        Args:
            args (tuple): A tuple containing the arguments in the following order:
                - market_name (str): The name of the market (e.g., "JUL 2024 Auction").
                - start_row (int): The starting row number for the batch to fetch.
                - kwargs (dict): A dictionary of additional keyword arguments for filtering or configuration.

        Returns:
            pd.DataFrame: A DataFrame containing the fetched data for the specified batch.
        """
        market_name, start_row, kwargs = args
        return self.get_monthly_ftr_bids(market_name, start_row, **kwargs)

    def fetch_all_monthly_ftr_bids_for_market(self, market_name, **kwargs):
        """Fetch all FTR bids data for a single market name by handling pagination using multiprocessing.

        Args:
            market_name (str): A market_name to target what auction data to access.

        Returns:
            pd.DataFrame: A new DataFrame with all FTR bids from a single market name.
        """
        logging.info(f"Checking data availability for market_name: {market_name}")
        data_check = self.get_monthly_ftr_bids(market_name, row_count=1, **kwargs)
        if data_check.empty:
            logging.warning(f"No data found for market_name: {market_name}")
            return pd.DataFrame()

        max_batches = int(1.5 * cpu_count())  # current conservative upperbound
        batch_starts = [
            (market_name, i * self.row_count + 1, kwargs) for i in range(max_batches)
        ]
        all_data_df = pd.DataFrame()

        while batch_starts:
            with Pool(cpu_count()) as pool:
                results = pool.map(self._fetch_single_batch, batch_starts)

            non_empty_results = [df for df in results if not df.empty]
            all_data_df = pd.concat(
                [all_data_df] + non_empty_results, axis=0, ignore_index=True
            )

            if any(df.empty for df in results):
                break

            # If there is more data to pull
            last_row = max([batch[1] - 1 for batch in batch_starts])
            batch_starts = [
                (market_name, last_row + i * self.row_count + 1, kwargs)
                for i in range(max_batches)
            ]

        if not self.keep_true_ups:
            all_data_df = self.remove_true_ups(all_data_df, market_name)

        logging.info(f"Completed downloading data for market_name: {market_name}")
        return all_data_df.reset_index(drop=True)

    def remove_true_ups(self, df, market_name):
        """Removes all True-up bids. Refer to: https://www.pjm.com/-/media/committees-groups/task-forces/frmstf/20191018/20191018-item-04-ftr-product-range-and-auction-process-oa-redlines.ashx

        Args:
            df (pd.DataFrame): A DataFrame containing all bids from the specified market_name.
            market_name (str): A string representing the auction that is accessed.

        Returns:
            pd.DataFrame: A DataFrame with true-up bids
        """
        cutoff_date = datetime.strptime(market_name.split()[0], "%b").replace(
            year=int(market_name.split()[1])
        )

        def parse_period_type(month_str):
            """Converts quarterly FTR contracts to the month corresponding to its expiry month and removes any expire ("true up") contracts.

            Args:
                month_str (str): String appreviation of the months or quarters ("period_type").

            Returns:
                pd.DataFrame: A dataframe containing only active contracts for the market_name (ie. auction) that was provided.
            """
            quarter_to_month = {"Q1": "MAR", "Q2": "JUN", "Q3": "SEP", "Q4": "DEC"}

            if month_str in quarter_to_month:
                month_str = quarter_to_month[month_str]
            try:
                month = datetime.strptime(month_str, "%b").month
                return datetime(cutoff_date.year, month, 1)
            except ValueError:
                logging.warning(f"Unexpected period_type format: {month_str}")
                return None

        df.loc[:, "period_date"] = df["period_type"].apply(parse_period_type)
        filtered_df = df[df["period_date"] >= cutoff_date]
        filtered_df = filtered_df.drop(columns=["period_date"])

        logging.info(f"Removed True-up bids from {market_name}")
        return filtered_df

    def save_data(self, output_file_path, file_name, **kwargs):
        """Saves FTR bids data based on the initialized configuration into the provided output_file_path and file_name. This method fetches data across a range of months if `date_range` was specified during initialization,
        or for a single month if `market_name` was provided OR if either start_datetime or end_datetime is provided.

        Args:
            output_file_path (str): File path to save parquet file.
            file_name (str): Name of the file.
        """
        if output_file_path[-1] == "/":
            file_path = f"{output_file_path}{file_name}.parquet"
        else:
            file_path = f"{output_file_path}/{file_name}.parquet"
        if self.date_range_mode:
            current_date = self.start_date
            final_df = pd.DataFrame()

            while current_date <= self.end_date:
                market_name = current_date.strftime("%b %Y ") + "Auction"
                logging.info(f"Fetching data for market_name: {market_name}")

                monthly_data = self.fetch_all_monthly_ftr_bids_for_market(
                    market_name, **kwargs
                )
                final_df = pd.concat(
                    [final_df, monthly_data], axis=0, ignore_index=True
                )
                current_date += relativedelta(months=1)
            final_df.to_parquet(file_path, index=False, engine="pyarrow")
            logging.info(f"All data saved to {output_file_path}")
        else:
            logging.info(f"Fetching data for market_name: {self.market_name}")
            final_df = self.fetch_all_monthly_ftr_bids_for_market(
                self.market_name, **kwargs
            )
            final_df.to_parquet(file_path, index=False, engine="pyarrow")
            logging.info(f"Data saved to {output_file_path}")

    def fetch_data(self, **kwargs):
        """Fetch FTR bids data based on the initialized configuration. This method fetches data across a range of months if `date_range` was specified during initialization,
        or for a single month if `market_name` was provided OR if either start_datetime or end_datetime is provided.

        Returns:
            pd.DataFrame: A dataframe containing monthly FTR Bids for the defined parameters.
        """
        if self.date_range_mode:
            current_date = self.start_date
            final_df = pd.DataFrame()

            while current_date <= self.end_date:
                market_name = current_date.strftime("%b %Y ") + "Auction"
                logging.info(f"Fetching data for market_name: {market_name}")

                monthly_data = self.fetch_all_monthly_ftr_bids_for_market(
                    market_name, **kwargs
                )
                final_df = pd.concat(
                    [final_df, monthly_data], axis=0, ignore_index=True
                )
                current_date += relativedelta(months=1)
            logging.info("All data fetched.")
            return final_df
        else:
            logging.info(f"Fetching data for market_name: {self.market_name}")
            final_df = self.fetch_all_monthly_ftr_bids_for_market(
                self.market_name, **kwargs
            )
            logging.info("All data fetched.")
            return final_df


class PJMMonthlyFTRZonalLmps(PJMHeaderSetUp):
    """A class to retrieve monthly FTR zonal LMPs."""

    def __init__(
        self,
        start_datetime=None,
        end_datetime=None,
        remove_day_light_saving=True,
        is_range=True,
        row_count=50000,
    ):
        """Initialize the PJMMonthlyFTRZonalLmps with options for date ranges.

        Args:
            start_datetime (datetime, optional): A datetime object of the start date. Defaults to None.
            end_datetime (datetime, optional): A datetime object of the end date; When included with a start_datetime, this datetime will be INCLUSIVE in the data retrieved. Defaults to None.
            remove_day_light_saving (bool, optional): Determines if daylight saving datapoints are removed. Defaults to True.
            is_range (bool, optional): Determines if the retrieved data is from a date range (True) or a static date (False). Defaults to True.
            row_count (int, optional): Number of rows to fetch per request. Defaults to 50000.

        Raises:
            ValueError: end_datetime was before start_datetime.
        """
        super().__init__()

        if start_datetime is not None and end_datetime is not None:
            days_diff = (end_datetime - start_datetime).days
            if start_datetime > end_datetime:
                raise ValueError("start_datetime must be earlier than end_datetime")
            elif days_diff > 366:
                self.start_date = start_datetime
                self.end_date = None
                self.upper_bound = end_datetime
                self.date_range_mode = True
            else:
                self.start_date = f"{self.convert_datetime_string(start_datetime)} to {self.convert_datetime_string(end_datetime)}"
                self.end_date = None
                self.upper_bound = end_datetime
                self.date_range_mode = False
        elif start_datetime is None and end_datetime is not None:
            end_date = self.convert_datetime_string(end_datetime)
            self.date_range_mode = False
            if is_range:
                self.start_date = start_datetime
                self.end_date = f"to {end_date}"
                self.upper_bound = None
            else:
                print(end_date)
                self.start_date = end_datetime - timedelta(hours=1)
                print(self.start_date)
                self.end_date = None
                self.upper_bound = end_date
        elif start_datetime is not None and end_datetime is None:
            self.end_date = end_datetime
            self.date_range_mode = False
            start_date = self.convert_datetime_string(start_datetime)
            if is_range:
                self.start_date = f"{start_date} to"
                self.upper_bound = None
            else:
                self.start_date = start_date
                self.upper_bound = start_datetime + timedelta(hours=1)
        else:
            self.start_date = datetime(
                2010, 8, 1, 0, 0, 0
            )  # starting date provided by pjm
            self.end_date = end_datetime
            self.upper_bound = datetime.now()
            self.date_range_mode = True
        self.is_range = is_range
        self.remove_dsl = remove_day_light_saving
        self.row_count = row_count
        self.base_url = "https://api.pjm.com/api/v1/mnt_ftr_zonal_lmps"

    def convert_datetime_string(self, dt):
        """Convert datetime objects to string.

        Args:
            dt (datetime): A datetime object.

        Returns:
            str: A string converted from datetime object in format MM/DD/YYYY HH:MM:SS.
        """
        format_datetime = dt.strftime("%m/%d/%y %H:%M:%S")
        return format_datetime

    def _treat_day_light_saving(self, df):
        """Removes any overlapping or replaces any missing hourly periods due to day light saving.

        Args:
            df (pd.DataFrame): Dataframe of zonal LMPs that will be treated for day light saving hourly periods.

        Returns:
            pd.DataFrame: Dataframe of zonal LMPs that have the day light saving hourly periods treated.
        """
        df["hourly_delta"] = (
            df["datetime_ending_ept"] - df["datetime_beginning_ept"]
        ).dt.total_seconds() // 3600

        def split_to_hourly(row):
            num_hours = int(row["hourly_delta"])
            rows = []
            for i in range(num_hours):
                new_row = row.copy()
                new_row["datetime_beginning_ept"] = row[
                    "datetime_beginning_ept"
                ] + pd.Timedelta(hours=i)
                new_row["datetime_ending_ept"] = new_row[
                    "datetime_beginning_ept"
                ] + pd.Timedelta(hours=1)
                rows.append(new_row)
            return rows

        to_split = df[df["hourly_delta"] > 1]
        if to_split.empty:
            df.drop(columns=["hourly_delta"], inplace=True)
            return df
        split_rows = pd.DataFrame(
            [row for rows in to_split.apply(split_to_hourly, axis=1) for row in rows]
        )
        no_split = df[df["hourly_delta"] == 1]
        final_data = pd.concat([split_rows, no_split], ignore_index=True).drop(
            columns="hourly_delta"
        )
        final_data = final_data.sort_values(
            by=["pnode_name", "datetime_beginning_ept"]
        ).reset_index(drop=True)
        return final_data

    def build_date_ranges(self, start_dte, end_dte, max_days=366):
        """Creates a list of date ranges in string type as "start_date to end_date".

        Args:
            start_dte (datetime): Start of datetime range.
            end_dte (datetime): End of datetime range.
            max_days (int, optional): Length of each datetime range in days. Defaults to 366.

        Returns:
            list of str: List of strings of date range.
        """
        date_ranges = []
        current_start = start_dte

        while current_start < end_dte:
            current_end = min(current_start + timedelta(days=max_days), end_dte)
            range_str = f"{self.convert_datetime_string(current_start)} to {self.convert_datetime_string(current_end)}"
            date_ranges.append(range_str)
            current_start = current_end + timedelta(hours=1)
        return date_ranges

    def get_monthly_ftr_zonal_lmps(
        self, start_dte, end_dte, start_row=1, row_count=None, download=True, **kwargs
    ):
        """Fetch a single batch of monthly FTR Zonal LMPs data.

        Args:
            start_dte (str, optional): The starting date to retrieve data.
            end_dte (str, optional): The ending date to retrieve data.
            start_row (int, optional): The starting row number for the batch to fetch. Defaults to 1.
            row_count (int, optional): The number of rows to fetch. Defaults to 50000.
            download (bool, optional): Determines how results should be returned (If True, no
        links or search criteria will be echoed). Defaults to True.

        Raises:
            requests.exceptions.RequestException: Request was unsucessful.

        Returns:
            pd.DataFrame: A dataframe of a single batch of monthly FTR Zonal LMPs data.
        """
        headers = self.header
        params = {
            "datetime_beginning_ept": start_dte,
            "datetime_ending_ept": end_dte,
            "download": download,
            "rowCount": row_count if row_count is not None else self.row_count,
            "startRow": start_row,
        }
        params.update(kwargs)
        response = requests.get(self.base_url, headers=headers, params=params)
        if response.status_code == 200:
            data = response.json()
            return pd.DataFrame(data)
        else:
            raise requests.exceptions.RequestException(
                f"Failed to retrieve data from {self._settings_url}: {response.status_code} - {response.text}"
            )

    def _fetch_single_batch(self, args):
        """Process a single batch of data with time range of at least 366 days.

        Args:
            args (tuple): A tuple containings the arguments in the following order:
                - start_dte (str): Start date of the batch in collecting zonal LMP data.
                - end_dte (str): End data of the batch in collecting zonal LMP data.
                - start_row (int): The starting row number for the batch to fetch.

        Returns:
            pd.DataFrame: A dataframe containing data received from the request sent.
        """
        start_dte, end_dte, start_row, kwargs = args
        return self.get_monthly_ftr_zonal_lmps(
            start_dte=start_dte, end_dte=end_dte, start_row=start_row, **kwargs
        )

    def fetch_all_ftr_zonal_lmp_for_daterange(self, start_dte, end_dte, **kwargs):
        """Starting point that develops and processes all the batches of zonal LMP data between the start and end dates.

        Args:
            start_dte (str): Start date all zonal LMP data.
            end_dte (str): End date all zonal LMP data.

        Returns:
            pd.DataFrame: A dataframe containing all the batch data concatenated together.
        """
        if start_dte is not None and self.upper_bound is not None:
            logging.info(f"Checking data availability for: {start_dte}")
        elif start_dte is not None and end_dte is None:
            if self.is_range:
                logging.info(
                    f"Checking data availability for: {start_dte} Most Updated Data Point Available."
                )
            else:
                logging.info(f"Checking data availability for: {start_dte}.")
        else:
            if self.is_range:
                logging.info(
                    f"Checking data availability for: 08/01/10 00:00:00 {end_dte}"
                )
            else:
                logging.info(f"Checking data availability for: {end_dte}")
        data_check = self.get_monthly_ftr_zonal_lmps(
            start_dte=start_dte, end_dte=end_dte
        )
        if data_check.empty:
            logging.warning("No data found for market_name")
            return pd.DataFrame()

        max_batches = int(1.5 * cpu_count())
        batch_starts = [
            (start_dte, end_dte, i * self.row_count + 1, kwargs)
            for i in range(max_batches)
        ]
        all_data_df = pd.DataFrame()

        while batch_starts:
            with Pool(cpu_count()) as pool:
                results = pool.map(self._fetch_single_batch, batch_starts)

            non_empty_results = [df for df in results if not df.empty]
            all_data_df = pd.concat(
                [all_data_df] + non_empty_results, axis=0, ignore_index=True
            )

            if any(df.empty for df in results):
                break

            last_row = max([batch[1] - 1 for batch in batch_starts])
            batch_starts = [
                (start_dte, end_dte, last_row + i * self.row_count + 1, kwargs)
                for i in range(max_batches)
            ]

        if start_dte is not None and self.upper_bound is not None:
            logging.info(f"Completed downloading data for range: {start_dte}")
        elif start_dte is not None and end_dte is None:
            if self.is_range:
                logging.info(
                    f"Completed downloading data for range: {start_dte} Most Updated Data Point Available."
                )
            else:
                logging.info(f"Completed downloading data for range: {start_dte}")
        else:
            if self.is_range:
                logging.info(
                    f"Completed downloading data for range: 08/01/10 00:00:00 {end_dte}"
                )
            else:
                logging.info(f"Completed downloading data for range: {end_dte}")
        return all_data_df.reset_index(drop=True)

    def save_data(self, output_file_path, file_name, **kwargs):
        """Takes the dataframe of data concatenates from all batches and outputs it into the provided file path under the provided file name.

        Args:
            output_file_path (str): File path to save parquet file.
            file_name (str): Name of the file.
        """
        if output_file_path[-1] == "/":
            file_path = f"{output_file_path}{file_name}.parquet"
        else:
            file_path = f"{output_file_path}/{file_name}.parquet"
        if self.date_range_mode:
            date_ranges = self.build_date_ranges(self.start_date, self.upper_bound)
            final_df = pd.DataFrame()

            for date_rng in date_ranges:
                logging.info(f"Fetching data for range: {date_rng}")
                current_data = self.fetch_all_ftr_zonal_lmp_for_daterange(
                    start_dte=date_rng, end_dte=self.end_date, **kwargs
                )
                final_df = pd.concat(
                    [final_df, current_data], axis=0, ignore_index=True
                )
        else:
            logging.info("Fetching data...")
            final_df = self.fetch_all_ftr_zonal_lmp_for_daterange(
                start_dte=self.start_date, end_dte=self.end_date, **kwargs
            )

        datetime_columns = [
            "datetime_beginning_utc",
            "datetime_beginning_ept",
            "datetime_ending_utc",
            "datetime_ending_ept",
        ]
        for col in datetime_columns:
            if col in final_df.columns:
                final_df[col] = pd.to_datetime(final_df[col])

        if self.upper_bound is not None:
            final_df = final_df[final_df["datetime_ending_ept"] <= self.upper_bound]
            final_df.reset_index(drop=True, inplace=True)
        if self.remove_dsl:
            final_df = final_df[
                final_df["datetime_beginning_ept"] < final_df["datetime_ending_ept"]
            ]
            final_df = self._treat_day_light_saving(final_df)
        final_df.to_parquet(file_path, index=False, engine="pyarrow")
        logging.info(f"Data saved to {file_path}")

    def fetch_data(self, **kwargs):
        """Takes the dataframe of data concatenates from all batches and returns it.

        Returns:
            pd.DataFrame: A dataframe containing monthly FTR Zonal LMPs for the defined parameters.
        """
        if self.date_range_mode:
            date_ranges = self.build_date_ranges(self.start_date, self.upper_bound)
            final_df = pd.DataFrame()

            for date_rng in date_ranges:
                logging.info(f"Fetching data for range: {date_rng}")
                current_data = self.fetch_all_ftr_zonal_lmp_for_daterange(
                    start_dte=date_rng, end_dte=self.end_date, **kwargs
                )
                final_df = pd.concat(
                    [final_df, current_data], axis=0, ignore_index=True
                )
        else:
            logging.info("Fetching data...")
            final_df = self.fetch_all_ftr_zonal_lmp_for_daterange(
                start_dte=self.start_date, end_dte=self.end_date, **kwargs
            )

        datetime_columns = [
            "datetime_beginning_utc",
            "datetime_beginning_ept",
            "datetime_ending_utc",
            "datetime_ending_ept",
        ]
        for col in datetime_columns:
            if col in final_df.columns:
                final_df[col] = pd.to_datetime(final_df[col])

        if self.upper_bound is not None:
            final_df = final_df[final_df["datetime_ending_ept"] <= self.upper_bound]
            final_df.reset_index(drop=True, inplace=True)
        if self.remove_dsl:
            final_df = final_df[
                final_df["datetime_beginning_ept"] < final_df["datetime_ending_ept"]
            ]
            final_df = self._treat_day_light_saving(final_df)
        logging.info("All data fetched.")
        return final_df


class PJMDailyGenerationCapacity(PJMHeaderSetUp):
    """A class to retrieve daily generation capacity."""

    def __init__(
        self, start_datetime=None, end_datetime=None, is_range=True, row_count=50000
    ):
        """Initialize the PJMDailyGenerationCapacity.

        Args:
            start_datetime (datetime, optional): A datetime object of the start date. Defaults to None.
            end_datetime (datetime, optional): A datetime object of the end date; When included with a start_datetime, this datetime will be INCLUSIVE in the data retrieved. Defaults to None.
            is_range (bool, optional): Determines if the retrieved data is from a date range (True) or a static date (False). Defaults to True.
            row_count (int, optional): Number of rows to fetch per request. Defaults to 50000.
        """
        super().__init__()

        if start_datetime is not None:
            start_date = self.convert_datetime_string(start_datetime)
            if is_range:
                self.start_date = f"{start_date} to"
            else:
                self.start_date = start_date
        else:
            self.start_date = start_datetime
        self.end_date = end_datetime
        self.row_count = row_count
        self.base_url = "https://api.pjm.com/api/v1/day_gen_capacity"

    def convert_datetime_string(self, dt):
        """Convert datetime objects to string.

        Args:
            dt (datetime): A datetime object.

        Returns:
            str: A string converted from datetime object in format MM/DD/YYYY HH:MM:SS.
        """
        format_datetime = dt.strftime("%m/%d/%y %H:%M:%S")
        return format_datetime

    def get_daily_generation_capacity(
        self, start_dte=None, start_row=1, row_count=None, download=True, **kwargs
    ):
        """Fetch a single batch of monthly FTR Zonal LMPs data.

        Args:
            start_dte (str, optional): The starting date to retrieve data. Defaults to None.
            start_row (int, optional): The starting row number for the batch to fetch. Defaults to 1.
            row_count (int, optional): The number of rows to fetch. Defaults to 50000.
            download (bool, optional): Determines how results should be returned (If True, no
        links or search criteria will be echoed). Defaults to True.

        Raises:
            requests.exceptions.RequestException: Request was unsucessful.

        Returns:
            pd.DataFrame: A dataframe of a single batch of daily generation capacity.
        """
        headers = self.header
        params = {
            "bid_datetime_beginning_ept": start_dte,
            "download": download,
            "rowCount": row_count if row_count is not None else self.row_count,
            "startRow": start_row,
        }
        params.update(kwargs)
        response = requests.get(self.base_url, headers=headers, params=params)
        if response.status_code == 200:
            data = response.json()
            return pd.DataFrame(data)
        else:
            raise requests.exceptions.RequestException(
                f"Failed to retrieve data from {self._settings_url}: {response.status_code} - {response.text}"
            )

    def save_data(self, output_file_path, file_name, **kwargs):
        """Saves daily generation capacity data based on the initialized configuration into the provided output_file_path and file_name. This method fetches data across a range of months if `date_range` was specified during initialization,
        or for a single day if is_range is set False.

        Args:
            output_file_path (str): File path to save parquet file.
            file_name (str): Name of the file.
        """
        if output_file_path[-1] == "/":
            file_path = f"{output_file_path}{file_name}.parquet"
        else:
            file_path = f"{output_file_path}/{file_name}.parquet"

        final_df = pd.DataFrame()

        empty = False
        count = 1
        logging.info("Checking data availability.")
        data_check = self.get_daily_generation_capacity(
            start_dte=self.start_date, row_count=1, **kwargs
        )
        if data_check.empty:
            logging.warning("No data found.")
            final_df = pd.DataFrame()
        else:
            logging.info(f"Fetching Daily Generation Capacity from {self.start_date}.")
            while not empty:
                monthly_data = self.get_daily_generation_capacity(
                    start_dte=self.start_date,
                    start_row=(count - 1) * self.row_count + 1,
                    **kwargs,
                )
                if monthly_data.empty:
                    break
                final_df = pd.concat(
                    [final_df, monthly_data], axis=0, ignore_index=True
                )
                count += 1

        datetime_columns = ["bid_datetime_beginning_ept"]
        for col in datetime_columns:
            if col in final_df.columns:
                final_df[col] = pd.to_datetime(final_df[col])

        if self.end_date is not None:
            final_df = final_df[final_df["bid_datetime_beginning_ept"] <= self.end_date]
            final_df.reset_index(drop=True, inplace=True)
        final_df.to_parquet(file_path, index=False, engine="pyarrow")
        logging.info(f"Data saved to {file_path}")

    def fetch_data(self, **kwargs):
        """Fetches daily generation capacity data based on the initialized configuration. This method fetches data across a range of months if `date_range` was specified during initialization,
        or for a single day if is_range is set False.

        Returns:
            pd.DataFrame: A dataframe containing daily Generation Capacity data for the defined parameters.
        """
        final_df = pd.DataFrame()

        empty = False
        count = 1
        logging.info("Checking data availability.")
        data_check = self.get_daily_generation_capacity(
            start_dte=self.start_date, row_count=1, **kwargs
        )
        if data_check.empty:
            logging.warning("No data found.")
            final_df = pd.DataFrame()
        else:
            logging.info(f"Fetching Daily Generation Capacity from {self.start_date}.")
            while not empty:
                monthly_data = self.get_daily_generation_capacity(
                    start_dte=self.start_date,
                    start_row=(count - 1) * self.row_count + 1,
                    **kwargs,
                )
                if monthly_data.empty:
                    break
                final_df = pd.concat(
                    [final_df, monthly_data], axis=0, ignore_index=True
                )
                count += 1

        datetime_columns = ["bid_datetime_beginning_ept"]
        for col in datetime_columns:
            if col in final_df.columns:
                final_df[col] = pd.to_datetime(final_df[col])

        if self.end_date is not None:
            final_df = final_df[final_df["bid_datetime_beginning_ept"] <= self.end_date]
            final_df.reset_index(drop=True, inplace=True)
        logging.info("All data fetched.")
        return final_df


class PJMForecastedGenerationOutages(PJMHeaderSetUp):
    """A class to retrieve forecasted generation outages for the next ninetydays."""

    def __init__(
        self, start_datetime=None, end_datetime=None, is_range=True, row_count=50000
    ):
        """Initialize the PJMForecastedGenerationOutages.

        Args:
            start_datetime (datetime, optional): A datetime object of the start date. Defaults to None.
            end_datetime (datetime, optional): A datetime object of the end date; When included with a start_datetime, this datetime will be INCLUSIVE in the data retrieved. Defaults to None.
            is_range (bool, optional): Determines if the retrieved data is from a date range (True) or a static date (False). Defaults to True.
            row_count (int, optional): Number of rows to fetch per request. Defaults to 50000.
        """
        super().__init__()

        if start_datetime is not None:
            start_date = self.convert_datetime_string(start_datetime)
            if is_range:
                self.start_date = f"{start_date} to"
            else:
                self.start_date = start_date
        else:
            self.start_date = start_datetime
        self.end_date = end_datetime
        self.row_count = row_count
        self.base_url = "https://api.pjm.com/api/v1/frcstd_gen_outages"

    def convert_datetime_string(self, dt):
        """Convert datetime objects to string.

        Args:
            dt (datetime): A datetime object.

        Returns:
            str: A string converted from datetime object in format MM/DD/YYYY HH:MM:SS.
        """
        format_datetime = dt.strftime("%m/%d/%y %H:%M:%S")
        return format_datetime

    def get_forecasted_generation_outages(
        self, start_dte=None, start_row=1, row_count=None, download=True, **kwargs
    ):
        """Fetch a single batch of monthly FTR Zonal LMPs data.

        Args:
            start_dte (str, optional): The starting date to retrieve data. Defaults to None.
            start_row (int, optional): The starting row number for the batch to fetch. Defaults to 1.
            row_count (int, optional): The number of rows to fetch. Defaults to 50000.
            download (bool, optional): Determines how results should be returned (If True, no
        links or search criteria will be echoed). Defaults to True.

        Raises:
            requests.exceptions.RequestException: Request was unsucessful.

        Returns:
            pd.DataFrame: A dataframe of a single batch of forecasted generation outages for the next ninety days.
        """
        headers = self.header
        params = {
            "forecast_execution_date_ept": start_dte,
            "download": download,
            "rowCount": row_count if row_count is not None else self.row_count,
            "startRow": start_row,
        }
        params.update(kwargs)
        response = requests.get(self.base_url, headers=headers, params=params)
        if response.status_code == 200:
            data = response.json()
            return pd.DataFrame(data)
        else:
            raise requests.exceptions.RequestException(
                f"Failed to retrieve data from {self._settings_url}: {response.status_code} - {response.text}"
            )

    def save_data(self, output_file_path, file_name, **kwargs):
        """Saves forecasted generation outages for the next ninety days capacity data based on the initialized configuration into the provided output_file_path and file_name. This method fetches data across a range of months if `date_range` was specified during initialization,
        or for a single day if is_range is set False.

        Args:
            output_file_path (str): File path to save parquet file.
            file_name (str):  Name of the file.
        """
        if output_file_path[-1] == "/":
            file_path = f"{output_file_path}{file_name}.parquet"
        else:
            file_path = f"{output_file_path}/{file_name}.parquet"

        final_df = pd.DataFrame()

        empty = False
        count = 1
        logging.info("Checking data availability.")
        data_check = self.get_forecasted_generation_outages(
            start_dte=self.start_date, row_count=1, **kwargs
        )
        if data_check.empty:
            logging.warning("No data found.")
            final_df = pd.DataFrame()
        else:
            logging.info(
                f"Fetching Forecasted Generation Outages for the next Ninety Days from {self.start_date}."
            )
            while not empty:
                monthly_data = self.get_forecasted_generation_outages(
                    start_dte=self.start_date,
                    start_row=(count - 1) * self.row_count + 1,
                    **kwargs,
                )
                if monthly_data.empty:
                    break
                final_df = pd.concat(
                    [final_df, monthly_data], axis=0, ignore_index=True
                )
                count += 1

        datetime_columns = ["forecast_execution_date_ept", "forecast_date"]
        for col in datetime_columns:
            if col in final_df.columns:
                final_df[col] = pd.to_datetime(final_df[col])

        if self.end_date is not None:
            final_df = final_df[final_df["forecast_date"] <= self.end_date]
            final_df.reset_index(drop=True, inplace=True)
        final_df.to_parquet(file_path, index=False, engine="pyarrow")
        logging.info(f"Data saved to {file_path}")

    def fetch_data(self, **kwargs):
        """Fetches forecasted generation outages for the next ninety days capacity data based on the initialized configuration. This method fetches data across a range of months if `date_range` was specified during initialization,
        or for a single day if is_range is set False.

        Returns:
            pd.DataFrame: A dataframe containing forecasted generation outages for the defined parameters.
        """
        final_df = pd.DataFrame()

        empty = False
        count = 1
        logging.info("Checking data availability.")
        data_check = self.get_forecasted_generation_outages(
            start_dte=self.start_date, row_count=1, **kwargs
        )
        if data_check.empty:
            logging.warning("No data found.")
            final_df = pd.DataFrame()
        else:
            logging.info(
                f"Fetching Forecasted Generation Outages for the next Ninety Days from {self.start_date}."
            )
            while not empty:
                monthly_data = self.get_forecasted_generation_outages(
                    start_dte=self.start_date,
                    start_row=(count - 1) * self.row_count + 1,
                    **kwargs,
                )
                if monthly_data.empty:
                    break
                final_df = pd.concat(
                    [final_df, monthly_data], axis=0, ignore_index=True
                )
                count += 1

        datetime_columns = ["forecast_execution_date_ept", "forecast_date"]
        for col in datetime_columns:
            if col in final_df.columns:
                final_df[col] = pd.to_datetime(final_df[col])

        if self.end_date is not None:
            final_df = final_df[final_df["forecast_date"] <= self.end_date]
            final_df.reset_index(drop=True, inplace=True)
        logging.info("All data fetched.")
        return final_df


class PJMGenerationOutageForSevenDays(PJMHeaderSetUp):
    """A class to retrieve generation outages for the next seven days."""

    def __init__(
        self, start_datetime=None, end_datetime=None, is_range=True, row_count=50000
    ):
        """Initialize the PJMGenerationOutageForSevenDays.

        Args:
            start_datetime (datetime, optional): A datetime object of the start date. Defaults to None.
            end_datetime (datetime, optional): A datetime object of the end date; When included with a start_datetime, this datetime will be INCLUSIVE in the data retrieved. Defaults to None.
            is_range (bool, optional): Determines if the retrieved data is from a date range (True) or a static date (False). Defaults to True.
            row_count (int, optional): Number of rows to fetch per request. Defaults to 50000.
        """
        super().__init__()

        if start_datetime is not None:
            start_date = self.convert_datetime_string(start_datetime)
            if is_range:
                self.start_date = f"{start_date} to"
            else:
                self.start_date = start_date
        else:
            self.start_date = start_datetime
        self.end_date = end_datetime
        self.row_count = row_count
        self.base_url = "https://api.pjm.com/api/v1/gen_outages_by_type"

    def convert_datetime_string(self, dt):
        """Convert datetime objects to string.

        Args:
            dt (datetime): A datetime object.

        Returns:
            str: A string converted from datetime object in format MM/DD/YYYY HH:MM:SS.
        """
        format_datetime = dt.strftime("%m/%d/%y %H:%M:%S")
        return format_datetime

    def get_generation_outages_by_seven_days(
        self, start_dte=None, start_row=1, row_count=None, download=True, **kwargs
    ):
        """Fetch a single batch of monthly FTR Zonal LMPs data.

        Args:
            start_dte (str, optional): The starting date to retrieve data. Defaults to None.
            start_row (int, optional): The starting row number for the batch to fetch. Defaults to 1.
            row_count (int, optional): The number of rows to fetch. Defaults to 50000.
            download (bool, optional): Determines how results should be returned (If True, no
        links or search criteria will be echoed). Defaults to True.

        Raises:
            requests.exceptions.RequestException: Request was unsucessful.

        Returns:
            pd.DataFrame: A dataframe of a single batch of scheduled and unplanned generation outages.
        """
        headers = self.header
        params = {
            "forecast_execution_date_ept": start_dte,
            "download": download,
            "rowCount": row_count if row_count is not None else self.row_count,
            "startRow": start_row,
        }
        params.update(kwargs)
        response = requests.get(self.base_url, headers=headers, params=params)
        if response.status_code == 200:
            data = response.json()
            return pd.DataFrame(data)
        else:
            raise requests.exceptions.RequestException(
                f"Failed to retrieve data from {self._settings_url}: {response.status_code} - {response.text}"
            )

    def save_data(self, output_file_path, file_name, **kwargs):
        """Saves scheduled and unplanned generation outages data based on the initialized configuration into the provided output_file_path and file_name. This method fetches data across a range of months if `date_range` was specified during initialization,
        or for a single day if is_range is set False.

        Args:
            output_file_path (str): File path to save parquet file.
            file_name (str): Name of the file.
        """
        if output_file_path[-1] == "/":
            file_path = f"{output_file_path}{file_name}.parquet"
        else:
            file_path = f"{output_file_path}/{file_name}.parquet"

        final_df = pd.DataFrame()

        empty = False
        count = 1
        logging.info("Checking data availability.")
        data_check = self.get_generation_outages_by_seven_days(
            start_dte=self.start_date, row_count=1, **kwargs
        )
        if data_check.empty:
            logging.warning("No data found.")
            final_df = pd.DataFrame()
        else:
            logging.info(
                f"Fetching Generation Outages for the next Seven Days from {self.start_date}_{self.end_date}."
            )
            while not empty:
                monthly_data = self.get_generation_outages_by_seven_days(
                    start_dte=self.start_date,
                    start_row=(count - 1) * self.row_count + 1,
                    **kwargs,
                )
                if monthly_data.empty:
                    break
                final_df = pd.concat(
                    [final_df, monthly_data], axis=0, ignore_index=True
                )
                count += 1

        datetime_columns = ["forecast_execution_date_ept", "forecast_date"]
        for col in datetime_columns:
            if col in final_df.columns:
                final_df[col] = pd.to_datetime(final_df[col])

        if self.end_date is not None:
            final_df = final_df[final_df["forecast_date"] <= self.end_date]
            final_df.reset_index(drop=True, inplace=True)
        final_df.to_parquet(file_path, index=False, engine="pyarrow")
        logging.info(f"Data saved to {file_path}")

    def fetch_data(self, **kwargs):
        """Fetches scheduled and unplanned generation outages data based on the initialized configuration. This method fetches data across a range of months if `date_range` was specified during initialization,
        or for a single day if is_range is set False.

        Returns:
            pd.DataFrame: A dataframe containing monthly FTR Bids for the defined parameters.
        """
        final_df = pd.DataFrame()

        empty = False
        count = 1
        logging.info("Checking data availability.")
        data_check = self.get_generation_outages_by_seven_days(
            start_dte=self.start_date, row_count=1, **kwargs
        )
        if data_check.empty:
            logging.warning("No data found.")
            final_df = pd.DataFrame()
        else:
            logging.info(
                f"Fetching Generation Outages for the next Seven Days from {self.start_date}_{self.end_date}."
            )
            while not empty:
                monthly_data = self.get_generation_outages_by_seven_days(
                    start_dte=self.start_date,
                    start_row=(count - 1) * self.row_count + 1,
                    **kwargs,
                )
                if monthly_data.empty:
                    break
                final_df = pd.concat(
                    [final_df, monthly_data], axis=0, ignore_index=True
                )
                count += 1

        datetime_columns = ["forecast_execution_date_ept", "forecast_date"]
        for col in datetime_columns:
            if col in final_df.columns:
                final_df[col] = pd.to_datetime(final_df[col])

        if self.end_date is not None:
            final_df = final_df[final_df["forecast_date"] <= self.end_date]
            final_df.reset_index(drop=True, inplace=True)
        logging.info("All data fetched.")
        return final_df


class PJMPnodesSearch(PJMHeaderSetUp):
    """A class to retrieve master information on pnodes (pricing nodes)."""

    def __init__(self, row_count=50000):
        """Initialize PJMPnodesSearch

        Args:
            row_count (int, optional): Number of rows to fetch per request. Defaults to 50000.
        """
        super().__init__()
        self.row_count = row_count
        self.base_url = "https://api.pjm.com/api/v1/pnode"

    def get_pnode_data(self, start_row=1, row_count=None, download=True, **kwargs):
        """Fetch a single batch of pricing node data.

        Args:
            start_row (int, optional): The starting row number for the batch to fetch. Defaults to 1.
            row_count (int, optional): The number of rows to fetch. Defaults to None.
            download (bool, optional): Determines how results should be returned (If True, no
        links or search criteria will be echoed). Defaults to True.

        Raises:
            requests.exceptions.RequestException: Request was unsucessful.

        Returns:
            pd.DataFrame: A dataframe of a single batch of pricing node data.
        """
        headers = self.header
        params = {
            "download": download,
            "rowCount": row_count if row_count is not None else self.row_count,
            "startRow": start_row,
        }
        params.update(kwargs)
        response = requests.get(self.base_url, headers=headers, params=params)
        if response.status_code == 200:
            data = response.json()
            return pd.DataFrame(data)
        else:
            raise requests.exceptions.RequestException(
                f"Failed to retrieve data from {self._settings_url}: {response.status_code} - {response.text}"
            )

    def save_data(self, output_file_path, file_name, **kwargs):
        """Saves all master information on pnodes (pricing nodes) data into the provided output_file_path and file_name.

        Args:
            output_file_path (str): File path to save parquet file.
            file_name (str): Name of the file.
        """
        if output_file_path[-1] == "/":
            file_path = f"{output_file_path}{file_name}.parquet"
        else:
            file_path = f"{output_file_path}/{file_name}.parquet"

        empty = False
        count = 1
        logging.info("Checking data availability.")
        data_check = self.get_pnode_data(row_count=1, **kwargs)
        if data_check.empty:
            logging.warning("No data found.")
            final_df = pd.DataFrame()
        logging.info("Fetching pricing node data.")
        final_df = pd.DataFrame()
        while not empty:
            pnode_data = self.get_pnode_data(start_row=(count - 1) * self.row_count + 1)
            if pnode_data.empty:
                break
            final_df = pd.concat([final_df, pnode_data], axis=0, ignore_index=True)
            count += 1

        final_df["effective_date"] = pd.to_datetime(final_df["effective_date"])
        final_df.to_parquet(file_path, index=False, engine="pyarrow")
        logging.info(f"Data saved to {output_file_path}")

    def fetch_data(self, **kwargs):
        """Fetch all master information on pnodes (pricing nodes) data.

        Returns:
            pd.DataFrame: A dataframe containing master information on pricing nodes (pnodes).
        """
        empty = False
        count = 1
        logging.info("Checking data availability.")
        data_check = self.get_pnode_data(row_count=1, **kwargs)
        if data_check.empty:
            logging.warning("No data found.")
            final_df = pd.DataFrame()
        logging.info("Fetching pricing node data.")
        final_df = pd.DataFrame()
        while not empty:
            pnode_data = self.get_pnode_data(start_row=(count - 1) * self.row_count + 1)
            if pnode_data.empty:
                break
            final_df = pd.concat([final_df, pnode_data], axis=0, ignore_index=True)
            count += 1

        final_df["effective_date"] = pd.to_datetime(final_df["effective_date"])
        logging.info("All data fetched.")
        return final_df
