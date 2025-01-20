import pandas as pd
import numpy as np

class DataCleaner:
    def __init__(self, filename, datapath, print_info=False):
        """
        Initialize the DataCleaner class.

        :param filename: Name of the CSV file to be cleaned.
        :param datapath: Path to the directory containing the CSV file.
        :param print_info: Boolean to control logging.
        """
        self.filename = filename
        self.datapath = datapath
        self.print_info = print_info
        self.df = pd.read_csv(self.datapath + self.filename)

    def log(self, message):
        """
        Logs messages if print_info is True.

        :param message: Message to log.
        """
        if self.print_info:
            print(message)

    def check_nan(self, final=False):
        """
        Checks for NaN values in the DataFrame and logs details.

        :param final: If True, raises an error if NaN values are found after cleaning.
        """
        if self.df.isnull().values.any():
            self.log("NaN values found in the DataFrame.")
            nan_count = self.df.isnull().sum()
            if final:
                raise ValueError(
                    f"Error: NaN values remain in the DataFrame after cleaning. \nNumber of NaN by column: \n{nan_count}"
                )
            else:
                self.log(f"Number of NaN by column: \n{nan_count}")
        else:
            self.log("No NaN values in the DataFrame.")

    @staticmethod
    def convert_min_to_float(x):
        """
        Converts time from 'MM:SS' format to a float representing total minutes.

        :param x: Time string in 'MM:SS' format.
        :return: Total minutes as a float or NaN if conversion fails.
        """
        try:
            minutes, seconds = map(int, x.split(':'))
            return round(minutes + seconds / 60, 2)
        except (ValueError, AttributeError):
            return np.nan

    @staticmethod
    def convert_teams(x):
        """
        Standardizes team names.

        :param x: Team name as a string.
        :return: Standardized team name.
        """
        team_mapping = {
            'Golden St': 'Golden State Warriors',
            'e Warriors': 'Golden State Warriors',
            'Miami H': 'Miami Heat',
            'Miami He': 'Miami Heat'
        }
        return team_mapping.get(x, x)

    def clean_data(self):
        """
        Cleans the DataFrame by handling NaN values and standardizing data formats.

        :return: Cleaned DataFrame.
        """
        self.check_nan()

        # Convert 'FG' column to numeric, coercing errors to NaN
        self.df['FG'] = pd.to_numeric(self.df['FG'], errors='coerce')

        # Convert 'MP' column (minutes played) to float
        self.df['MP'] = self.df['MP'].apply(self.convert_min_to_float)

        # Standardize team names
        self.df['Team'] = self.df['Team'].apply(self.convert_teams)
        self.df['Against'] = self.df['Against'].apply(self.convert_teams)

        # Fill NaN with 0
        self.df.fillna(0, inplace=True)

        # Check for NaN values after cleaning
        self.check_nan(final=True)

        # Save cleaned data to a parquet file
        cleaned_filename = self.filename.replace('.csv', '_cleaned.csv')
        output_path = self.datapath + cleaned_filename
        self.df.to_csv(output_path)
        self.log(f"Cleaned data saved to {output_path}")

        return self.df
