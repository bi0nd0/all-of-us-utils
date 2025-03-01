import pandas as pd
from scipy.stats import chi2_contingency

class StatisticsUtils:
    
    @staticmethod
    def _format_count_percentage(count, percentage):
        """
        Helper method to format the count and percentage as 'count (percentage%)',
        with the percentage truncated to two decimal places.

        Parameters:
        count (int): The count value.
        percentage (float): The percentage value.

        Returns:
        str: A formatted string in the form 'count (percentage%)'.
        """
        formatted_percentage = f"{percentage:.2f}"
        return f"{count} ({formatted_percentage}%)"
    
    @staticmethod
    def calculate_sex_statistics(df):
        """
        Calculates the counts and percentages for each sex_at_birth in the given DataFrame,
        and adds a formatted column with 'count (percentage%)'.

        Parameters:
        df (pd.DataFrame): The DataFrame containing the data with columns 'person_id' and 'sex_at_birth'.

        Returns:
        pd.DataFrame: A DataFrame containing the counts, percentages, and formatted column for each sex_at_birth.
        """
        sex_counts = df.groupby('sex_at_birth')['person_id'].nunique().reset_index()
        sex_counts.columns = ['sex_at_birth', 'count']
        sex_counts['percentage'] = (sex_counts['count'] / sex_counts['count'].sum()) * 100
        sex_counts['formatted'] = sex_counts.apply(
            lambda row: StatisticsUtils._format_count_percentage(row['count'], row['percentage']), axis=1
        )
        return sex_counts

    @staticmethod
    def calculate_age_statistics(df):
        """
        Calculates the mean and standard deviation of the age column in the given DataFrame.

        Parameters:
        df (pd.DataFrame): The DataFrame containing the data with the column 'age'.

        Returns:
        tuple: A tuple containing the mean and standard deviation of the age column.
        """
        age_mean = df['age'].mean()
        age_std = df['age'].std()
        return age_mean, age_std

    @staticmethod
    def calculate_race_ethnicity_statistics(df):
        """
        Calculates the counts and percentages for race and ethnicity in the given DataFrame,
        and adds a formatted column with 'count (percentage%)'.

        Parameters:
        df (pd.DataFrame): The DataFrame containing the data with columns 'race' and 'ethnicity'.

        Returns:
        pd.DataFrame: A DataFrame containing the counts, percentages, and formatted column for each race and ethnicity.
        """
        df['race_ethnicity'] = df['race'] + ' ' + df['ethnicity']
        counts = df['race_ethnicity'].value_counts().reset_index()
        counts.columns = ['race_ethnicity', 'count']
        counts['percentage'] = (counts['count'] / counts['count'].sum()) * 100
        counts['formatted'] = counts.apply(
            lambda row: StatisticsUtils._format_count_percentage(row['count'], row['percentage']), axis=1
        )
        return counts

    @staticmethod
    def calculate_smoker_status_statistics(df):
        """
        Calculates the counts and percentages for smoker status in the given DataFrame,
        and adds a formatted column with 'count (percentage%)'.

        Parameters:
        df (pd.DataFrame): The DataFrame containing the data with the column 'smoker_status'.

        Returns:
        pd.DataFrame: A DataFrame containing the counts, percentages, and formatted column for each smoker status.
        """
        smoker_status_counts = df['smoker_status'].value_counts().reset_index()
        smoker_status_counts.columns = ['smoker_status', 'count']
        smoker_status_counts['percentage'] = (smoker_status_counts['count'] / smoker_status_counts['count'].sum()) * 100
        smoker_status_counts['formatted'] = smoker_status_counts.apply(
            lambda row: StatisticsUtils._format_count_percentage(row['count'], row['percentage']), axis=1
        )
        return smoker_status_counts
    
    @staticmethod
    def calculate_survey_statistics(df, total=None):
        """
        Groups the DataFrame by `answer_concept_id`, and calculates the count and percentage for each group.
        The resulting DataFrame is ordered by `answer_concept_id` in ascending order.
        Adds a formatted column with 'count (percentage%)'.

        Parameters:
        df (pd.DataFrame): The DataFrame containing the columns 'answer_concept_id' and 'answer'.
        total (int, optional): The total number of responses to use for calculating percentages. 
                            If None, the sum of counts in the DataFrame is used.

        Returns:
        pd.DataFrame: A DataFrame grouped by 'answer_concept_id' with columns 'answer', 'count', 'percentage',
                    and a formatted column, ordered by `answer_concept_id` in ascending order.
        """
        # Group by 'answer_concept_id' and count the occurrences
        grouped_df = df.groupby(['answer_concept_id', 'answer']).size().reset_index(name='count')

        # Use the provided total if availa

    @staticmethod
    def calculate_median(df: pd.DataFrame, column_name: str):
        """
        Calculate the median for a given column in a DataFrame.
        The data is converted to numeric (non-numeric values become NaN) and NaN values are dropped.
        
        Parameters:
            df (pd.DataFrame): The input DataFrame.
            column_name (str): The name of the column to calculate the median on.
            
        Returns:
            float: The median of the column.
        """
        # Convert the column to numeric, coercing errors (non-numeric values become NaN)
        numeric_data = pd.to_numeric(df[column_name], errors='coerce')
        # Drop NaN values
        clean_data = numeric_data.dropna()
        # Return the median
        return clean_data.median()

    @staticmethod
    def calculate_std(df: pd.DataFrame, column_name: str):
        """
        Calculate the standard deviation for a given column in a DataFrame.
        The data is converted to numeric (non-numeric values become NaN) and NaN values are dropped.
        
        Parameters:
            df (pd.DataFrame): The input DataFrame.
            column_name (str): The name of the column to calculate the standard deviation on.
            
        Returns:
            float: The standard deviation of the column.
        """
        # Convert the column to numeric, coercing errors (non-numeric values become NaN)
        numeric_data = pd.to_numeric(df[column_name], errors='coerce')
        # Drop NaN values
        clean_data = numeric_data.dropna()
        # Return the standard deviation
        return clean_data.std()
    
    @staticmethod
    def count_flagged_items(df: pd.DataFrame, keys: list) -> dict:
        """
        Counts the number of rows in the DataFrame that have at least one of the specified flag columns set to 1,
        and calculates the percentage of such rows relative to the total number of rows.
        
        Parameters:
        - df (pd.DataFrame): The input DataFrame.
        - keys (list): A list of column names corresponding to the flag columns.
        
        Returns:
        - dict: A dictionary with two keys:
            'count': The total number of rows with at least one flag set to 1.
            'percentage': The percentage of such rows with respect to the total number of rows in the DataFrame.
            
        Raises:
        - KeyError: If any of the keys are not found in the DataFrame's columns.
        """
        # Check if all keys are present in the DataFrame columns
        missing_keys = [key for key in keys if key not in df.columns]
        if missing_keys:
            raise KeyError(f"The following keys are not in the DataFrame columns: {missing_keys}")

        # Create a boolean mask where at least one flag column equals 1
        mask = df[keys].any(axis=1)
        
        # Count the number of rows where the condition is True
        count = mask.sum()
        
        # Calculate the percentage relative to the total number of rows
        percentage = (count / len(df)) * 100 if len(df) > 0 else 0
        
        return {"count": count, "percentage": percentage}