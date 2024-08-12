import pandas as pd
from scipy.stats import chi2_contingency

class StatisticsUtils:
    
    @staticmethod
    def calculate_sex_statistics(df):
        """
        Calculates the counts and percentages for each sex_at_birth in the given DataFrame.

        Parameters:
        df (pd.DataFrame): The DataFrame containing the data with columns 'person_id' and 'sex_at_birth'.

        Returns:
        pd.DataFrame: A DataFrame containing the counts and percentages for each sex_at_birth.
        """
        sex_counts = df.groupby('sex_at_birth')['person_id'].nunique().reset_index()
        sex_counts.columns = ['sex_at_birth', 'count']
        sex_counts['percentage'] = (sex_counts['count'] / sex_counts['count'].sum()) * 100
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
        Calculates the counts and percentages for race and ethnicity in the given DataFrame.

        Parameters:
        df (pd.DataFrame): The DataFrame containing the data with columns 'race' and 'ethnicity'.

        Returns:
        pd.DataFrame: A DataFrame containing the counts and percentages for each race and ethnicity.
        """
        df['race_ethnicity'] = df['race'] + ' ' + df['ethnicity']
        counts = df['race_ethnicity'].value_counts().reset_index()
        counts.columns = ['race_ethnicity', 'count']
        counts['percentage'] = (counts['count'] / counts['count'].sum()) * 100
        return counts

    @staticmethod
    def calculate_smoker_status_statistics(df):
        """
        Calculates the counts and percentages for smoker status in the given DataFrame.

        Parameters:
        df (pd.DataFrame): The DataFrame containing the data with the column 'smoker_status'.

        Returns:
        pd.DataFrame: A DataFrame containing the counts and percentages for each smoker status.
        """
        smoker_status_counts = df['smoker_status'].value_counts().reset_index()
        smoker_status_counts.columns = ['smoker_status', 'count']
        smoker_status_counts['percentage'] = (smoker_status_counts['count'] / smoker_status_counts['count'].sum()) * 100
        return smoker_status_counts
    
    @staticmethod
    def calculate_survey_statistics(df):
        """
        Groups the DataFrame by `answer_concept_id`, and calculates the count and percentage for each group.
        The resulting DataFrame is ordered by `answer_concept_id` in ascending order.

        Parameters:
        df (pd.DataFrame): The DataFrame containing the columns 'answer_concept_id' and 'answer'.

        Returns:
        pd.DataFrame: A DataFrame grouped by 'answer_concept_id' with columns 'answer', 'count', and 'percentage',
                    ordered by `answer_concept_id` in ascending order.
        """
        # Group by 'answer_concept_id' and count the occurrences
        grouped_df = df.groupby(['answer_concept_id', 'answer']).size().reset_index(name='count')

        # Calculate the percentage for each group
        total_count = grouped_df['count'].sum()
        grouped_df['percentage'] = (grouped_df['count'] / total_count) * 100

        # Order by 'answer_concept_id' in ascending order
        grouped_df = grouped_df.sort_values(by='answer_concept_id', ascending=True).reset_index(drop=True)

        return grouped_df
    