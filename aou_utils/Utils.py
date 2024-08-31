import re, pandas as pd, os
import numpy as np
class Utils:
    @staticmethod
    def list_to_string(items, quote=False):
        if quote:
            return ",".join([f"'{item}'" for item in items])
        return ",".join(map(str, items))

    @staticmethod
    def sanitize_label(label):
        # Replace any non-alphanumeric characters with underscores
        return re.sub(r'\W+', '_', label)

    @staticmethod
    def pivot_results(df, var_name='label', value_name='total'):
        # Pivot the table to have two columns: label and total
        df_melted = df.melt(var_name=var_name, value_name=value_name)
        return df_melted
    
    @staticmethod
    def get_dataframe(sql_query):
        return pd.read_gbq(
        sql_query,
        dialect="standard",
        use_bqstorage_api=("BIGQUERY_STORAGE_API_ENABLED" in os.environ),
        progress_bar_type="tqdm_notebook")
    
    @classmethod
    def get_pivot_dataframe(cls, sql_query):
        # get rotated results from a query
        df = cls.get_dataframe(sql_query)
        return cls.pivot_results(df)
    
    @staticmethod
    def calculate_age(df, current_date, dob_key="date_of_birth", age_key="age"):
        def calculate_single_age(birthdate):
            age = current_date.year - birthdate.year - ((current_date.month, current_date.day) < (birthdate.month, birthdate.day))
            return age

        # Create a copy of the DataFrame to avoid modifying the original
        df_copy = df.copy()

        # Ensure the date_of_birth column in the copied DataFrame is a datetime type
        df_copy[dob_key] = pd.to_datetime(df_copy[dob_key], errors='coerce')

        # Apply the age calculation to each row in the copied DataFrame
        df_copy[age_key] = df_copy[dob_key].apply(calculate_single_age)

        return df_copy

    @staticmethod
    def count_column_values(df, total_participants, count_column, label_column=None):
        """
        Counts the total number of occurrences of each unique value in count_column, optionally labeling the counts 
        with the corresponding values from label_column. Also calculates the percentage of each count based on the 
        provided total number of participants. The percentage is reported fully in one column and truncated to two 
        decimal places in the formatted column.

        Parameters:
        - df (pd.DataFrame): The input DataFrame.
        - total_participants (int): The total number of participants to calculate percentages.
        - count_column (str): The column name in df to count the occurrences of.
        - label_column (str, optional): An optional column name in df to use as labels for the counts.
        
        Returns:
        - pd.DataFrame: A DataFrame with four columns: 'label', 'total', 'percentage', and 'formatted_total_percentage'.
        'label' contains either the values from label_column (if provided) or count_column, 'total' contains the corresponding counts,
        'percentage' contains the full percentages, and 'formatted_total_percentage' contains the truncated 
        'total (percentage)' strings.
        """
        if label_column:
            # Group by count_column and label_column, count occurrences, and reset index
            result_df = df.groupby([count_column, label_column]).size().reset_index(name='total')
            # Use label_column as the label column
            result_df = result_df[[label_column, 'total']].rename(columns={label_column: 'label'})
        else:
            # Count occurrences of each unique value in count_column
            result_df = df[count_column].value_counts().reset_index()
            result_df.columns = ['label', 'total']
        
        # Calculate percentage using the total_participants provided
        result_df['percentage'] = (result_df['total'] / total_participants) * 100
        
        # Truncate the percentage for formatting, without altering the original percentage
        truncated_percentages = result_df['percentage'].apply(lambda x: np.floor(x * 100) / 100.0)
        
        # Create the formatted 'total (percentage)' column with truncated percentages
        result_df['formatted_total_percentage'] = result_df['total'].astype(str) + ' (' + truncated_percentages.round(2).astype(str) + '%)'
        
        return result_df