import re, pandas as pd
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