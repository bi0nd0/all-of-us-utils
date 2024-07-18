import pandas as pd

class AgeCalculator:
    @staticmethod
    def calculate_age(df, current_date, dob_key="date_of_birth", age_key="age"):
        # Create a copy of the DataFrame to avoid modifying the original
        today = pd.to_datetime('today')
        df_copy = df.copy()
        df_copy['age'] = df_copy[dob_key].apply(lambda dob: today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day)))

        return df_copy