import pandas as pd

class AgeCalculator:
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