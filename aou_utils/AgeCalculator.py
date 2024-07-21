from typing import Optional
import pandas as pd

class AgeCalculator:
    def __init__(self, current_date: Optional[str] = None, dob_key: str = "date_of_birth", age_key: str = "age"):
        self.current_date = pd.to_datetime(current_date) if current_date else pd.to_datetime('today')
        self.dob_key = dob_key
        self.age_key = age_key

    def calculate_age(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculates age and adds it to the DataFrame."""
        df_copy = df.copy()
        df_copy[self.age_key] = df_copy[self.dob_key].apply(
            lambda dob: self.current_date.year - dob.year - ((self.current_date.month, self.current_date.day) < (dob.month, dob.day))
        )
        return df_copy
