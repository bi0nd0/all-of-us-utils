import pandas as pd
import numpy as np
from .AgeCalculator import AgeCalculator

class CohortGenerator:
    SEX_KEY = "sex_at_birth"
    RACE_KEY = "race_concept_id"
    AGE_KEY = "age"
    AGE_DIFF_KEY = 'age_diff'
    SMOKER_KEY = "smoker_status"  # New key for smoker status

    def __init__(self, case_df: pd.DataFrame, control_df: pd.DataFrame):
        self.case_df = case_df.copy()
        self.control_df = control_df.copy()

        # # Debug: Print DataFrame columns to verify smoker_status column
        # print("Case DataFrame after applying smoker status and age:")
        # print(self.case_df)
        # print("\nControl DataFrame after applying smoker status and age:")
        # print(self.control_df)


    def find_matches(self, case_row: pd.Series, ratio: int, caliper: int) -> pd.DataFrame:
        """Finds matching controls for a given case based on sex, race, age within a specified caliper, and smoker status."""
        # Filter potential matches by sex, race, and smoker status
        potential_matches = self.control_df[
            (self.control_df[self.SEX_KEY] == case_row[self.SEX_KEY]) &
            (self.control_df[self.RACE_KEY] == case_row[self.RACE_KEY]) &
            (self.control_df[self.SMOKER_KEY] == case_row[self.SMOKER_KEY])
        ].copy()  # Make a copy to avoid warnings when setting with enlargement on a slice.
        
        # Calculate age difference and apply a caliper (e.g., 3 years)
        potential_matches[self.AGE_DIFF_KEY] = np.abs(potential_matches[self.AGE_KEY] - case_row[self.AGE_KEY])
        potential_matches = potential_matches[potential_matches[self.AGE_DIFF_KEY] <= caliper]
        
        # Sort by the smallest age difference and select up to the specified ratio of matches
        matched_controls = potential_matches.sort_values(self.AGE_DIFF_KEY).head(ratio)

        # Add a column for the matched case person_id
        matched_controls['matched_case_id'] = case_row['person_id']
        
        return matched_controls

    def match_cases_to_controls(self, ratio: int = 4, caliper: int = 3) -> pd.DataFrame:
        """Processes each case in case_df to find matching controls in control_df."""
        matched_controls_df = pd.DataFrame()  # Local DataFrame to store matched controls
        
        for index, case_row in self.case_df.iterrows():
            matches = self.find_matches(case_row, ratio, caliper)
            matched_controls_df = pd.concat([matched_controls_df, matches])

            # Remove matched controls from control_df to avoid reusing controls
            self.control_df = self.control_df.drop(matches.index)

        return matched_controls_df