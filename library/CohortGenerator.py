import pandas as pd
import numpy as np

class CohortGenerator:
    SEX_KEY = "sex_at_birth"
    RACE_KEY = "race_concept_id"
    AGE_KEY = "age"
    AGE_DIFF_KEY = 'age_diff'
    
    def calculate_age(self, date_of_birth, current_year):
        """
        Calculate the age based on the date of birth and the current year.

        Parameters:
        - date_of_birth (pandas Series): Series containing dates of birth.
        - current_year (int): The current year.

        Returns:
        - pandas Series: Series containing ages.
        """
        return current_year - date_of_birth.dt.year

    def find_matches(self, case, controls, ratio):
        """Finds matching controls for a given case based on sex, race, and age within a specified caliper."""
        # Filter potential matches by sex and race
        potential_matches = controls[
            (controls[self.SEX_KEY] == case[self.SEX_KEY]) & (controls[self.RACE_KEY] == case[self.RACE_KEY])
        ].copy()  # Make a copy to avoid warnings when setting with enlargement on a slice.
        
        
        # Calculate age difference and apply a caliper (e.g., 3 years)
        potential_matches[self.AGE_DIFF_KEY] = np.abs(potential_matches[self.AGE_KEY] - case[self.AGE_KEY])
        potential_matches = potential_matches[potential_matches[self.AGE_DIFF_KEY] <= 3]
        
        # Sort by the smallest age difference and select up to max 4 matches
        matched_controls = potential_matches.sort_values(self.AGE_DIFF_KEY).head(ratio)
        return matched_controls

    def match_cases_to_controls(self, person_df, others_df, ratio=4, caliper=3):        
        """Processes each case in person_df to find matching controls in others_df."""
        matched_controls_df = pd.DataFrame()  # Local DataFrame to store matched controls
        
        for index, case_row in person_df.iterrows():
            matches = self.find_matches(case_row, others_df, ratio)
            matched_controls_df = pd.concat([matched_controls_df, matches])

            # Remove matched controls from others_df to avoid reusing controls
            others_df = others_df.drop(matches.index)

        return matched_controls_df