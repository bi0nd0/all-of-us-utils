import pandas as pd
import numpy as np

class CohortGenerator:
    SEX_KEY = "sex_at_birth"
    RACE_KEY = "race_concept_id"
    AGE_KEY = "age"
    AGE_DIFF_KEY = 'age_diff'
    SMOKER_KEY = "smoker_status"

    def __init__(self, case_df: pd.DataFrame, control_df: pd.DataFrame):
        # Sort the DataFrames by person_id to ensure consistent processing order
        self.case_df = case_df.sort_values(by='person_id').reset_index(drop=True)
        self.control_df = control_df.sort_values(by='person_id').reset_index(drop=True)
        self.criteria = []

    def withAge(self, caliper):
        def age_criteria(case_row, potential_matches):
            potential_matches[self.AGE_DIFF_KEY] = np.abs(potential_matches[self.AGE_KEY] - case_row[self.AGE_KEY])
            potential_matches = potential_matches[potential_matches[self.AGE_DIFF_KEY] <= caliper]
            return potential_matches.sort_values(self.AGE_DIFF_KEY)
        self.criteria.append(age_criteria)
        return self

    def withSex(self):
        def sex_criteria(case_row, potential_matches):
            return potential_matches[potential_matches[self.SEX_KEY] == case_row[self.SEX_KEY]]
        self.criteria.append(sex_criteria)
        return self

    def withRace(self):
        def race_criteria(case_row, potential_matches):
            return potential_matches[potential_matches[self.RACE_KEY] == case_row[self.RACE_KEY]]
        self.criteria.append(race_criteria)
        return self

    def withSmokerStatus(self):
        def smoker_status_criteria(case_row, potential_matches):
            return potential_matches[potential_matches[self.SMOKER_KEY] == case_row[self.SMOKER_KEY]]
        self.criteria.append(smoker_status_criteria)
        return self

    def find_matches(self, case_row: pd.Series, ratio: int) -> pd.DataFrame:
        """Finds matching controls for a given case based on dynamic criteria."""
        potential_matches = self.control_df.copy()

        for criterion in self.criteria:
            potential_matches = criterion(case_row, potential_matches)

        matched_controls = potential_matches.head(ratio).copy()  # Ensure it's a copy, not a view

        # Add a column for the matched case person_id
        matched_controls['matched_case_id'] = case_row['person_id']
        
        return matched_controls

    def match_cases_to_controls(self, ratio: int = 4) -> pd.DataFrame:
        """Processes each case in case_df to find matching controls in control_df."""
        matched_controls_df = pd.DataFrame()  # Local DataFrame to store matched controls
        
        for index, case_row in self.case_df.iterrows():
            matches = self.find_matches(case_row, ratio)
            matched_controls_df = pd.concat([matched_controls_df, matches])

            # Remove matched controls from control_df to avoid reusing controls
            self.control_df = self.control_df.drop(matches.index)

        return matched_controls_df
