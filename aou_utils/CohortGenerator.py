import pandas as pd
import numpy as np

import pandas as pd
import numpy as np

class CohortGenerator:
    AGE_DIFF_KEY = 'age_diff'

    def __init__(self, case_df: pd.DataFrame, control_df: pd.DataFrame):
        # Sort the DataFrames by person_id to ensure consistent processing order
        self.case_df = case_df.sort_values(by='person_id').reset_index(drop=True)
        self.control_df = control_df.sort_values(by='person_id').reset_index(drop=True)
        self.criteria = []

    def withAge(self, caliper: float = 5, age_key: str = "age"):
        def age_criteria(case_row, potential_matches):
            potential_matches[self.AGE_DIFF_KEY] = np.abs(potential_matches[age_key] - case_row[age_key])
            potential_matches = potential_matches[potential_matches[self.AGE_DIFF_KEY] <= caliper]
            return potential_matches.sort_values(self.AGE_DIFF_KEY)
        self.criteria.append(age_criteria)
        return self

    def withSex(self, sex_key: str = "sex_at_birth"):
        def sex_criteria(case_row, potential_matches):
            return potential_matches[potential_matches[sex_key] == case_row[sex_key]]
        self.criteria.append(sex_criteria)
        return self

    def withRace(self, race_key: str = "race_concept_id"):
        def race_criteria(case_row, potential_matches):
            return potential_matches[potential_matches[race_key] == case_row[race_key]]
        self.criteria.append(race_criteria)
        return self
    
    def withEthnicity(self, ethnicity_key: str = "ethnicity_concept_id"):
        def ethnicity_criteria(case_row, potential_matches):
            return potential_matches[potential_matches[ethnicity_key] == case_row[ethnicity_key]]
        self.criteria.append(ethnicity_criteria)
        return self

    def withSmokerStatus(self, smoker_key: str = "smoker_status"):
        def smoker_status_criteria(case_row, potential_matches):
            return potential_matches[potential_matches[smoker_key] == case_row[smoker_key]]
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
