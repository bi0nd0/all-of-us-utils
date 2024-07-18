import sys
import os
import pandas as pd
import numpy as np

# Set the PYTHONPATH to include the library directory
sys.path.append(os.path.abspath('../'))



from aou_utils import Utils
from aou_utils.CohortGenerator import CohortGenerator
from aou_utils.AgeCalculator import AgeCalculator

# Example usage
current_date = '2024-07-17'

# Seed for reproducibility
np.random.seed(42)

# Generate case data
case_data = {
    'person_id': range(1, 21),
    'sex_at_birth': np.random.choice(['M', 'F'], 20),
    'race_concept_id': np.random.choice([1, 2, 3], 20),
    'birth_datetime': pd.to_datetime(np.random.choice(pd.date_range('1980-01-01', '2000-12-31'), 20))
}

# Generate control data
control_data = {
    'person_id': range(21, 121),
    'sex_at_birth': np.random.choice(['M', 'F'], 100),
    'race_concept_id': np.random.choice([1, 2, 3], 100),
    'birth_datetime': pd.to_datetime(np.random.choice(pd.date_range('1980-01-01', '2000-12-31'), 100))
}

# Generate survey data for case
case_survey_data = {
    'person_id': range(1, 21),
    'question_concept_id': [1585860] * 20,
    'answer': np.random.choice(['Not At All', 'Daily', 'Occasionally'], 20)
}

# Generate survey data for control
control_survey_data = {
    'person_id': range(21, 121),
    'question_concept_id': [1585860] * 100,
    'answer': np.random.choice(['Not At All', 'Daily', 'Occasionally'], 100)
}

case_df = pd.DataFrame(case_data)
control_df = pd.DataFrame(control_data)
case_survey_df = pd.DataFrame(case_survey_data)
control_survey_df = pd.DataFrame(control_survey_data)

# Initialize AgeCalculator
age_calculator = AgeCalculator(current_date=current_date)

# Initialize and run the cohort generator
cohort_gen = CohortGenerator(case_df, control_df, case_survey_df, control_survey_df, age_calculator)

matched_controls = cohort_gen.match_cases_to_controls(ratio=2, caliper=3)

print(matched_controls)