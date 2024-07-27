import sys
import os
import pandas as pd
import numpy as np

# Set the PYTHONPATH to include the library directory
sys.path.append(os.path.abspath('../'))



from aou_utils import Utils
from aou_utils.CohortGenerator import CohortGenerator

# Example usage
current_date = '2024-07-17'

# Seed for reproducibility
np.random.seed(42)

# Generate case data
case_data = {
    'person_id': range(1, 21),
    'sex_at_birth': np.random.choice(['Male', 'Female'], 20),
    'race_concept_id': np.random.choice([1, 2, 3], 20),
    'birth_datetime': pd.to_datetime(np.random.choice(pd.date_range('1980-01-01', '2000-12-31'), 20)),
    'age': np.random.randint(20, 60, 20),
    'smoker_status': np.random.choice(['Smoker', 'Non-smoker'], 20)
}
case_df = pd.DataFrame(case_data)

# Generate control data
control_data = {
    'person_id': range(21, 101),
    'sex_at_birth': np.random.choice(['Male', 'Female'], 80),
    'race_concept_id': np.random.choice([1, 2, 3], 80),
    'birth_datetime': pd.to_datetime(np.random.choice(pd.date_range('1980-01-01', '2000-12-31'), 80)),
    'age': np.random.randint(20, 60, 80),
    'smoker_status': np.random.choice(['Smoker', 'Non-smoker'], 80)
}
control_df = pd.DataFrame(control_data)

# Display the first few rows of the generated data
print("Case DataFrame:")
print(case_df.head())
print("\nControl DataFrame:")
print(control_df.head())

# Initialize the CohortGenerator class
cohort_generator = CohortGenerator(case_df, control_df)

# Add criteria and match cases to controls
matched_controls = cohort_generator.withAge(caliper=3).withSex().withRace().withSmokerStatus().match_cases_to_controls(ratio=2)

# Display the matched controls
print("\nMatched Controls DataFrame:")
print(matched_controls)