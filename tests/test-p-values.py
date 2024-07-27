import sys
import os
import pandas as pd
import numpy as np

# Set the PYTHONPATH to include the library directory
sys.path.append(os.path.abspath('../'))



from aou_utils import Utils
from aou_utils.PValueUtils import PValueUtils

# Example usage:
# Assuming study_group_df and control_group_df are your DataFrames for the study and control groups
study_group_df = pd.DataFrame({
    'person_id': range(1, 21),
    'sex_at_birth': np.random.choice(['Male', 'Female'], 20),
    'ethnicity': np.random.choice(['Hispanic', 'Non-Hispanic'], 20),
    'age': np.random.randint(20, 60, 20),
    'smoker_status': np.random.choice(['Smoker', 'Non-smoker'], 20)
})

control_group_df = pd.DataFrame({
    'person_id': range(21, 101),
    'sex_at_birth': np.random.choice(['Male', 'Female'], 80),
    'ethnicity': np.random.choice(['Hispanic', 'Non-Hispanic'], 80),
    'age': np.random.randint(20, 60, 80),
    'smoker_status': np.random.choice(['Smoker', 'Non-smoker'], 80)
})

# Calculate study P-values
p_values_df = PValueUtils.calculate_study_p_values(study_group_df, control_group_df)
print("P-values DataFrame:")
print(p_values_df)

# Format P-values
formatted_p_values_df = PValueUtils.format_p_values(p_values_df)
print("\nFormatted P-values DataFrame:")
print(formatted_p_values_df)