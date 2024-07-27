import sys
import os
import pandas as pd
import numpy as np

# Set the PYTHONPATH to include the library directory
sys.path.append(os.path.abspath('../'))



from aou_utils import Utils
from aou_utils.PValueUtils import PValueUtils
from aou_utils.StatisticsUtils import StatisticsUtils


# Example usage:
np.random.seed(42)

# Example data for participants
study_participants = pd.DataFrame({
    'person_id': np.random.randint(1000, 2000, 50),
    'sex_at_birth': np.random.choice(['Male', 'Female'], 50)
})

control_participants = pd.DataFrame({
    'person_id': np.random.randint(2000, 3000, 50),
    'sex_at_birth': np.random.choice(['Male', 'Female'], 50)
})

# Calculate sex statistics
study_sex_stats = StatisticsUtils.calculate_sex_statistics(study_participants)
control_sex_stats = StatisticsUtils.calculate_sex_statistics(control_participants)

# Total number of participants
study_total_participants = study_participants['person_id'].nunique()
control_total_participants = control_participants['person_id'].nunique()

# Calculate P-values
sex_p_values_df = PValueUtils.calculate_sex_at_birth_p_values(
    study_sex_stats,
    control_sex_stats,
    study_total_participants,
    control_total_participants
)

print("P-values DataFrame:")
print(sex_p_values_df)