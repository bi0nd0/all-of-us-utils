import sys
import os
import pandas as pd

# Set the PYTHONPATH to include the library directory
sys.path.append(os.path.abspath('../'))


from aou_utils.StatisticsUtils import StatisticsUtils

# Example usage:
# Assuming df is your DataFrame with the required columns
df = pd.DataFrame({
    'person_id': [1, 2, 3, 4, 5],
    'sex_at_birth': ['Male', 'Female', 'Female', 'Male', 'Female'],
    'age': [25, 30, 22, 28, 35],
    'race': ['White', 'Black', 'Asian', 'White', 'Black'],
    'ethnicity': ['Non-Hispanic', 'Hispanic', 'Non-Hispanic', 'Hispanic', 'Non-Hispanic'],
    'smoker_status': ['Non-smoker', 'Smoker', 'Non-smoker', 'Smoker', 'Non-smoker']
})

stats_utils = StatisticsUtils()

# Calculate sex statistics
sex_stats = stats_utils.calculate_sex_statistics(df)
print("Sex Statistics:")
print(sex_stats)

# Calculate age statistics
age_mean, age_std = stats_utils.calculate_age_statistics(df)
print(f"Mean age: {age_mean}, Standard deviation of age: {age_std}")

# Calculate race and ethnicity statistics
race_ethnicity_stats = stats_utils.calculate_race_ethnicity_statistics(df)
print("Race and Ethnicity Statistics:")
print(race_ethnicity_stats)

# Calculate smoker status statistics
smoker_status_stats = stats_utils.calculate_smoker_status_statistics(df)
print("Smoker Status Statistics:")
print(smoker_status_stats)