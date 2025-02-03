# Example usage of PValueCalculator classes

# Create sample study and control DataFrames.
import numpy as np
import pandas as pd

np.random.seed(42)

# Study group: 50 participants.
study_df = pd.DataFrame({
    'age': np.random.normal(60, 5, 50),
    'sex_at_birth': np.random.choice(['M', 'F'], 50),
    'race_concept_id': np.random.choice(['Race1', 'Race2', 'Race3'], 50),
    'smoker_status': np.random.choice(['Yes', 'No'], 50),
    'count': np.random.randint(1, 10, 50)  # Used for specialized analyses.
})

# Control group: 100 participants.
control_df = pd.DataFrame({
    'age': np.random.normal(62, 6, 100),
    'sex_at_birth': np.random.choice(['M', 'F'], 100),
    'race_concept_id': np.random.choice(['Race1', 'Race2', 'Race3'], 100),
    'smoker_status': np.random.choice(['Yes', 'No'], 100),
    'count': np.random.randint(1, 10, 100)
})

# Instantiate calculators for generic analyses.
age_calculator = ContinuousPValueCalculator(
    study_df, control_df, 'age', label='Age (years)'
)

sex_calculator = CategoricalPValueCalculator(
    study_df, control_df, 'sex_at_birth', label='Sex at Birth'
)

race_calculator = CategoricalPValueCalculator(
    study_df, control_df, 'race_concept_id', label='Race'
)

smoker_calculator = CategoricalPValueCalculator(
    study_df, control_df, 'smoker_status', label='Ever Smoker'
)

# Instantiate specialized calculators.
sex_special_calculator = SexAtBirthPValueCalculator(
    study_df, control_df, label='Sex at Birth (Specialized)'
)

survey_calculator = SurveyPValueCalculator(
    study_df, control_df, label='Survey Analysis'
)

# Aggregate all calculators using the aggregator.
aggregator = PValueAggregator()
aggregator.add_calculator(age_calculator)\
          .add_calculator(sex_calculator)\
          .add_calculator(race_calculator)\
          .add_calculator(smoker_calculator)\
          .add_calculator(sex_special_calculator)\
          .add_calculator(survey_calculator)

# Calculate all p-values and display the summary DataFrame.
final_results = aggregator.calculate_all()
print(final_results)
