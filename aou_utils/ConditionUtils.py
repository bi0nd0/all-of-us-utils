import pandas as pd
from scipy.stats import fisher_exact
import statsmodels.api as sm
from aou_utils.PValueUtils import PValueUtils
class ConditionUtils:
    def __init__(self, dataset):
        self.dataset = dataset

    def getAgeAtDiagnosis(self, person_ids, condition_concept_ids):
      dataset = self.dataset
      CONDITION_CONCEPT_IDS = ", ".join(map(str, condition_concept_ids))
      COHORT_QUERY = ", ".join(map(str, person_ids))
      query = f"""
          WITH
            persons AS (
              SELECT
                person_id,
                birth_datetime,
                concept_name AS sex_at_birth,
                CAST(birth_datetime AS DATE) AS birth_date
              FROM
                `{dataset}.person`
              LEFT JOIN `{dataset}.concept` ON concept_id = sex_at_birth_concept_id
            ),
            earliest_conditions AS (
              SELECT
                person_id,
                MIN(condition_start_date) AS earliest_condition_start_date
              FROM
                `{dataset}.condition_occurrence` con
              LEFT JOIN `{dataset}.concept_ancestor` anc ON con.condition_concept_id = anc.descendant_concept_id
              WHERE
                anc.ancestor_concept_id IN ({CONDITION_CONCEPT_IDS})
                AND con.person_id IN ({COHORT_QUERY})
              GROUP BY person_id
            ),
            condition_occurrences AS (
              SELECT
                con.person_id,
                con.condition_occurrence_id,
                con.condition_concept_id,
                con.condition_start_date,
                con.condition_start_datetime,
                con.condition_end_date,
                con.condition_end_datetime,
                con.condition_type_concept_id,
                con.stop_reason,
                con.provider_id,
                con.visit_occurrence_id,
                ROW_NUMBER() OVER (PARTITION BY con.person_id
                                  ORDER BY con.condition_start_date ASC,
                                            con.condition_start_datetime ASC,
                                            con.condition_occurrence_id ASC) AS occurrence_rank
              FROM
                `{dataset}.condition_occurrence` con
              LEFT JOIN `{dataset}.concept_ancestor` anc ON con.condition_concept_id = anc.descendant_concept_id
              JOIN
                earliest_conditions ec ON con.person_id = ec.person_id
              WHERE
                anc.ancestor_concept_id IN ({CONDITION_CONCEPT_IDS})
                AND con.person_id IN ({COHORT_QUERY})
                AND con.condition_start_date = ec.earliest_condition_start_date
            )
          SELECT
            persons.*,
            condition_occurrences.* EXCEPT(person_id, condition_occurrence_id),
            DATE_DIFF(CAST(condition_occurrences.condition_start_date AS DATE), persons.birth_date, YEAR) AS age_at_diagnosis
          FROM
            condition_occurrences
          LEFT JOIN
            persons USING (person_id)
          WHERE
            occurrence_rank = 1
          ORDER BY
            person_id;
        """
      return query
    
    @staticmethod
    def add_condition_column(person_df, condition_df, label):
      """
      Adds a new column to a copy of person_df based on matches with condition_df.
      
      Parameters:
      - person_df (pd.DataFrame): The initial DataFrame containing all persons.
      - condition_df (pd.DataFrame): The DataFrame containing persons with a specific condition.
      - label (str): The name of the new column to be added to person_df.
      
      Returns:
      - pd.DataFrame: A copy of person_df with the new column, where 1 indicates a match with condition_df and 0 otherwise.
      """
      # Make a copy of person_df to avoid altering the original DataFrame
      result_df = person_df.copy()
      
      # Initialize the new column with 0 for all rows
      result_df[label] = 0
      
      # Get the list of person_ids that have the condition
      condition_person_ids = condition_df['person_id'].unique()
      
      # Update the new column to 1 for matching person_ids
      result_df.loc[result_df['person_id'].isin(condition_person_ids), label] = 1
      
      return result_df
    
    def calculate_statistics(study_df, control_df, conditions):
      """
      Generates a summary table comparing the prevalence of specific conditions between
      a study group and a control group. For each condition, the method calculates the 
      total count and percentage of individuals with the condition in both groups, 
      computes the odds ratio (OR) with its 95% confidence interval (CI), and determines 
      the statistical significance (P-value) of the difference in prevalence between 
      the groups. The results are returned as a DataFrame structured similarly to 
      summary tables used in clinical research.

      Parameters:
      - study_df (pd.DataFrame): DataFrame containing the study group data, where columns 
        represent specific conditions and values are binary (1 for presence, 0 for absence).
      - control_df (pd.DataFrame): DataFrame containing the control group data, structured 
        similarly to study_df.
      - conditions (list of str): List of column names corresponding to the conditions to 
        be analyzed.

      Returns:
      - pd.DataFrame: A DataFrame containing the condition name, counts and percentages 
        for the study and control groups, odds ratios with 95% confidence intervals, 
        and P-values for each condition.
      """

      results = []
      
      total_study = len(study_df)
      total_control = len(control_df)
      
      for condition in conditions:
          # Calculate counts and percentages
          study_count = study_df[condition].sum()
          control_count = control_df[condition].sum()
          
          study_percentage = (study_count / total_study) * 100
          control_percentage = (control_count / total_control) * 100
          
          # Contingency table for Fisher's exact test
          contingency_table = [
              [study_count, total_study - study_count],
              [control_count, total_control - control_count]
          ]
          
          # Calculate OR, CI, and P-value
          oddsratio, p_value = fisher_exact(contingency_table)
          ci_low, ci_high = sm.stats.Table2x2(contingency_table).oddsratio_confint()
          
          # Append the results for this condition
          results.append({
              'Condition': condition,
              'Study Group (N)': f"{study_count} ({study_percentage:.1f})",
              'Control Group (N)': f"{control_count} ({control_percentage:.1f})",
              'OR (95% CI)': f"{oddsratio:.2f} ({ci_low:.2f}-{ci_high:.2f})",
              'P-value': PValueUtils.format_p_value(p_value)
          })
      
      # Convert results to a DataFrame
      results_df = pd.DataFrame(results)
      
      return results_df
    
