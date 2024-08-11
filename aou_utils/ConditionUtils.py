import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, fisher_exact

class MedicationUtils:
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
          condition_occurrences AS (
            SELECT
              person_id,
              condition_occurrence_id,
              condition_concept_id,
              condition_start_date,
              condition_start_datetime,
              condition_end_date,
              condition_end_datetime,
              condition_type_concept_id,
              stop_reason,
              provider_id,
              visit_occurrence_id,
              ROW_NUMBER() OVER (PARTITION BY person_id, condition_concept_id
                                ORDER BY condition_start_date ASC,
                                          condition_start_datetime ASC,
                                          condition_occurrence_id ASC) AS occurrence_rank
            FROM
              `{dataset}.condition_occurrence`
            WHERE true
              AND condition_concept_id IN ({CONDITION_CONCEPT_IDS})
              AND person_id IN ({COHORT_QUERY})
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
          person_id,
          condition_occurrence_id;
      """
      return query

    