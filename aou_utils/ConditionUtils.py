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