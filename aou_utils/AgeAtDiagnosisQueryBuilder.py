class AgeAtDiagnosisQueryBuilder:
    def __init__(self, dataset):
        self.dataset = dataset
        self.condition_ids = []
        self.include_concepts = []
        self.exclude_concepts = []

    def withConditions(self, condition_concept_ids):
        """Add condition concept IDs."""
        self.condition_ids.extend(condition_concept_ids)
        return self

    def include_concept_ids(self, concept_ids):
        """Add inclusion concept IDs."""
        self.include_concepts.extend(concept_ids)
        return self

    def exclude_concept_ids(self, concept_ids):
        """Add exclusion concept IDs."""
        self.exclude_concepts.extend(concept_ids)
        return self

    def build_query(self):
        """Construct the query."""
        condition_ids_str = Utils.list_to_string(self.condition_ids, quote=False)
        include_query = (
            f"AND con.condition_concept_id IN ({Utils.list_to_string(self.include_concepts, quote=False)})"
            if self.include_concepts else ""
        )
        exclude_query = (
            f"AND con.condition_concept_id NOT IN ({Utils.list_to_string(self.exclude_concepts, quote=False)})"
            if self.exclude_concepts else ""
        )

        query = f"""
            WITH
              persons AS (
                SELECT
                  person_id,
                  birth_datetime,
                  concept_name AS sex_at_birth,
                  CAST(birth_datetime AS DATE) AS birth_date
                FROM
                  `{self.dataset}.person`
                LEFT JOIN `{self.dataset}.concept` ON concept_id = sex_at_birth_concept_id
              ),
              earliest_conditions AS (
                SELECT
                  person_id,
                  MIN(condition_start_date) AS earliest_condition_start_date
                FROM
                  `{self.dataset}.condition_occurrence` con
                LEFT JOIN `{self.dataset}.concept_ancestor` anc ON con.condition_concept_id = anc.descendant_concept_id
                WHERE
                  anc.ancestor_concept_id IN ({condition_ids_str})
                  {include_query}
                  {exclude_query}
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
                  `{self.dataset}.condition_occurrence` con
                LEFT JOIN `{self.dataset}.concept_ancestor` anc ON con.condition_concept_id = anc.descendant_concept_id
                JOIN
                  earliest_conditions ec ON con.person_id = ec.person_id
                WHERE
                  anc.ancestor_concept_id IN ({condition_ids_str})
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
