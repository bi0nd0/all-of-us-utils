class ConditionMedicationQueryBuilder:
    """
    A Builder class for creating a BigQuery query that:
      1) Identifies a cohort of patients with certain condition(s).
      2) Retrieves their person-level info.
      3) Retrieves their most-recent medication info.
    """
    def __init__(self, dataset: str):
        self.dataset = dataset
        self.condition_ids = []
        self.medication_ids = []

    def with_conditions(self, condition_ids: list[int]):
        """
        Specify a list of condition concept IDs (e.g. [316866, 4066824, etc.])
        to include in the 'cohort' logic.
        """
        self.condition_ids = condition_ids
        return self  # Return self to allow chaining

    def with_medications(self, medication_ids: list[int]):
        """
        Specify a list of medication concept IDs (e.g. [1901903, 1310149, ...])
        for the medication query filter.
        """
        self.medication_ids = medication_ids
        return self  # Return self to allow chaining

    def _build_condition_subquery(self, condition_id: int) -> str:
        """
        Returns a snippet of SQL that checks for the given condition_id.
        
        Note: This logic is modeled on your existing examples. 
        It pulls all person_ids who have the selected condition_id
        in the cb_search_all_events/cb_criteria tables.
        """
        return f"""
            SELECT criteria.person_id
            FROM (
                SELECT DISTINCT person_id, entry_date, concept_id
                FROM `{self.dataset}.cb_search_all_events`
                WHERE (
                  concept_id IN (
                    SELECT DISTINCT c.concept_id
                    FROM `{self.dataset}.cb_criteria` c
                    JOIN (
                        SELECT CAST(cr.id as STRING) AS id
                        FROM `{self.dataset}.cb_criteria` cr
                        WHERE concept_id IN ({condition_id})
                          AND full_text LIKE '%_rank1]%'
                    ) a
                      ON (
                           c.path LIKE CONCAT('%.', a.id, '.%')
                        OR c.path LIKE CONCAT('%.', a.id)
                        OR c.path LIKE CONCAT(a.id, '.%')
                        OR c.path = a.id
                      )
                    WHERE is_standard = 1
                      AND is_selectable = 1
                  )
                  AND is_standard = 1
                )
            ) criteria
        """

    def build(self) -> str:
      """
      Construct and return the final BigQuery SQL query.
      """
      # 1) Build up the "cohort AS (...)" with one subquery per condition
      #
      # We start with "SELECT DISTINCT person_id FROM cb_search_person" 
      # then for each condition, we do "AND person_id IN ( ... )"
      #
      # If no conditions were provided, you might want different logic 
      # (e.g. no WHERE conditions). For demonstration, we'll handle 
      # an empty condition list by returning everyone from cb_search_person.
      #
      # If you prefer, you can throw an error if condition_ids is empty. 
      # That is up to your business logic.
      
      cohort_conditions = ""
      if self.condition_ids:
          for cond_id in self.condition_ids:
              cohort_conditions += f"""
                AND cb_search_person.person_id IN (
                    {self._build_condition_subquery(cond_id)}
                )
              """
      
      cohort_cte = f"""
        cohort AS (
          SELECT DISTINCT person_id
          FROM `{self.dataset}.cb_search_person` cb_search_person
          WHERE 1=1
          {cohort_conditions}
        )
      """

      # 2) Build the persons CTE: minimal example pulling birth_datetime / sex_at_birth
      persons_cte = f"""
        persons AS (
          SELECT
            p.person_id,
            p.birth_datetime,
            c.concept_name AS sex_at_birth
          FROM `{self.dataset}.person` p
          LEFT JOIN `{self.dataset}.concept` c
            ON c.concept_id = p.sex_at_birth_concept_id
          WHERE p.person_id IN (SELECT person_id FROM cohort)
        )
      """

      # 3) Build the medications CTE: filter by medication IDs (if any)
      # If there are no medication_ids, we can either remove the filter or handle differently.
      # We'll assume the user always supplies at least one medication ID. If empty, it would cause
      # `drug_concept_id IN ()` which is invalid. So we skip or raise an error.
      medication_filter = ""
      if self.medication_ids:
          # e.g. "1901903, 1310149"
          medication_list_str = ", ".join(str(m) for m in self.medication_ids)
          medication_filter = f"AND drug_concept_id IN ({medication_list_str})"
      else:
          # If no medication_ids are specified, you might want to omit the filter,
          # or handle it as needed. We'll simply omit the filter entirely, returning all meds
          medication_filter = ""

      medications_cte = f"""
        medications AS (
          SELECT
            person_id,
            drug_exposure_id,
            drug_concept_id,
            drug_exposure_start_date,
            drug_exposure_end_date,
            drug_type_concept_id,
            stop_reason,
            refills,
            quantity,
            days_supply,
            sig,
            route_concept_id,
            lot_number,
            provider_id,
            visit_occurrence_id,
            ROW_NUMBER() OVER (
              PARTITION BY person_id, drug_concept_id
              ORDER BY drug_exposure_start_date DESC,
                      drug_exposure_end_date   DESC,
                      drug_exposure_id         DESC
            ) AS recency_rank
          FROM `{self.dataset}.drug_exposure`
          WHERE person_id IN (SELECT person_id FROM cohort)
            {medication_filter}
        )
      """

      # 4) Final SELECT
      final_select = f"""
          SELECT
            persons.*,
            ext.src_id,
            medications.* EXCEPT (person_id, drug_exposure_id, recency_rank)
          FROM medications
          LEFT JOIN persons USING (person_id)
          LEFT JOIN `{self.dataset}.drug_exposure_ext` ext USING (drug_exposure_id)
          WHERE recency_rank = 1
          ORDER BY person_id, drug_exposure_id
        """

          # Combine everything into one query with CTEs
      query = f"""
        -- Example single query with multiple CTEs
        WITH
        {cohort_cte},
        {persons_cte},
        {medications_cte}
        {final_select};
      """

      # Return the final SQL string
      return query.strip()
