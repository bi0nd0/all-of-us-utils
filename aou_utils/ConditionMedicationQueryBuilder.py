class ConditionMedicationQueryBuilder:
    """
    A Builder class for creating a BigQuery query that:
      1) Identifies a cohort of patients based on one or more 'blocks' of condition IDs,
         using the original subquery logic referencing cb_search_all_events + cb_criteria.
         - Each block uses OR logic among its IDs via "concept_id IN (...)".
         - Multiple blocks are ANDed together in the final 'cohort' CTE.
      2) Retrieves their person-level info (CTE 'persons').
      3) Retrieves their most-recent medication info (CTE 'medications').
    """

    def __init__(self, dataset: str):
        self.dataset = dataset

        # We'll store each "block" as a list of condition IDs.
        # For example:
        #   with_conditions([c1, c2]) => block #1
        #   and_with_conditions([c3, c4]) => block #2
        # In the final query, block #1 and block #2 are ANDed, while
        # within each block it's concept_id IN (c1, c2...) => OR logic.
        self.condition_blocks = []

        # We'll store medication concept IDs in a list as well.
        self.medication_ids = []

    def with_conditions(self, condition_ids: list[int]):
        """
        Create (or replace) the list of blocks, starting with the given condition_ids
        as the first block. If your design wants to *append* to any existing blocks
        instead of clearing them, adjust accordingly.
        
        This block will produce a single subquery with `concept_id IN (c1, c2, ...)`.
        """
        self.condition_blocks = []
        self.condition_blocks.append(condition_ids)
        return self

    def and_with_conditions(self, condition_ids: list[int]):
        """
        Append another block of condition IDs. This results in an additional
        `AND cb_search_person.person_id IN ( subquery_for_these_ids )` line
        in the final query. 
        """
        self.condition_blocks.append(condition_ids)
        return self

    def with_medications(self, medication_ids: list[int]):
        """
        Specify a list of medication concept IDs for the medication query filter
        in drug_exposure.
        """
        self.medication_ids = medication_ids
        return self

    def _build_condition_block_subquery(self, condition_ids: list[int]) -> str:
        """
        Return the subquery snippet (wrapped in the original logic) for a block
        of condition IDs, using your 'rank1' pattern from cb_criteria.
        
        The difference is:
          WHERE concept_id IN ({c1, c2, ...})  (OR among c1, c2, etc.)
        rather than concept_id IN ({single_id}).
        """

        # e.g., "316866, 4066824"
        condition_list_str = ", ".join(str(cid) for cid in condition_ids)

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
                        WHERE concept_id IN ({condition_list_str})
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

        # 1) Build the "cohort" CTE, iterating over each block
        #    Each block yields an AND subquery.
        cohort_conditions = ""
        for block in self.condition_blocks:
            sub_query = self._build_condition_block_subquery(block)
            cohort_conditions += f"""
              AND cb_search_person.person_id IN (
                  {sub_query}
              )
            """

        cohort_cte = f"""
          cohort AS (
            SELECT DISTINCT cb_search_person.person_id
            FROM `{self.dataset}.cb_search_person` cb_search_person
            WHERE 1=1
            {cohort_conditions}
          )
        """

        # 2) Persons CTE
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

        # 3) Medications CTE: Filter by medication_ids if given
        medication_filter = ""
        if self.medication_ids:
            meds_str = ", ".join(str(m) for m in self.medication_ids)
            medication_filter = f"AND drug_concept_id IN ({meds_str})"

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

        # Combine everything:
        query = f"""
            -- Example single query with multiple CTEs
            WITH
            {cohort_cte},
            {persons_cte},
            {medications_cte}
            {final_select};
        """

        return query.strip()
