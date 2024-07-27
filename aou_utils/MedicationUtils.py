import pandas as pd

class MedicationUtils:
    def __init__(self, medications):
        """
        Initialize the MedicationUtils class with a list of medications.

        Parameters:
        medications (list of dict): A list of medication dictionaries containing 'label', 'vocabulary', 'code', and 'id'.
        """
        self.medications = medications
        self.meds_list = [med['id'] for med in medications]
        self.drug_label_mapping = {med['id']: med['label'] for med in medications}

    def get_most_recent_drugs_query(self, person_ids, dataset):
        """
        Generate a SQL query to retrieve the most recent drug exposures for the specified person IDs.

        Parameters:
        person_ids (list of int): A list of person IDs to include in the query.
        dataset (str): The dataset name.

        Returns:
        str: The generated SQL query.
        """
        drug_concept_ids = self.meds_list
        DRUG_CONCEPT_ID = ", ".join(map(str, drug_concept_ids))
        COHORT_QUERY = ", ".join(map(str, person_ids))
        query = f"""
        WITH
          persons AS (
            SELECT
              person_id,
              birth_datetime,
              concept_name AS sex_at_birth
            FROM
              `{dataset}.person`
            LEFT JOIN `{dataset}.concept` ON concept_id = sex_at_birth_concept_id
          ),
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
              ROW_NUMBER() OVER (PARTITION BY person_id, drug_concept_id
                                 ORDER BY drug_exposure_start_date DESC,
                                          drug_exposure_end_date DESC,
                                          drug_exposure_id DESC) AS recency_rank
            FROM
              `{dataset}.drug_exposure`
            WHERE
              drug_concept_id IN ({DRUG_CONCEPT_ID})
              AND person_id IN ({COHORT_QUERY})
          )
        SELECT
          persons.*,
          src_id,
          medications.* EXCEPT(person_id, drug_exposure_id, recency_rank)
        FROM
          medications
        LEFT JOIN
          persons USING (person_id)
        LEFT JOIN
          `{dataset}.drug_exposure_ext` USING (drug_exposure_id)
        WHERE
          recency_rank = 1
        ORDER BY
          person_id,
          drug_exposure_id;
        """
        return query

    def calculate_medication_usage(self, person_df, med_df):
        """
        Calculate the medication usage percentage for each drug concept ID.

        Parameters:
        person_df (pd.DataFrame): A DataFrame containing person information.
        med_df (pd.DataFrame): A DataFrame containing medication exposure information.

        Returns:
        pd.DataFrame: A DataFrame with medication usage statistics.
        """
        # Merge the two dataframes on person_id
        merged_df = pd.merge(person_df, med_df, on='person_id', how='left')
        
        # Group the merged dataframe by medication ID and count the number of participants
        med_df_counts = merged_df.groupby('drug_concept_id').size().reset_index(name='med_df_count')
        
        # Group the merged dataframe by medication ID and count the total number of participants
        total_counts = merged_df.groupby('drug_concept_id')['person_id'].count().reset_index(name='total')
        
        # Calculate the total number of participants in the person_df group
        total_participants = person_df.shape[0]
        
        # Merge the two dataframes on medication ID
        merged_df = pd.merge(med_df_counts, total_counts, on='drug_concept_id')
        
        # Calculate the percentage of usage in the person_df group
        merged_df['percentage'] = (merged_df['total'] / total_participants) * 100
        
        # Map the medication IDs to labels
        merged_df['label'] = merged_df['drug_concept_id'].map(self.drug_label_mapping)
        
        # Select the desired columns
        results_df = merged_df[['drug_concept_id', 'label', 'total', 'percentage']]
        
        return results_df