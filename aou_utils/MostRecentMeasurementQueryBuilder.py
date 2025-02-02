import pandas as pd

class MostRecentMeasurementQueryBuilder:
    def __init__(self, dataset: str):
        """
        Initialize the QueryBuilder with the dataset name.
        
        Parameters:
        - dataset (str): The name of the dataset used in the query.
        """
        self.dataset = dataset
        self.measurement_concept_id = None  # Required
        self.unit_concept_id = None         # Optional
        self.person_ids = None              # Optional

    def withMeasurementConceptId(self, measurement_concept_id: int):
        """
        Set the measurement_concept_id parameter.
        
        Parameters:
        - measurement_concept_id (int): The measurement concept id (e.g., 3004410 for Hemoglobin A1c).
        
        Returns:
        - mostRecentMeasurementQueryBuilder: The builder instance.
        """
        self.measurement_concept_id = measurement_concept_id
        return self

    def withUnitConceptId(self, unit_concept_id: int):
        """
        Set the unit_concept_id parameter.
        
        Parameters:
        - unit_concept_id (int): The unit concept id (e.g., 8554 for percent).
        
        Returns:
        - mostRecentMeasurementQueryBuilder: The builder instance.
        """
        self.unit_concept_id = unit_concept_id
        return self

    def withPersonIds(self, person_ids: list):
        """
        Set the person IDs for the query. The list is converted to a comma-separated string.
        
        Parameters:
        - person_ids (list): List of person_id values.
        
        Returns:
        - mostRecentMeasurementQueryBuilder: The builder instance.
        """
        self.person_ids = ", ".join(str(pid) for pid in person_ids)
        return self

    def build(self) -> str:
        """
        Build and return the final query string.
        
        Returns:
        - str: The complete SQL query.
        
        Raises:
        - ValueError: If the required measurement_concept_id has not been set.
        """
        if self.measurement_concept_id is None:
            raise ValueError("measurement_concept_id is not set. Please use withMeasurementConceptId().")

        # Build the dynamic WHERE clause for the measurements query
        where_clauses = [f"measurement_concept_id = {self.measurement_concept_id}"]
        if self.unit_concept_id is not None:
            where_clauses.append(f"unit_concept_id = {self.unit_concept_id}")
        if self.person_ids is not None:
            where_clauses.append(f"person_id IN ({self.person_ids})")
        
        # Join the conditions with AND
        where_clause = " AND ".join(where_clauses)
        
        query = f"""
            -- Return row level data for a measurement, limited to only the most recent result per person in our cohort.
            --
            -- PARAMETERS:
            --   MEASUREMENT_CONCEPT_ID: for example 3004410        # Hemoglobin A1c
            --   UNIT_CONCEPT_ID: for example 8554                  # percent (optional)
            --   PERSON_IDS: list of person IDs to filter by         (optional)

            WITH
            --
            -- Retrieve participants birthdate and sex_at_birth.
            --
            persons AS (
            SELECT
                person_id,
                birth_datetime,
                concept_name AS sex_at_birth
            FROM
                `{self.dataset}.person`
            LEFT JOIN `{self.dataset}.concept` ON concept_id = sex_at_birth_concept_id
            ),
            --
            -- Retrieve the row-level data for our measurement of interest. Also compute
            -- a new column for the recency rank of the measurement per person, a rank of
            -- 1 being the most recent lab result for that person.
            --
            measurements AS (
            SELECT
                person_id,
                measurement_id,
                measurement_concept_id,
                unit_concept_id,
                measurement_date,
                measurement_datetime,
                measurement_type_concept_id,
                operator_concept_id,
                value_as_number,
                value_as_concept_id,
                range_low,
                range_high,
                ROW_NUMBER() OVER (
                PARTITION BY person_id
                ORDER BY measurement_date DESC,
                        measurement_datetime DESC,
                        measurement_id DESC
                ) AS recency_rank
            FROM
                `{self.dataset}.measurement`
            WHERE
                {where_clause}
            )
            --
            -- Lastly, JOIN all this data together so that we have the birthdate, sex_at_birth and site for each
            -- measurement, retaining only the most recent result per person.
            --
            SELECT
            persons.*,
            src_id,
            measurements.* EXCEPT(person_id, measurement_id, recency_rank)
            FROM
            measurements
            LEFT JOIN
            persons USING (person_id)
            LEFT JOIN
            `{self.dataset}.measurement_ext` USING (measurement_id)
            WHERE
            recency_rank = 1
            ORDER BY
            person_id,
            measurement_id
            """
        return query

    @staticmethod
    def applyMeasurement(patient_df: pd.DataFrame, measurement_df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies measurement data to the patient DataFrame by merging with the measurement DataFrame.
        Assumes measurement_df has 'person_id' and 'value_as_number' columns.
        The 'value_as_number' column is renamed to 'bmi' after merging.
        
        Parameters:
        - patient_df (pd.DataFrame): DataFrame containing patient data with a 'person_id' column.
        - measurement_df (pd.DataFrame): DataFrame containing measurement data with 'person_id' and 'value_as_number'.
        
        Returns:
        - pd.DataFrame: The merged DataFrame with an added 'bmi' column.
        """
        merged_df = patient_df.merge(
            measurement_df[['person_id', 'value_as_number']],
            on='person_id',
            how='left'
        )
        merged_df.rename(columns={'value_as_number': 'bmi'}, inplace=True)
        return merged_df

