import pandas as pd

class SmokerStatusUtility:
    @staticmethod
    def add_smoker_status(survey_df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds a 'smoker_status' column to the DataFrame based on specified conditions.
        1585860 = smoke frequency
        1585873 = smoking; number of years

        Parameters:
        survey_df (pd.DataFrame): DataFrame containing 'question_concept_id', 'answer', and 'answer_concept_id' columns.

        Returns:
        pd.DataFrame: DataFrame with the additional 'smoker_status' column.
        """
        df_copy = survey_df.copy()

        def determine_smoker_status(row):
            try:
                if row['question_concept_id'] == 1585860 and row['answer_concept_id'] in (1585861, 1585862, 1585863):
                    return 'smoker'
                elif row['question_concept_id'] == 1585873 and int(row['answer']) > 0:
                    return 'smoker'
            except ValueError:
                pass
            return 'non-smoker'

        df_copy['smoker_status'] = df_copy.apply(determine_smoker_status, axis=1)
        return df_copy

    @staticmethod
    def apply_smoker_status(main_df: pd.DataFrame, survey_df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds a smoker/non-smoker field to the DataFrame based on the provided survey data.

        Parameters:
        main_df (pd.DataFrame): The main DataFrame to which the smoker status will be added.
        survey_df (pd.DataFrame): The survey DataFrame containing the answers to determine smoker status.

        Returns:
        pd.DataFrame: The main DataFrame with the added smoker status.
        """
        # Add smoker status to survey_df using the add_smoker_status method
        survey_df_with_status = SmokerStatusUtility.add_smoker_status(survey_df)

        # Keep only the relevant columns for merging
        smoker_status_df = survey_df_with_status[['person_id', 'smoker_status']]

        # Merge the smoker status into the main DataFrame
        merged_df = main_df.merge(smoker_status_df, on='person_id', how='left')
        merged_df['smoker_status'] = merged_df['smoker_status'].fillna('NA')

        return merged_df
