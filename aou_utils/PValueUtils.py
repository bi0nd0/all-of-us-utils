import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, chi2_contingency, normaltest, ranksums, fisher_exact

class PValueUtils:
    @staticmethod
    def calculate_p_value_from_totals(variable_name, study_total, control_total, study_n, control_n):
        """
        Calculate the P-value for comparing data between study and control groups using the chi-square test or Fisher's exact test.

        Parameters:
        variable_name (str): Name of the variable being tested.
        study_total (int): Total number of cases in the study group.
        control_total (int): Total number of cases in the control group.
        study_n (int): Total number of participants in the study group.
        control_n (int): Total number of participants in the control group.

        Returns:
        pd.DataFrame: A DataFrame containing the P-value for the specified variable.
        """
        try:
            # Construct the contingency table
            contingency_table = np.array([
                [study_total, study_n - study_total],
                [control_total, control_n - control_total]
            ])
            print(f"Contingency table:\n{contingency_table}\n")

            # Perform Chi-Square Test
            chi2, p_value, dof, expected = chi2_contingency(contingency_table, correction=False)
            print(f"Chi-square test results: chi2 = {chi2}, p-value = {p_value}, dof = {dof}")
            print("Expected frequencies:\n", expected)

            # If expected frequencies are too low, use Fisher's Exact Test
            if (expected < 5).any():
                odds_ratio, p_value = fisher_exact(contingency_table)
                print(f"Fisher's exact test results: odds_ratio = {odds_ratio}, p-value = {p_value}")
            else:
                print("Chi-square test is reliable.")

            # Convert result to DataFrame
            result = pd.DataFrame([{'Variable': variable_name, 'P-value': p_value}])
            result['P-value'] = result['P-value'].map('{:.2e}'.format)
            return result

        except Exception as e:
            # Handle exceptions and return a valid structure with "N/A" for P-value
            print(f"Error calculating p-value for {variable_name}: {str(e)}")
            result = pd.DataFrame([{'Variable': variable_name, 'P-value': 'N/A'}])
            return result



    @staticmethod
    def format_p_value(p, threshold=0.001):
        """
        Format a single P-value to display values like "<.001" if below the threshold.

        Parameters:
        p (float): The P-value to format.
        threshold (float): The threshold below which P-values are formatted as "<threshold".

        Returns:
        str: The formatted P-value as a string.
        """
        if p < threshold:
            return f"<{threshold}"
        else:
            return f"{p:.3f}"
        
    @staticmethod
    def format_p_values(df, p_value_column='P-value', threshold=0.001):
        """
        Format P-values in the DataFrame to display values like "<.001" if below the threshold.

        Parameters:
        df (pd.DataFrame): The DataFrame containing P-values.
        p_value_column (str): The name of the column containing P-values.
        threshold (float): The threshold below which P-values are formatted as "<threshold".

        Returns:
        pd.DataFrame: A DataFrame with nicely formatted P-values.
        """
        formatted_df = df.copy()

        # Ensure all P-values are floats
        formatted_df[p_value_column] = formatted_df[p_value_column].astype(float)

        # Apply formatting
        formatted_df[p_value_column] = formatted_df[p_value_column].apply(
            lambda p: PValueUtils.format_p_value(p, threshold)
        )

        return formatted_df

    @staticmethod
    def calculate_p_value_continuous(group1_df, group2_df, column_name):
        """
        Calculate the P-value for a continuous variable using the Wilcoxon rank-sum test.

        Parameters:
        group1_df (pd.DataFrame): DataFrame for the first group.
        group2_df (pd.DataFrame): DataFrame for the second group.
        column_name (str): The column name of the continuous variable.

        Returns:
        float: The P-value.
        """
        try:
            group1_values = group1_df[column_name]
            group2_values = group2_df[column_name]

            # Perform Wilcoxon rank-sum test
            _, p_value = ranksums(group1_values, group2_values)

            print(f"Wilcoxon rank-sum test for {column_name}: p-value = {p_value}\n")
            return p_value

        except Exception as e:
            print(f"Error calculating p-value for {column_name}: {str(e)}")
            return None

    @staticmethod
    def calculate_p_value_categorical(group1_df, group2_df, column_name):
        """
        Calculate the P-value for a categorical variable using the chi-square test.

        Parameters:
        group1_df (pd.DataFrame): DataFrame for the first group.
        group2_df (pd.DataFrame): DataFrame for the second group.
        column_name (str): The column name of the categorical variable.

        Returns:
        float: The P-value.
        """
        try:
            # Create contingency table
            contingency_table = pd.DataFrame({
                'Group 1': group1_df[column_name].value_counts(),
                'Group 2': group2_df[column_name].value_counts()
            })

            print(f"Contingency table for {column_name}:\n{contingency_table}\n")
            chi2, p_value, dof, expected = chi2_contingency(contingency_table, correction=False)

            # Check for expected frequencies < 5
            if (expected < 5).any():
                print("Warning: some expected frequencies are less than 5. Chi-square test may not be reliable.")

            print(f"Chi-square test for {column_name}: chi2 = {chi2}, p-value = {p_value}, dof = {dof}\n")
            print(f"Expected frequencies for {column_name}:\n{expected}\n")
            return p_value

        except Exception as e:
            print(f"Error calculating p-value for {column_name}: {str(e)}")
            return None

    @staticmethod
    def calculate_study_p_values(study_group_df, control_group_df):
        """
        Calculate P-values for age, sex, ethnicity, and ever-smoker between study and control groups.

        Parameters:
        study_group_df (pd.DataFrame): DataFrame for the study group.
        control_group_df (pd.DataFrame): DataFrame for the control group.

        Returns:
        pd.DataFrame: A DataFrame containing P-values for each variable.
        """
        results = []

        # Age (continuous variable)
        age_p_value = PValueUtils.calculate_p_value_continuous(study_group_df, control_group_df, 'age')
        results.append({'Variable': 'Age', 'P-value': age_p_value})

        # Sex (categorical variable)
        sex_p_value = PValueUtils.calculate_p_value_categorical(study_group_df, control_group_df, 'sex_at_birth')
        results.append({'Variable': 'Sex', 'P-value': sex_p_value})

        # Ethnicity (categorical variable)
        ethnicity_p_value = PValueUtils.calculate_p_value_categorical(study_group_df, control_group_df, 'race_concept_id')
        results.append({'Variable': 'Race', 'P-value': ethnicity_p_value})

        # Ever Smoker (categorical variable)
        ever_smoker_p_value = PValueUtils.calculate_p_value_categorical(study_group_df, control_group_df, 'smoker_status')
        results.append({'Variable': 'Ever Smoker', 'P-value': ever_smoker_p_value})

        # Convert results to DataFrame
        p_values_df = pd.DataFrame(results)
        return p_values_df
    
    @staticmethod
    def calculate_sex_at_birth_p_values(study_df, control_df, study_total_participants, control_total_participants):
        """
        Calculate P-values for sex at birth categories between study and control groups using the chi-square test or Fisher's exact test.

        Parameters:
        study_df (pd.DataFrame): DataFrame for the study group with columns 'sex_at_birth' and 'count'.
        control_df (pd.DataFrame): DataFrame for the control group with columns 'sex_at_birth' and 'count'.
        study_total_participants (int): Total number of participants in the study group.
        control_total_participants (int): Total number of participants in the control group.

        Returns:
        pd.DataFrame: A DataFrame containing P-values for each sex at birth category.
        """
        try:
            results = []

            for _, row in study_df.iterrows():
                sex = row['sex_at_birth']
                study_count = row['count']
                control_count = control_df.loc[control_df['sex_at_birth'] == sex, 'count'].values[0]

                contingency_table = np.array([
                    [study_count, study_total_participants - study_count],
                    [control_count, control_total_participants - control_count]
                ])

                # Perform Chi-Square Test
                chi2, p_value, dof, expected = chi2_contingency(contingency_table, correction=False)

                # If expected frequencies are too low, use Fisher's Exact Test
                if (expected < 5).any():
                    odds_ratio, p_value = fisher_exact(contingency_table)
                    print(f"Fisher's exact test used for {sex} due to low expected frequencies.")
                else:
                    print(f"Chi-square test used for {sex}.")

                results.append({
                    'sex_at_birth': sex,
                    'study_count': study_count,
                    'control_count': control_count,
                    'P-value': p_value
                })
            
            p_values_df = pd.DataFrame(results)
            p_values_df['P-value'] = p_values_df['P-value'].map('{:.2e}'.format)
            return p_values_df
        
        except Exception as e:
            print(f"Error calculating p-values: {str(e)}")
            return None

    @staticmethod
    def calculate_survey_p_value(study_df, control_df):
        """
        Calculates the p-value comparing the distributions of answers between the study group
        and the control group using the Chi-Square test.

        Parameters:
        study_df (pd.DataFrame): DataFrame containing the survey results for the study group.
        control_df (pd.DataFrame): DataFrame containing the survey results for the control group.

        Returns:
        float: The p-value from the Chi-Square test.
        """
        # Merge the two DataFrames on 'answer_concept_id' and 'answer'
        merged_df = pd.merge(study_df[['answer_concept_id', 'answer', 'count']],
                             control_df[['answer_concept_id', 'answer', 'count']],
                             on=['answer_concept_id', 'answer'],
                             suffixes=('_study', '_control'))

        # Create a contingency table
        contingency_table = merged_df[['count_study', 'count_control']].values

        # Perform the Chi-Square test
        chi2, p, dof, expected = chi2_contingency(contingency_table)

        return p