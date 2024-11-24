import re, pandas as pd, os
import numpy as np
from scipy.stats import chi2_contingency, fisher_exact
class Utils:
    @staticmethod
    def list_to_string(items, quote=False):
        if quote:
            return ",".join([f"'{item}'" for item in items])
        return ",".join(map(str, items))

    @staticmethod
    def sanitize_label(label):
        # Replace any non-alphanumeric characters with underscores
        return re.sub(r'\W+', '_', label)

    @staticmethod
    def pivot_results(df, var_name='label', value_name='total'):
        # Pivot the table to have two columns: label and total
        df_melted = df.melt(var_name=var_name, value_name=value_name)
        return df_melted
    
    @staticmethod
    def get_dataframe(sql_query):
        return pd.read_gbq(
        sql_query,
        dialect="standard",
        use_bqstorage_api=("BIGQUERY_STORAGE_API_ENABLED" in os.environ),
        progress_bar_type="tqdm_notebook")
    
    @classmethod
    def get_pivot_dataframe(cls, sql_query):
        # get rotated results from a query
        df = cls.get_dataframe(sql_query)
        return cls.pivot_results(df)
    
    @staticmethod
    def calculate_age(df, current_date, dob_key="date_of_birth", age_key="age"):
        def calculate_single_age(birthdate):
            age = current_date.year - birthdate.year - ((current_date.month, current_date.day) < (birthdate.month, birthdate.day))
            return age

        # Create a copy of the DataFrame to avoid modifying the original
        df_copy = df.copy()

        # Ensure the date_of_birth column in the copied DataFrame is a datetime type
        df_copy[dob_key] = pd.to_datetime(df_copy[dob_key], errors='coerce')

        # Apply the age calculation to each row in the copied DataFrame
        df_copy[age_key] = df_copy[dob_key].apply(calculate_single_age)

        return df_copy

    @staticmethod
    def count_column_values(df, total_participants, count_column, label_column=None):
        """
        Counts the total number of occurrences of each unique value in count_column, optionally labeling the counts 
        with the corresponding values from label_column. Also calculates the percentage of each count based on the 
        provided total number of participants. The percentage is reported fully in one column and truncated to two 
        decimal places in the formatted column.

        Parameters:
        - df (pd.DataFrame): The input DataFrame.
        - total_participants (int): The total number of participants to calculate percentages.
        - count_column (str): The column name in df to count the occurrences of.
        - label_column (str, optional): An optional column name in df to use as labels for the counts.
        
        Returns:
        - pd.DataFrame: A DataFrame with four columns: 'label', 'total', 'percentage', and 'formatted_total_percentage'.
        'label' contains either the values from label_column (if provided) or count_column, 'total' contains the corresponding counts,
        'percentage' contains the full percentages, and 'formatted_total_percentage' contains the truncated 
        'total (percentage)' strings.
        """
        if label_column:
            # Group by count_column and label_column, count occurrences, and reset index
            result_df = df.groupby([count_column, label_column]).size().reset_index(name='total')
            # Use label_column as the label column
            result_df = result_df[[label_column, 'total']].rename(columns={label_column: 'label'})
        else:
            # Count occurrences of each unique value in count_column
            result_df = df[count_column].value_counts().reset_index()
            result_df.columns = ['label', 'total']
        
        # Calculate percentage using the total_participants provided
        result_df['percentage'] = (result_df['total'] / total_participants) * 100
        
        # Truncate the percentage for formatting, without altering the original percentage
        truncated_percentages = result_df['percentage'].apply(lambda x: np.floor(x * 100) / 100.0)
        
        # Create the formatted 'total (percentage)' column with truncated percentages
        result_df['formatted_total_percentage'] = result_df['total'].astype(str) + ' (' + truncated_percentages.round(2).astype(str) + '%)'
        
        return result_df
    
    @staticmethod
    def calculate_p_values(study_df, control_df, label_column='label', total_column='total'):
        """
        Calculates the P-value for the difference between study and control groups 
        for each label using a Chi-square test or Fisher's exact test.

        Parameters:
        - study_df (pd.DataFrame): A DataFrame containing the study group data.
        - control_df (pd.DataFrame): A DataFrame containing the control group data.
        - label_column (str): The column name that contains the labels (e.g., 'label'). Default is 'label'.
        - total_column (str): The column name that contains the totals (e.g., 'total'). Default is 'total'.

        Returns:
        - pd.DataFrame: A DataFrame containing the labels and their corresponding P-values.
        """
        p_values = []

        # Ensure both DataFrames are sorted by label
        study_df = study_df.sort_values(label_column).reset_index(drop=True)
        control_df = control_df.sort_values(label_column).reset_index(drop=True)

        for label in study_df[label_column]:
            # Check if the label exists in both DataFrames
            if label not in control_df[label_column].values:
                print(f"Label '{label}' not found in control group.")
                continue
            
            # Extract the totals for the study and control groups
            study_total = study_df[study_df[label_column] == label][total_column].values[0]
            control_total = control_df[control_df[label_column] == label][total_column].values[0]

            # Create the contingency table
            contingency_table = [[study_total, control_total], 
                                [sum(study_df[total_column]) - study_total, sum(control_df[total_column]) - control_total]]

            # Use Fisher's exact test if any cell in the table is less than 5
            if any(cell < 5 for cell in contingency_table[0]):
                _, p = fisher_exact(contingency_table)
            else:
                _, p, _, _ = chi2_contingency(contingency_table)

            # Store the label and the P-value
            p_values.append({label_column: label, 'p_value': p})

        # Convert the results to a DataFrame
        p_values_df = pd.DataFrame(p_values)

        return p_values_df

    @staticmethod
    def calculate_overall_p_value(study_df, control_df, label_column='label', total_column='total'):
        """
        Calculates the overall P-value for the difference in distribution between 
        the study and control groups using a Chi-square test.

        Parameters:
        - study_df (pd.DataFrame): A DataFrame containing the study group data.
        - control_df (pd.DataFrame): A DataFrame containing the control group data.
        - label_column (str): The column name that contains the labels (e.g., 'label'). Default is 'label'.
        - total_column (str): The column name that contains the totals (e.g., 'total'). Default is 'total'.

        Returns:
        - float: The overall P-value for the distribution comparison.
        """
        # Ensure both DataFrames are sorted by label
        study_df = study_df.sort_values(label_column).reset_index(drop=True)
        control_df = control_df.sort_values(label_column).reset_index(drop=True)
        
        # Ensure that both DataFrames contain the same set of labels
        common_labels = study_df[label_column].isin(control_df[label_column])
        if not common_labels.all():
            missing_labels = study_df.loc[~common_labels, label_column].values
            print(f"Missing labels in control group: {missing_labels}")
            study_df = study_df[common_labels]

        common_labels_control = control_df[label_column].isin(study_df[label_column])
        if not common_labels_control.all():
            missing_labels_control = control_df.loc[~common_labels_control, label_column].values
            print(f"Missing labels in study group: {missing_labels_control}")
            control_df = control_df[common_labels_control]
        
        # Create the contingency table for the overall distribution comparison
        contingency_table = [study_df[total_column].values, control_df[total_column].values]

        # Perform Chi-square test
        _, overall_p, _, _ = chi2_contingency(contingency_table)

        return overall_p

    @staticmethod
    def add_concept_columns(dataframe, concepts_df, concepts, person_id_col='person_id', concept_id_col='concept_id'):
        """
        Updates the dataframe by adding columns for each concept in the concepts list.
        The added columns will have a value of 1 if the person_id and concept_id match between
        concepts_df and the dataframe, otherwise 0. The original dataframe will remain unaltered,
        and a copy will be modified.

        Parameters:
        dataframe (pd.DataFrame): The DataFrame containing the main data with person_id.
        concepts_df (pd.DataFrame): The DataFrame containing the concept data with person_id and concept_id.
        concepts (list): A list of dictionaries where each dictionary represents a concept and contains:
                        'label' (str): The column name to be added,
                        'id' (int): The concept_id to match.
        person_id_col (str): The column name for the person ID in both dataframes. Defaults to 'person_id'.
        concept_id_col (str): The column name for the concept ID in concepts_df. Defaults to 'concept_id'.

        Returns:
        pd.DataFrame: A copy of the updated dataframe with new columns for each concept.
        """
        
        # Make a copy of the dataframe to avoid altering the original DataFrame
        updated_df = dataframe.copy()

        # Create a dictionary to map concept_id to concept labels
        concept_mapping = {concept['id']: concept['label'] for concept in concepts}

        # Add the concept columns to updated_df with default value 0
        for concept in concepts:
            updated_df[concept['label']] = 0

        # Update updated_df by checking for matches in concepts_df
        for _, row in concepts_df.iterrows():
            person_id = row[person_id_col]
            concept_id = row[concept_id_col]

            # If the concept_id exists in our concept mapping, update the corresponding label column
            if concept_id in concept_mapping:
                label = concept_mapping[concept_id]
                updated_df.loc[updated_df[person_id_col] == person_id, label] = 1

        # Ensure that all new concept columns are explicitly cast as integers
        for concept in concepts:
            updated_df[concept['label']] = updated_df[concept['label']].astype(int)

        return updated_df
    
    @staticmethod
    def add_flag_for_intersection(group_df, other_df, flag_name):
        """
        Adds a flag to the given group DataFrame based on the presence of person_id in another DataFrame.

        Parameters:
        group_df (pd.DataFrame): The DataFrame representing the group (e.g., study or control).
        other_df (pd.DataFrame): The DataFrame containing person_ids to check for intersection.
        flag_name (str): The name of the flag column to be added to the group DataFrame.

        Returns:
        pd.DataFrame: The group DataFrame with the added flag column.
        """
        group_df_copy = group_df.copy()
        group_df_copy[flag_name] = np.where(group_df_copy['person_id'].isin(other_df['person_id']), 1, 0)
        return group_df_copy
    
    @staticmethod
    def add_flag_based_on_condition(group_df, condition_df, flag_name):
        """
        Alias for add_flag_for_intersection.
        """
        return Utils.add_flag_for_intersection(group_df, condition_df, flag_name)
    
    @staticmethod
    def flag_group_by_condition(group_df, condition_df, flag_name):
        """
        Alias for add_flag_based_on_condition.
        """
        return MyClass.add_flag_based_on_condition(group_df, condition_df, flag_name)

    
    @staticmethod
    def get_totals_with_labels(df, key_label_map):
        """
        Function to compute totals for specified columns in a DataFrame and return a new DataFrame with labels.

        Parameters:
        df (pd.DataFrame): The input DataFrame.
        key_label_map (dict): A dictionary where keys are column headers and values are their corresponding labels.

        Returns:
        pd.DataFrame: A DataFrame with columns 'Label' and 'Total'.
        """
        df[list(key_label_map.keys())] = df[list(key_label_map.keys())].astype(int)

        # Compute the total for each specified column
        totals = df[list(key_label_map.keys())].sum().values

        # Create a new DataFrame with labels and totals
        result = pd.DataFrame({
            'Label': list(key_label_map.values()),
            'Total': totals
        })

        return result