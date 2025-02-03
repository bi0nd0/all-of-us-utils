import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, fisher_exact

class SexAtBirthPValueCalculator:
    def __init__(self, study_df: pd.DataFrame, control_df: pd.DataFrame, label: str = "Sex at Birth"):
        """
        Initialize the calculator.

        Parameters:
            study_df (pd.DataFrame): DataFrame for the study group.
                                     Expected to have a 'sex_at_birth' column.
                                     Optionally, may have a 'count' column if data is aggregated.
            control_df (pd.DataFrame): DataFrame for the control group.
                                       Expected to have a 'sex_at_birth' column.
                                       Optionally, may have a 'count' column if data is aggregated.
            label (str): Label to be used for the output.
        """
        self.study_df = study_df.copy()
        self.control_df = control_df.copy()
        self.label = label

    def _get_totals(self, df: pd.DataFrame) -> int:
        """
        Returns the total count for the DataFrame. If the DataFrame has a 'count' column,
        sum it up; otherwise, return the number of rows.
        """
        if 'count' in df.columns:
            return df['count'].sum()
        else:
            return len(df)

    def _get_category_count(self, df: pd.DataFrame, category: str) -> int:
        """
        Returns the count for a given category in 'sex_at_birth'.
        If a 'count' column exists, sum the counts for that category; 
        otherwise, count the number of occurrences.
        """
        if 'count' in df.columns:
            return df.loc[df['sex_at_birth'] == category, 'count'].sum()
        else:
            return (df['sex_at_birth'] == category).sum()

    def calculate(self) -> pd.DataFrame:
        """
        Calculate p-values for each sex category comparing study and control groups.
        
        For each sex category:
          - Build a 2x2 contingency table:
              [[study_count, study_total - study_count],
               [control_count, control_total - control_count]]
          - Use a chi-square test; if any expected frequency is less than 5 (and the table is 2x2),
            use Fisher's exact test instead.
        
        Returns:
            pd.DataFrame: A DataFrame containing the sex category, counts, and p-values.
        """
        results = []

        # Determine unique sex categories across both DataFrames.
        categories = np.union1d(
            self.study_df['sex_at_birth'].unique(),
            self.control_df['sex_at_birth'].unique()
        )
        
        # Determine totals for each group.
        study_total = self._get_totals(self.study_df)
        control_total = self._get_totals(self.control_df)
        
        for sex in categories:
            # Get counts for the current category.
            study_count = self._get_category_count(self.study_df, sex)
            control_count = self._get_category_count(self.control_df, sex)
            
            # Construct the 2x2 contingency table.
            contingency_table = np.array([
                [study_count, study_total - study_count],
                [control_count, control_total - control_count]
            ])
            
            # Perform the chi-square test.
            chi2, p_value, dof, expected = chi2_contingency(contingency_table, correction=False)
            
            # If any expected frequency is less than 5 and the table is 2x2, use Fisher's exact test.
            if contingency_table.shape == (2, 2) and (expected < 5).any():
                _, p_value = fisher_exact(contingency_table)
            
            results.append({
                "Variable": self.label,
                "Sex": sex,
                "Study Count": study_count,
                "Control Count": control_count,
                "P-value": p_value
            })
        
        return pd.DataFrame(results)