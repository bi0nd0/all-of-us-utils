import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, ranksums, fisher_exact

class PValueUtils:
    @staticmethod
    def calculate_p_value_continuous(group1_df, group2_df, column_name):
        """
        Calculate the P-value for a continuous variable using the Wilcoxon rank-sum test.
        """
        try:
            group1_values = group1_df[column_name]
            group2_values = group2_df[column_name]
            _, p_value = ranksums(group1_values, group2_values)
            print(f"Wilcoxon rank-sum test for {column_name}: p-value = {p_value}")
            return p_value
        except Exception as e:
            print(f"Error calculating continuous p-value for {column_name}: {e}")
            return None

    @staticmethod
    def calculate_p_value_categorical(group1_df, group2_df, column_name):
        """
        Calculate the P-value for a categorical variable using the chi-square test.
        """
        try:
            # Build a contingency table from the value counts.
            counts1 = group1_df[column_name].value_counts().sort_index()
            counts2 = group2_df[column_name].value_counts().sort_index()
            # Merge on the index (categories) and fill missing values with 0.
            contingency_df = pd.DataFrame({
                'Group1': counts1,
                'Group2': counts2
            }).fillna(0)
            contingency_table = contingency_df.values

            print(f"Contingency table for {column_name}:\n{contingency_table}")
            chi2, p_value, dof, expected = chi2_contingency(contingency_table, correction=False)
            if (expected < 5).any():
                print(f"Expected frequencies for {column_name} are low; trying Fisher's exact test.")
                # For Fisher's exact test, we can only handle 2x2 tables.
                if contingency_table.shape == (2, 2):
                    _, p_value = fisher_exact(contingency_table)
                else:
                    print("Fisher's exact test cannot be applied to tables larger than 2x2. Using chi-square p-value.")
            print(f"Chi-square test for {column_name}: p-value = {p_value}")
            return p_value
        except Exception as e:
            print(f"Error calculating categorical p-value for {column_name}: {e}")
            return None

    @staticmethod
    def format_p_value(p, threshold=0.001):
        """
        Format a single P-value, showing values below threshold as "<threshold".
        """
        try:
            if p is None:
                return "N/A"
            if p < threshold:
                return f"<{threshold}"
            else:
                return f"{p:.3f}"
        except Exception as e:
            print(f"Error formatting p-value: {e}")
            return "N/A"
