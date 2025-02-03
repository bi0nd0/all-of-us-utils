import pandas as pd
from .PValueUtils import PValueUtils
from .PValueCalculator import PValueCalculator

class SexAtBirthPValueCalculator(PValueCalculator):
    def __init__(self, study_df: pd.DataFrame, control_df: pd.DataFrame, label: str = "Sex at Birth"):
        """
        Initialize the calculator without needing to pass total counts explicitly.
        
        Parameters:
            study_df (pd.DataFrame): DataFrame for the study group.
            control_df (pd.DataFrame): DataFrame for the control group.
            label (str): Label for the output.
        """
        self.study_df = study_df
        self.control_df = control_df
        self.label = label

    def calculate(self) -> pd.DataFrame:
        """
        Calculate the p-values for sex at birth using aggregated counts.
        The total number of participants is inferred from the DataFrames.
        """
        # Infer the total participants from each DataFrame.
        study_total = len(self.study_df)
        control_total = len(self.control_df)
        
        # Use the specialized utility function to compute the p-values.
        # This function should accept study_total and control_total.
        df = PValueUtils.calculate_sex_at_birth_p_values(
            self.study_df, self.control_df, study_total, control_total
        )
        # Optionally override the "Variable" column with the custom label.
        df["Variable"] = self.label
        return df
