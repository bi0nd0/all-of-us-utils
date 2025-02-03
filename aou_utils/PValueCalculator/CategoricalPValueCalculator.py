import pandas as pd
from .PValueUtils import PValueUtils
from .PValueCalculator import PValueCalculator

class CategoricalPValueCalculator(PValueCalculator):
    def __init__(self, study_df: pd.DataFrame, control_df: pd.DataFrame, variable: str, label: str = None):
        self.study_df = study_df
        self.control_df = control_df
        self.variable = variable
        self.label = label if label is not None else variable

    def calculate(self) -> pd.DataFrame:
        p_val = PValueUtils.calculate_p_value_categorical(self.study_df, self.control_df, self.variable)
        formatted = PValueUtils.format_p_value(p_val)
        return pd.DataFrame([{'Variable': self.label, 'Type': 'Categorical', 'P-value': formatted}])
