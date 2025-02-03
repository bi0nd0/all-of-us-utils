import pandas as pd
from .PValueHelper import PValueHelper
from .PValueCalculator import PValueCalculator

class SurveyPValueCalculator(PValueCalculator):
    def __init__(self, study_df: pd.DataFrame, control_df: pd.DataFrame, label: str = "Survey"):
        self.study_df = study_df
        self.control_df = control_df
        self.label = label

    def calculate(self) -> pd.DataFrame:
        p_val = PValueHelper.calculate_survey_p_value(self.study_df, self.control_df)
        formatted = PValueHelper.format_p_value(p_val)
        return pd.DataFrame([{'Variable': self.label, 'Type': 'Survey', 'P-value': formatted}])
