from abc import ABC, abstractmethod
import pandas as pd

class PValueCalculator(ABC):
    @abstractmethod
    def calculate(self) -> pd.DataFrame:
        """
        Calculate the p-value(s) and return them as a DataFrame.
        """
        pass
