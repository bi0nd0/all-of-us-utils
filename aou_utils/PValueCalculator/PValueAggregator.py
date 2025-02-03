import pandas as pd
from PValueCalculator import PValueCalculator

class PValueAggregator:
    def __init__(self):
        self.calculators = []

    def add_calculator(self, calculator: PValueCalculator):
        self.calculators.append(calculator)
        return self  # enable chaining

    def calculate_all(self) -> pd.DataFrame:
        results = []
        for calc in self.calculators:
            results.append(calc.calculate())
        # Combine all results into one DataFrame.
        return pd.concat(results, ignore_index=True)
