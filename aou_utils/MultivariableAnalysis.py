import pandas as pd
import numpy as np
import statsmodels.api as sm
import re
import warnings
import traceback
from statsmodels.stats.outliers_influence import variance_inflation_factor
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from scipy.stats import fisher_exact  # Import for Fisher's Exact Test


class MultivariableAnalysis:
    def __init__(self, study_group_df, control_group_df):
        """
        Initialize the MultivariableAnalysis class.

        Parameters:
        study_group_df (pd.DataFrame): The DataFrame for the study group.
        control_group_df (pd.DataFrame): The DataFrame for the control group.
        """
        if 'person_id' not in study_group_df.columns or 'person_id' not in control_group_df.columns:
            raise ValueError("Both dataframes must contain a 'person_id' column.")

        # Combine the dataframes
        self.combined_df = pd.concat([study_group_df.copy(), control_group_df.copy()], ignore_index=True)
        self.study_group_df = study_group_df.copy()
        self.independent_vars = []
        self.debug = False
        self.model = None
        self.maxModelIteration = 1000
        self.main_predictor = None  # Add this attribute to track the main predictor


    def set_maxModelIteration(self, count: int):
        self.maxModelIteration = count
        return self

    def set_debug(self, flag: bool = True):
        """
        Enable or disable debug mode.

        Parameters:
        flag (bool): True to enable debug mode, False to disable.
        """
        self.debug = flag
        return self

    def add_flag(self, flag_var):
        """
        Add a flag variable to indicate the study group status.

        Parameters:
        flag_var (str): The name of the flag variable to add, which will indicate whether a person belongs to the study group.

        Returns:
        MultivariableAnalysis: Returns the instance to allow method chaining.
        """
        if 'person_id' not in self.combined_df.columns:
            raise ValueError("The combined DataFrame must contain a 'person_id' column.")

        study_ids = set(self.study_group_df['person_id'])
        self.combined_df[flag_var] = self.combined_df['person_id'].apply(lambda x: 1 if x in study_ids else 0)
        self.independent_vars.append(flag_var)
        self.main_predictor = flag_var  # Set the main predictor variable
        return self

    @staticmethod
    def _clean_column_name(column_name):
        """
        Clean column names by:
        - Lowercasing
        - Replacing spaces with underscores
        - Removing non-alphanumeric characters except underscores

        Parameters:
        column_name (str): The column name to clean.

        Returns:
        str: The cleaned column name.
        """
        column_name = column_name.lower().replace(' ', '_')
        return re.sub(r'[^\w]', '', column_name)

    def add_dummies(self, column_name, drop_category=None):
        """
        Convert a categorical variable into dummy variables for a specified column,
        and drop a specified category or the first category if none is specified.

        Parameters:
        column_name (str): The name of the categorical column to convert into dummy variables.
        drop_category (str, optional): The category to drop, which will serve as the reference group. If not specified, the first category will be dropped.

        Returns:
        MultivariableAnalysis: Returns the instance to allow method chaining.
        """
        if column_name not in self.combined_df.columns:
            raise ValueError(f"Column '{column_name}' does not exist in the DataFrame.")

        # Generate dummy variables and apply name cleaning
        dummies = pd.get_dummies(self.combined_df[column_name], prefix=column_name, drop_first=False, dtype=int)
        # Clean column names
        dummies.columns = [self._clean_column_name(col) for col in dummies.columns]

        if self.debug:
            print(f"Dummy variables for '{column_name}': {list(dummies.columns)}")

        # If drop_category is specified, drop the corresponding dummy variable
        if drop_category is not None:
            drop_col_name = self._clean_column_name(f"{column_name}_{drop_category}")
            if self.debug:
                print(f"Attempting to drop category '{drop_col_name}' from dummies.")
            if drop_col_name in dummies.columns:
                dummies.drop(columns=[drop_col_name], inplace=True)
                if self.debug:
                    print(f"Dropped category '{drop_col_name}'.")
            else:
                print(f"Warning: The category '{drop_category}' does not exist in the column '{column_name}'. No category will be dropped.")
        else:
            # If no drop_category is specified, drop the first column automatically
            first_column = dummies.columns[0]
            dummies.drop(columns=[first_column], inplace=True)
            if self.debug:
                print(f"Dropped first category '{first_column}' from dummies.")

        # Concatenate the dummy variables to the combined DataFrame
        self.combined_df = pd.concat([self.combined_df, dummies], axis=1)

        # Add the dummy variable column names to the list of independent variables
        self.independent_vars.extend(dummies.columns)

        return self

    def add_independent_var(self, variable_name):
        """
        Add an independent variable to the model.

        Parameters:
        variable_name (str): The name of the independent variable to add.

        Returns:
        MultivariableAnalysis: Returns the instance to allow method chaining.
        """
        if variable_name not in self.combined_df.columns:
            raise ValueError(f"Variable '{variable_name}' does not exist in the DataFrame.")
        self.independent_vars.append(variable_name)
        return self

    def ensure_numeric(self, column):
        """
        Ensure that a column is numeric, and attempt to convert it if necessary.

        Parameters:
        column (str): The name of the column to check and convert.

        Returns:
        None
        """
        if column not in self.combined_df.columns:
            raise ValueError(f"Column '{column}' does not exist in the DataFrame.")
        if not pd.api.types.is_numeric_dtype(self.combined_df[column]):
            if self.debug:
                print(f"Warning: The column '{column}' is not numeric. Attempting to convert.")
            self.combined_df[column] = pd.to_numeric(self.combined_df[column], errors='coerce')
        if self.combined_df[column].isna().any():
            if self.debug:
                print(f"Warning: Conversion of column '{column}' resulted in NaNs. These rows will be dropped.")
        if np.isinf(self.combined_df[column]).any():
            if self.debug:
                print(f"Warning: Column '{column}' contains infinite values. Replacing with NaN.")
            self.combined_df[column].replace([np.inf, -np.inf], np.nan, inplace=True)

    def check_perfect_separation(self, dependent_var):
        """
        Check for perfect separation between independent variables and the dependent variable.

        Parameters:
        dependent_var (str): The name of the dependent variable.

        Returns:
        bool: True if perfect separation involving the main predictor is detected.
        """
        vars_to_remove = []
        perfect_separation_main_predictor = False

        for var in self.independent_vars:
            if var in self.combined_df.columns:
                crosstab = pd.crosstab(self.combined_df[var], self.combined_df[dependent_var])
                if self.debug:
                    print(f"\nCrosstab of {var} and {dependent_var}:")
                    print(crosstab)
                # Check for categories where the outcome is always the same
                zero_in_column = (crosstab == 0).any(axis=1)
                if zero_in_column.any():
                    if self.debug:
                        print(f"Variable '{var}' may cause perfect separation.")
                    if var == self.main_predictor:
                        perfect_separation_main_predictor = True
                        if self.debug:
                            print(f"Perfect separation involves the main predictor '{var}'.")
                    else:
                        if self.debug:
                            print(f"Variable '{var}' will be removed due to perfect separation.")
                        vars_to_remove.append(var)
        # Remove variables causing perfect separation (excluding the main predictor)
        for var in vars_to_remove:
            self.independent_vars.remove(var)

        return perfect_separation_main_predictor

    def perform_fishers_exact_test(self, dependent_var):
        """
        Perform Fisher's Exact Test between the main predictor and the dependent variable.

        Parameters:
        dependent_var (str): The name of the dependent variable.

        Returns:
        dict: A dictionary containing the odds ratio and p-value.
        """
        if self.main_predictor is None:
            raise ValueError("Main predictor variable is not set.")

        # Construct contingency table
        contingency_table = pd.crosstab(
            self.combined_df[self.main_predictor],
            self.combined_df[dependent_var]
        )

        if self.debug:
            print("\nContingency Table for Fisher's Exact Test:")
            print(contingency_table)

        # Ensure the table is 2x2
        if contingency_table.shape != (2, 2):
            raise ValueError("Fisher's Exact Test requires a 2x2 contingency table.")

        odds_ratio, p_value = fisher_exact(contingency_table)
        if self.debug:
            print(f"\nFisher's Exact Test Results:\nOdds Ratio: {odds_ratio}\nP-value: {p_value}")

        # Prepare the results in the same format as get_results()
        results_df = pd.DataFrame({
            'Variable': [self.main_predictor],
            'Odds Ratio': [odds_ratio],
            'Lower CI': [np.nan],  # Fisher's Exact Test does not provide CI by default
            'Upper CI': [np.nan],
            'P-value': [self.format_p_value(p_value)],
            'Formatted Results': [f"{odds_ratio:.2f} (Exact)"]
        })

        return results_df

    def check_multicollinearity(self):
        """
        Calculate VIF for each independent variable to detect multicollinearity.

        Returns:
        pd.DataFrame: A DataFrame containing VIF values.
        """
        X = self.combined_df[self.independent_vars].copy()
        X = sm.add_constant(X)
        vif_data = pd.DataFrame()
        vif_data["Variable"] = X.columns

        vif_values = []
        for i in range(X.shape[1]):
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                try:
                    vif = variance_inflation_factor(X.values, i)
                    vif_values.append(vif)
                except (np.linalg.LinAlgError, ZeroDivisionError):
                    vif_values.append(np.inf)

        vif_data["VIF"] = vif_values

        if self.debug:
            print("\nVariance Inflation Factor (VIF):")
            print(vif_data)
        return vif_data

    def check_data_quality(self, dependent_var):
        """
        Check data types, missing values, and variables with zero variance.

        Parameters:
        dependent_var (str): The name of the dependent variable.

        Returns:
        None
        """
        cols_to_check = self.independent_vars + [dependent_var]
        missing_cols = [col for col in cols_to_check if col not in self.combined_df.columns]
        if missing_cols:
            raise ValueError(f"The following columns are missing in the DataFrame: {missing_cols}")

        print("\nData Types:")
        print(self.combined_df[cols_to_check].dtypes)

        print("\nMissing Values:")
        print(self.combined_df[cols_to_check].isnull().sum())

        print("\nVariables with Zero Variance:")
        for var in self.independent_vars:
            unique_values = self.combined_df[var].nunique()
            if unique_values <= 1:
                print(f"Variable '{var}' has {unique_values} unique value(s).")

    def plot_correlation_matrix(self):
        """
        Plot a heatmap of the correlation matrix of independent variables.

        Returns:
        None
        """
        X = self.combined_df[self.independent_vars]
        corr_matrix = X.corr()

        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
        plt.title("Correlation Matrix of Independent Variables")
        plt.show()

    def fit_model_with_regularization(self, dependent_var, method='l1', alpha=0.1):
        """
        Fit a logistic regression model using regularization.

        Parameters:
        dependent_var (str): The name of the dependent variable.
        method (str): Regularization method ('l1' or 'l2').
        alpha (float): Regularization strength.

        Returns:
        MultivariableAnalysis: Returns the instance to allow method chaining.
        """
        # Ensure all relevant columns are numeric
        for var in self.independent_vars + [dependent_var]:
            self.ensure_numeric(var)

        # Drop rows with NaN values
        self.combined_df.dropna(subset=self.independent_vars + [dependent_var], inplace=True)

        # Define the independent variables (X) and dependent variable (y)
        X = sm.add_constant(self.combined_df[self.independent_vars])
        y = self.combined_df[dependent_var]

        if self.debug:
            print("\nData used for regularized model fitting (first 5 rows):")
            print(X.head())
            print(y.head())

        # Fit the logistic regression model with regularization
        try:
            self.model = sm.Logit(y, X).fit_regularized(method=method, alpha=alpha, maxiter=self.maxModelIteration)
        except Exception as e:
            if self.debug:
                print(f"Error fitting model with regularization: {e}")
                print(traceback.format_exc())
            raise RuntimeError(f"Error fitting model with regularization: {e}")
        return self

    def fit_model(self, dependent_var, robust=False, standardize=False):
        """
        Fit a logistic regression model using the specified dependent variable.

        Parameters:
        dependent_var (str): The name of the dependent variable.
        robust (bool): Whether to use robust standard errors.
        standardize (bool): Whether to standardize the independent variables.

        Returns:
        MultivariableAnalysis: Returns the instance to allow method chaining.
        """
        for var in self.independent_vars + [dependent_var]:
            self.ensure_numeric(var)

        self.combined_df.dropna(subset=self.independent_vars + [dependent_var], inplace=True)

        X = self.combined_df[self.independent_vars].copy()
        if standardize:
            X = (X - X.mean()) / X.std()
        X = sm.add_constant(X)
        y = self.combined_df[dependent_var]

        if self.debug:
            print("\nData used for model fitting (first 5 rows):")
            print(X.head())
            print(y.head())

        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                if robust:
                    self.model = sm.Logit(y, X).fit(cov_type='HC3', maxiter=self.maxModelIteration)
                else:
                    self.model = sm.Logit(y, X).fit(maxiter=self.maxModelIteration)

                # Check for convergence warnings
                for warning in w:
                    if issubclass(warning.category, ConvergenceWarning):
                        if self.debug:
                            print(f"Convergence warning: {warning.message}")
                        # Try a different solver
                        self.model = sm.Logit(y, X).fit(method='bfgs', maxiter=self.maxModelIteration)
                        break
        except Exception as e:
            if self.debug:
                print(f"Error fitting model: {e}")
                print(traceback.format_exc())
            raise RuntimeError(f"Error fitting model: {e}")
        return self

    def convert_column_to_type(self, column_name, dtype):
        """
        Convert a column to a specified data type.

        Parameters:
        column_name (str): The name of the column to convert.
        dtype (str): The target data type.

        Returns:
        MultivariableAnalysis: Returns the instance to allow method chaining.
        """
        if column_name not in self.combined_df.columns:
            raise ValueError(f"Column '{column_name}' does not exist in the DataFrame.")
        try:
            self.combined_df[column_name] = self.combined_df[column_name].astype(dtype)
            if self.debug:
                print(f"Successfully converted column '{column_name}' to {dtype}.")
        except Exception as e:
            if self.debug:
                print(f"Error converting column '{column_name}' to {dtype}: {e}")
            raise ValueError(f"Error converting column '{column_name}' to {dtype}: {e}")
        return self

    @staticmethod
    def format_p_value(p_value):
        """
        Format the p-value for display purposes.

        Parameters:
        p_value (float): The p-value to format.

        Returns:
        str: The formatted p-value as a string.
        """
        if p_value < 0.001:
            return "<0.001"
        else:
            return f"{p_value:.3f}"

    def get_results(self, print_results=True):
        """
        Retrieve and display the results of the logistic regression analysis.

        Parameters:
        print_results (bool): Whether to print the results.

        Returns:
        pd.DataFrame: A DataFrame containing the odds ratios, confidence intervals, and formatted results.
        """
        if hasattr(self, 'results_df') and self.results_df is not None:
            # Results from Fisher's Exact Test or other alternative method
            results_df = self.results_df
        elif not hasattr(self, 'model') or self.model is None:
            raise AttributeError("The model has not been fitted yet. Please fit the model before getting results.")
        else:
            # Get the odds ratios and confidence intervals
            odds_ratios = self.model.params
            confidence_intervals = self.model.conf_int()

            # Use a safe exponential function to handle overflows
            def safe_exp(x):
                with np.errstate(over='ignore', under='ignore'):
                    return np.exp(x)

            odds_ratios_exp = odds_ratios.apply(safe_exp)
            confidence_intervals_exp = confidence_intervals.applymap(safe_exp)

            # Replace infinite values with NaN
            odds_ratios_exp.replace([np.inf, -np.inf], np.nan, inplace=True)
            confidence_intervals_exp.replace([np.inf, -np.inf], np.nan, inplace=True)

            # Create a DataFrame to store the results
            results_df = pd.DataFrame({
                'Variable': odds_ratios.index,
                'Odds Ratio': odds_ratios_exp,
                'Lower CI': confidence_intervals_exp.iloc[:, 0],
                'Upper CI': confidence_intervals_exp.iloc[:, 1],
                'P-value': self.model.pvalues
            })

            # Format the p-values
            results_df['P-value'] = results_df['P-value'].apply(self.format_p_value)

            # Round the odds ratios and confidence intervals to 2 decimal places
            results_df[['Odds Ratio', 'Lower CI', 'Upper CI']] = results_df[['Odds Ratio', 'Lower CI', 'Upper CI']].round(2)

            # Create a new column with the formatted results
            results_df['Formatted Results'] = results_df.apply(
                lambda row: f"{row['Odds Ratio']} ({row['Lower CI']}-{row['Upper CI']})", axis=1)

        # Print the results DataFrame if required
        if print_results:
            print(results_df)

        return results_df


    def automate_analysis(self, dependent_var, vif_threshold=5.0, regularization_method='l1', alpha=0.1, standardize=False):
        """
        Automate the analysis by performing diagnostics, adjusting variables, and fitting the model.

        Parameters:
        dependent_var (str): The name of the dependent variable.
        vif_threshold (float): Threshold for VIF to detect multicollinearity. Variables with VIF above this value will be removed.
        regularization_method (str): Regularization method to use if needed ('l1' for Lasso, 'l2' for Ridge).
        alpha (float): Regularization strength.
        standardize (bool): Whether to standardize the independent variables.

        Returns:
        MultivariableAnalysis: Returns the instance to allow method chaining.
        """
        # Step 1: Check data quality
        if self.debug:
            print("Checking data quality...")
        self.check_data_quality(dependent_var)

        # Step 2: Check for perfect separation and remove problematic variables
        if self.debug:
            print("\nChecking for perfect separation...")
        perfect_separation_main_predictor = self.check_perfect_separation(dependent_var)

        # If perfect separation involves the main predictor, perform Fisher's Exact Test
        if perfect_separation_main_predictor:
            if self.debug:
                print("\nPerfect separation detected involving the main predictor. Performing Fisher's Exact Test.")
            self.model = None  # Set model to None to indicate alternative analysis
            self.results_df = self.perform_fishers_exact_test(dependent_var)
            return self

        # Step 3: Check for multicollinearity and remove variables with high VIF
        if self.debug:
            print("\nChecking for multicollinearity...")
        iteration = 0
        while True:
            vif_data = self.check_multicollinearity()
            vif_data = vif_data[vif_data['Variable'] != 'const']  # Exclude constant term
            if vif_data.empty:
                if self.debug:
                    print("No variables to check for multicollinearity.")
                break
            max_vif = vif_data['VIF'].max()
            if max_vif > vif_threshold:
                # Remove the variable with the highest VIF
                max_vif_var = vif_data.loc[vif_data['VIF'] == max_vif, 'Variable'].values[0]
                if self.debug:
                    print(f"Variable '{max_vif_var}' has VIF={max_vif:.2f} and will be removed.")
                self.independent_vars.remove(max_vif_var)
            else:
                if self.debug:
                    print("No multicollinearity issues detected.")
                break
            iteration += 1
            if iteration > 10:
                if self.debug:
                    print("Reached maximum iterations for VIF checking.")
                break

        # Step 4: Fit the model
        if self.debug:
            print("\nFitting the model...")
        try:
            # First attempt without regularization
            self.fit_model(dependent_var, standardize=standardize)
        except Exception as e:
            if self.debug:
                print(f"Error fitting model: {e}")
                print("Attempting to fit model with regularization.")
            try:
                # Attempt to fit model with regularization as a fallback
                self.fit_model_with_regularization(dependent_var, method=regularization_method, alpha=alpha)
            except Exception as reg_error:
                if self.debug:
                    print(f"Error fitting model with regularization: {reg_error}")
                raise RuntimeError("Model fitting failed even after regularization. Please check your data and variables.")

        return self
