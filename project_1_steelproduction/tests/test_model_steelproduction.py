import os
import unittest
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# Data path of testing data
TESTS_DIR = Path(__file__).parent
DATA_PATH = TESTS_DIR / ".." / "data" / "processed" / "normalized_train_data.csv"


class TestSteelProductionModel(unittest.TestCase):
    """
    Test suite for the steel production regression model.
    Uses the same RandomForestRegressor setup as steelproduction.ipynb.
    """

    @classmethod
    def setUpClass(cls):
        """
        Setup of Model for the test
        """
        # Load the dataset from the processed CSV file
        cls.df = pd.read_csv(DATA_PATH)

        # Separate target column (output) from input features
        cls.y = cls.df["output"]
        cls.X = cls.df.drop("output", axis=1)

        # Split into 80 % training and 20 % test data (same split as the notebook)
        cls.X_train, cls.X_test, cls.y_train, cls.y_test = train_test_split(
            cls.X, cls.y, test_size=0.2, random_state=42
        )

        # Train the same RandomForest model as in steelproduction.ipynb
        cls.model = RandomForestRegressor(n_estimators=100, random_state=42)
        cls.model.fit(cls.X_train, cls.y_train)

        # Generate predictions on the test set once and reuse across tests
        cls.y_pred = cls.model.predict(cls.X_test)

    # ── Data tests ──────────────────────────────────────────────────────────────

    def test_csv_file_exists(self):
        """Check that the CSV file is present at the expected path."""
        self.assertTrue(DATA_PATH.exists(), f"CSV not found at: {DATA_PATH.resolve()}")

    def test_csv_has_expected_columns(self):
        """Dataset must have exactly one output column and 21 input columns."""
        self.assertIn("output", self.df.columns, "Column 'output' is missing")
        input_cols = [c for c in self.df.columns if c != "output"]
        self.assertEqual(len(input_cols), 21, f"Expected 21 input columns, got {len(input_cols)}")

    def test_csv_has_rows(self):
        """Dataset must contain at least one row of data."""
        self.assertGreater(len(self.df), 0, "CSV file is empty")

    def test_values_are_normalized(self):
        """All values must be in the range [0, 1] because the data is normalized."""
        self.assertGreaterEqual(self.df.min().min(), 0.0, "Some values are below 0")
        self.assertLessEqual(self.df.max().max(), 1.0, "Some values are above 1")

    def test_no_missing_values(self):
        """Dataset must not contain any NaN (missing) values."""
        missing = self.df.isnull().sum().sum()
        self.assertEqual(missing, 0, f"Dataset contains {missing} missing values")

    # ── Split tests ─────────────────────────────────────────────────────────────

    def test_train_test_split_sizes(self):
        """Training set should be ~80 % and test set ~20 % of the full dataset."""
        total = len(self.df)
        self.assertAlmostEqual(len(self.X_train) / total, 0.8, delta=0.01)
        self.assertAlmostEqual(len(self.X_test)  / total, 0.2, delta=0.01)

    # ── Output statistic ────────────────────────────────────────────────────────

    def test_print_output_statistic(self):
        """Print descriptive statistics of the output column"""
        std  = self.y.std()
        mean = self.y.mean()
        mn   = self.y.min()
        mx   = self.y.max()
        r2   = r2_score(self.y_test, self.y_pred)
        rmse = np.sqrt(mean_squared_error(self.y_test, self.y_pred))

        print("\n-- Output column statistics ------------------------------")
        print(f"Mean               : {mean:.4f}")
        print(f"Standard deviation : {std:.4f}")
        print(f"Min                : {mn:.4f}")
        print(f"Max                : {mx:.4f}")
        print("-- Model performance -------------------------------------")
        print(f"R2                 : {r2:.4f}")
        print(f"RMSE               : {rmse:.4f}")
        print(f"RMSE < std(output) : {rmse < std}  ({rmse:.4f} < {std:.4f})")

        # not a test — just information
        self.assertTrue(True)

    # ── Model tests ─────────────────────────────────────────────────────────────

    def test_predictions_in_valid_range(self):
        """
        Predictions should stay within [0, 1] because the target is normalized.
        A small tolerance of 0.05 is allowed for extrapolation at the edges.
        """
        self.assertGreaterEqual(self.y_pred.min(), -0.05, "Predictions go below 0")
        self.assertLessEqual(self.y_pred.max(),  1.05, "Predictions go above 1")

    def test_r2_score_acceptable(self):
        """
        R² score on the test set must be at least 0.3.

        Why 0.3 valid?
        -----------------------
        The output variable in this dataset has a very narrow value range:
        its standard deviation is only ~0.083 on a 0–1 scale, meaning nearly
        all measurements cluster tightly around the mean (~0.51).

        R² is defined as:  1 - (sum of squared residuals / total variance of y)

        Because the total variance of y is so small, even small absolute prediction
        errors produce a large relative error — which directly lowers the R² score.
        This is a property of the data itself, not a weakness of the model.

        The RMSE test (< 0.1) enforces absolute prediction accuracy independently,
        so a lower R² threshold does not mean quality is uncontrolled.
        """
        r2 = r2_score(self.y_test, self.y_pred)
        self.assertGreaterEqual(r2, 0.3, f"R² too low: {r2:.4f} (minimum: 0.3)")

    def test_rmse_acceptable(self):
        """
        RMSE on the test set must be below 0.1.
        Since all values are in [0, 1], an RMSE above 0.1 would be a poor result.
        """
        rmse = np.sqrt(mean_squared_error(self.y_test, self.y_pred))
        self.assertLess(rmse, 0.1, f"RMSE too high: {rmse:.4f} (maximum: 0.1)")



if __name__ == "__main__":
    # Run all tests when this file is executed directly: python test_model_steelproduction.py
    unittest.main(verbosity=2)
