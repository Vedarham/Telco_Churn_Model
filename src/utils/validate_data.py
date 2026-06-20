import pandas as pd
import pandera as pa
from pandera import Column, Check
from typing import Tuple, List


def validate_telco_data(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Validate Telco Customer Churn dataset using Pandera.
    Returns:
        (success, failed_checks)
    """

    print("🔍 Starting data validation...")

    failed_checks = []

    try:
        # Convert TotalCharges if needed
        if "TotalCharges" in df.columns:
            df["TotalCharges"] = pd.to_numeric(
                df["TotalCharges"],
                errors="coerce"
            )

        schema = pa.DataFrameSchema(
            {
                "customerID": Column(str, nullable=False),

                "gender": Column(
                    str,
                    Check.isin(["Male", "Female"])
                ),

                "Partner": Column(
                    str,
                    Check.isin(["Yes", "No"])
                ),

                "Dependents": Column(
                    str,
                    Check.isin(["Yes", "No"])
                ),

                "PhoneService": Column(
                    str,
                    Check.isin(["Yes", "No"])
                ),

                "InternetService": Column(
                    str,
                    Check.isin(["DSL", "Fiber optic", "No"])
                ),

                "Contract": Column(
                    str,
                    Check.isin([
                        "Month-to-month",
                        "One year",
                        "Two year"
                    ])
                ),

                "tenure": Column(
                    int,
                    checks=[
                        Check.ge(0),
                        Check.le(120)
                    ]
                ),

                "MonthlyCharges": Column(
                    float,
                    checks=[
                        Check.ge(0),
                        Check.le(200)
                    ]
                ),

                "TotalCharges": Column(
                    float,
                    checks=[
                        Check.ge(0)
                    ],
                    nullable=True
                )
            },
            strict=False
        )

        schema.validate(df)

        # Business Rule:
        # TotalCharges >= MonthlyCharges for 95% of records

        consistency_ratio = (
            (
                df["TotalCharges"].fillna(0)
                >= df["MonthlyCharges"]
            )
            .mean()
        )

        if consistency_ratio < 0.95:
            failed_checks.append(
                f"TotalCharges >= MonthlyCharges ratio = "
                f"{consistency_ratio:.2%} (<95%)"
            )

        success = len(failed_checks) == 0

        if success:
            print("✅ Data validation PASSED")
        else:
            print("❌ Data validation FAILED")
            print(failed_checks)

        return success, failed_checks

    except pa.errors.SchemaError as e:
        failed_checks.append(str(e))
        print("❌ Schema validation failed")
        return False, failed_checks

    except Exception as e:
        failed_checks.append(str(e))
        print("❌ Validation error")
        return False, failed_checks