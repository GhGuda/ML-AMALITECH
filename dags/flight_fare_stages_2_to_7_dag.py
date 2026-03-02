"""Airflow DAG to orchestrate flight fare pipeline stages 2 through 7.

This DAG intentionally excludes Stage 1.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
import sys

from airflow import DAG
from airflow.operators.python import PythonOperator


# Resolve project paths so this DAG can import local pipeline modules.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from flight_fare.settings import (  # noqa: E402
    Stage1Settings,
    Stage2Settings,
    Stage3Settings,
    Stage4Settings,
    # Stage5Settings,
    Stage6Settings,
    Stage7Settings,
)
from flight_fare.stage1_pipeline import run_stage_1 as stage_1_runner  # noqa: E402
from flight_fare.stage2_pipeline import run_stage_2 as stage_2_runner  # noqa: E402
from flight_fare.stage3_eda import run_stage_3 as stage_3_runner  # noqa: E402
from flight_fare.stage4_baseline import run_stage_4 as stage_4_runner  # noqa: E402
# from flight_fare.stage5_advanced import run_stage_5 as stage_5_runner  # noqa: E402
from flight_fare.stage6_interpretation import run_stage_6 as stage_6_runner  # noqa: E402
from flight_fare.stage7_delivery import run_stage_7 as stage_7_runner  # noqa: E402


def run_stage_1() -> None:
    """Execute Stage 1 preprocessing."""
    stage_1_runner(settings=Stage1Settings.from_env())
    
    
def run_stage_2() -> None:
    """Execute Stage 2 preprocessing."""
    stage_2_runner(settings=Stage2Settings.from_env())


def run_stage_3() -> None:
    """Execute Stage 3 EDA."""
    stage_3_runner(settings=Stage3Settings.from_env())


def run_stage_4() -> None:
    """Execute Stage 4 baseline modeling."""
    stage_4_runner(settings=Stage4Settings.from_env())


# def run_stage_5() -> None:
#     """Execute Stage 5 advanced modeling/tuning."""
#     stage_5_runner(settings=Stage5Settings.from_env())


def run_stage_6() -> None:
    """Execute Stage 6 interpretation and stakeholder insights."""
    stage_6_runner(settings=Stage6Settings.from_env())


def run_stage_7() -> None:
    """Execute Stage 7 packaging/delivery."""
    stage_7_runner(settings=Stage7Settings.from_env())

# ============================================================
# Default DAG Arguments
# ============================================================

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
    
    # EMAIL ALERTING (disabled by default to avoid SMTP failures in local/dev)
    "email": [],
    "email_on_failure": False,
    "email_on_retry": False,
}

# ============================================================
# DAG Definition
# ============================================================
with DAG(
    dag_id="flight_fare_stages_2_to_7",
    description="Orchestrate flight fare pipeline stages 2 to 7 (Stage 1 excluded).",
    default_args=default_args,
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    tags=["flight_fare", "ml_pipeline"],
) as dag:
    stage_1 = PythonOperator(
        task_id="stage_1_automated_problem-framing_preprocessing",
        python_callable=run_stage_1,
    )
    
    stage_2 = PythonOperator(
        task_id="stage_2_preprocessing",
        python_callable=run_stage_2,
    )

    stage_3 = PythonOperator(
        task_id="stage_3_eda",
        python_callable=run_stage_3,
    )

    stage_4 = PythonOperator(
        task_id="stage_4_baseline_model",
        python_callable=run_stage_4,
    )

    # stage_5 = PythonOperator(
    #     task_id="stage_5_advanced_modeling",
    #     python_callable=run_stage_5,
    # )

    stage_6 = PythonOperator(
        task_id="stage_6_interpretation",
        python_callable=run_stage_6,
    )

    stage_7 = PythonOperator(
        task_id="stage_7_packaging_delivery",
        python_callable=run_stage_7,
    )

    # --------------------------------------------------
    # DEPENDENCIES (clear & linear)
    # --------------------------------------------------
    stage_1 >> stage_2 >> stage_3 >> stage_4 >> stage_6 >> stage_7

