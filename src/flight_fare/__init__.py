"""Flight fare prediction package."""

from .settings import Stage1Settings, Stage2Settings, Stage3Settings, Stage4Settings
from .stage1_pipeline import Stage1Outputs, run_stage_1
from .stage2_pipeline import Stage2Outputs, run_stage_2
from .stage3_eda import Stage3Outputs, run_stage_3
from .stage4_baseline import Stage4Outputs, run_stage_4

__all__ = [
    "Stage1Outputs",
    "Stage1Settings",
    "Stage2Outputs",
    "Stage2Settings",
    "Stage3Outputs",
    "Stage3Settings",
    "Stage4Outputs",
    "Stage4Settings",
    "run_stage_1",
    "run_stage_2",
    "run_stage_3",
    "run_stage_4",
]
