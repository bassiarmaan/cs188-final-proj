"""Import this before robosuite.make so LineLayoutBlocksEnv is registered."""

from robosuite.environments.base import register_env

from .horizontal_line_policy import pick_indices_in_palette_order
from .instruction_bridge import (
    ControlBinding,
    ParsedInstruction,
    Subgoal,
    TaskType,
    build_subgoal_plan,
    control_binding_for_plan,
    instantiate_policy,
    parse_instruction,
    run_parsed_instruction,
)
from .layout_verify import (
    check_horizontal_adjacent_line,
    check_vertical_adjacent_stack,
    report_to_info_dict,
    vertical_stack_report_to_dict,
)
from .line_layout_blocks_env import LineLayoutBlocksEnv
from .vertical_line_policy import VerticalLineAssemblyPolicy

register_env(LineLayoutBlocksEnv)

__all__ = [
    "LineLayoutBlocksEnv",
    "VerticalLineAssemblyPolicy",
    "ControlBinding",
    "ParsedInstruction",
    "Subgoal",
    "TaskType",
    "build_subgoal_plan",
    "check_horizontal_adjacent_line",
    "check_vertical_adjacent_stack",
    "control_binding_for_plan",
    "instantiate_policy",
    "parse_instruction",
    "pick_indices_in_palette_order",
    "report_to_info_dict",
    "run_parsed_instruction",
    "vertical_stack_report_to_dict",
]
