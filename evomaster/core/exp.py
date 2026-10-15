"""EvoMaster Core - Base Classes and Common Workflows

Provides base implementations for Exp and Playground, for concrete playgrounds to inherit.
"""

import json
import logging
from pathlib import Path
from evomaster.utils.types import TaskInstance
from typing import Any


def extract_agent_response(trajectory: Any) -> str:
    """Extract the agent's final response from a trajectory (module-level utility function).

    Supports two data formats:
    - Object format (has .dialogs attribute, from runtime)
    - Dict format (JSON deserialization result, from trajectory files)

    Extraction priority:
    1. The message parameter from the finish tool_call in the last assistant message
    2. The content of the last assistant message that has content

    Args:
        trajectory: Execution trajectory (object or dict).

    Returns:
        The agent's response text; returns an empty string if extraction fails.
    """
    if not trajectory:
        return ""

    # Get dialogs (compatible with both object and dict)
    if isinstance(trajectory, dict):
        dialogs = trajectory.get("dialogs")
    elif hasattr(trajectory, "dialogs"):
        dialogs = trajectory.dialogs
    else:
        return ""

    if not dialogs:
        return ""

    last_dialog = dialogs[-1]

    # Get messages (compatible with both object and dict)
    if isinstance(last_dialog, dict):
        messages = last_dialog.get("messages", [])
    else:
        messages = getattr(last_dialog, "messages", [])

    if not messages:
        return ""

    # Iterate in reverse to find the last assistant message
    for message in reversed(messages):
        if isinstance(message, dict):
            role = message.get("role", "")
            content = message.get("content", "")
            tool_calls = message.get("tool_calls", [])
        else:
            role = getattr(message, "role", None)
            role = role.value if hasattr(role, "value") else str(role) if role else ""
            content = getattr(message, "content", "")
            tool_calls = getattr(message, "tool_calls", [])

        if role != "assistant":
            continue

        # Prefer checking the finish tool_call's message parameter
        for tc in (tool_calls or []):
            if isinstance(tc, dict):
                func = tc.get("function", {})
                name = func.get("name", "")
                args = func.get("arguments", "")
            else:
                func = getattr(tc, "function", None)
                name = getattr(func, "name", "") if func else ""
                args = getattr(func, "arguments", "") if func else ""

            if name == "finish":
                try:
                    finish_args = json.loads(args) if isinstance(args, str) else args
                    finish_msg = finish_args.get("message", "")
                    if finish_msg:
                        return finish_msg
                except (json.JSONDecodeError, AttributeError):
                    pass

        # Fall back to content (skip messages with tool_calls, whose content may be leaked formatting tokens)
        if content and content.strip() and not tool_calls:
            return content

    return ""


def _reconstruct_from_jsonl(header: dict, steps: list[dict]) -> list[dict]:
    """Reconstruct full entries from JSONL header + delta step lines.

    Args:
        header: Header dict with tools, task_id, agent_name.
        steps: List of JSONL step dicts with messages_delta, assistant_message, tool_responses.

    Returns:
        List of trajectory entry dicts with full cumulative messages.
    """
    tools = header.get("tools", [])
    task_id = header.get("task_id", "unknown")
    agent_name = header.get("agent_name", "unknown")

    entries = []
    accumulated_messages = []

    for step_data in steps:
        step = step_data.get("step", 0)
        delta = step_data.get("messages_delta", [])
        accumulated_messages.extend(delta)

        entry = {
            "task_id": step_data.get("task_id", f"{task_id}_{agent_name}_step_{step}"),
            "exp_name": step_data.get("meta", {}).get("exp_name"),
            "exp_index": step_data.get("meta", {}).get("exp_index"),
            "status": step_data.get("status", "running"),
            "steps": step,
            "trajectory": {
                "task_id": task_id,
                "agent_name": agent_name,
                "step": step,
                "dialogs": [{
                    "messages": list(accumulated_messages),
                    "tools": tools,
                    "meta": {}
                }],
                "steps": [{
                    "step_id": step,
                    "assistant_message": step_data.get("assistant_message"),
                    "tool_responses": step_data.get("tool_responses", []),
                    "meta": {}
                }],
                "start_time": None,
                "end_time": None,
                "status": step_data.get("status", "running"),
                "result": {},
                "meta": step_data.get("meta", {}),
            }
        }
        entries.append(entry)

    return entries


def _reconstruct_from_compact_json(header: dict, steps: list) -> list[dict]:
    """Reconstruct full entries from compact JSON with flat message list.

    In this format, steps is a flat list of message dicts (all roles):
    [{"role":"system",...}, {"role":"user",...}, {"role":"assistant",...}, {"role":"tool",...}, ...]

    Args:
        header: Header dict with tools, task_id, agent_name.
        steps: Flat list of message dicts.

    Returns:
        A single trajectory entry dict containing all messages.
    """
    tools = header.get("tools", [])
    task_id = header.get("task_id", "unknown")
    agent_name = header.get("agent_name", "unknown")

    # Find the last assistant message
    last_assistant = None
    last_tool_responses = []
    for msg in reversed(steps):
        if isinstance(msg, dict) and msg.get("role") == "assistant":
            last_assistant = msg
            break

    return [{
        "task_id": f"{task_id}_{agent_name}",
        "exp_name": None,
        "exp_index": None,
        "status": "running",
        "steps": len(steps),
        "trajectory": {
            "task_id": task_id,
            "agent_name": agent_name,
            "step": len(steps),
            "dialogs": [{
                "messages": list(steps),
                "tools": tools,
                "meta": {}
            }],
            "steps": [{
                "step_id": len(steps),
                "assistant_message": last_assistant,
                "tool_responses": last_tool_responses,
                "meta": {}
            }],
            "start_time": None,
            "end_time": None,
            "status": "running",
            "result": {},
            "meta": {},
        }
    }]


def load_trajectory_file(file_path: str | Path) -> list[dict]:
    """Load a trajectory file (JSONL, compact JSON, or legacy JSON) and reconstruct full entries.

    Supports three formats:
    - JSONL (.jsonl): header line + delta step lines
    - Compact JSON (.json): {"type":"header","tools":[...],"steps":[...]}
    - Legacy JSON (.json): [{"task_id":"...","trajectory":{...}}, ...]

    Args:
        file_path: Path to the trajectory file.

    Returns:
        List of trajectory entry dicts with full cumulative messages.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        return []

    # Try JSONL first
    if file_path.suffix == '.jsonl':
        lines = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    lines.append(json.loads(line))
        if not lines:
            return []
        header = lines[0] if lines[0].get("type") == "header" else None
        if header is None:
            return []
        steps = [l for l in lines[1:] if l.get("type") == "step"]
        return _reconstruct_from_jsonl(header, steps)

    # JSON file
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Compact JSON format: {"type": "header", "tools": [...], "steps": [...]}
    if isinstance(data, dict) and data.get("type") == "header":
        steps = data.get("steps", [])
        if steps:
            first = steps[0]
            # New format: each step is a list of message dicts
            if isinstance(first, list):
                return _reconstruct_from_compact_json(data, steps)
            # Old compact: each step is a single assistant message dict with "role"
            if isinstance(first, dict) and "role" in first:
                return _reconstruct_from_compact_json(data, steps)
        # JSONL-like delta format
        return _reconstruct_from_jsonl(data, steps)

    # Legacy JSON format: array of entries
    if isinstance(data, list):
        return data

    return []


def migrate_trajectory_file(old_path: str | Path, new_path: str | Path | None = None) -> Path:
    """Convert a legacy JSON trajectory file to compact JSONL format.

    Args:
        old_path: Path to the existing .json trajectory file.
        new_path: Path for the output .jsonl file. Defaults to same name with .jsonl extension.

    Returns:
        Path to the migrated .jsonl file.
    """
    old_path = Path(old_path)
    if new_path is None:
        new_path = old_path.with_suffix('.jsonl')
    else:
        new_path = Path(new_path)

    with open(old_path, 'r', encoding='utf-8') as f:
        entries = json.load(f)

    if not entries:
        return new_path

    # Extract tools from the first entry's dialog
    first_traj = entries[0].get("trajectory", {})
    first_dialog = first_traj.get("dialogs", [{}])[0] if first_traj.get("dialogs") else {}
    tools = first_dialog.get("tools", [])
    task_id = first_traj.get("task_id", "unknown")
    agent_name = first_traj.get("agent_name", "unknown")

    with open(new_path, 'w', encoding='utf-8') as f:
        # Write header
        header = {
            "type": "header",
            "tools": tools,
            "task_id": task_id,
            "agent_name": agent_name,
        }
        f.write(json.dumps(header, default=str, ensure_ascii=False) + "\n")

        prev_message_count = 0
        for entry in entries:
            traj = entry.get("trajectory", {})
            step = entry.get("steps", traj.get("step", 0))
            dialogs = traj.get("dialogs", [])
            all_messages = dialogs[0].get("messages", []) if dialogs else []

            # Compute delta
            delta = all_messages[prev_message_count:]
            prev_message_count = len(all_messages)

            # Extract assistant_message and tool_responses from steps
            step_records = traj.get("steps", [])
            assistant_message = step_records[0].get("assistant_message") if step_records else None
            tool_responses = step_records[0].get("tool_responses", []) if step_records else []

            # Strip raw_response from delta messages
            for msg in delta:
                if isinstance(msg, dict) and "meta" in msg:
                    msg["meta"] = {k: v for k, v in msg["meta"].items() if k != "raw_response"}

            step_entry = {
                "type": "step",
                "task_id": entry.get("task_id", ""),
                "step": step,
                "status": entry.get("status", "running"),
                "messages_delta": delta,
                "assistant_message": assistant_message,
                "tool_responses": tool_responses,
                "meta": traj.get("meta", {}),
            }
            f.write(json.dumps(step_entry, default=str, ensure_ascii=False) + "\n")

    return new_path


class BaseExp:
    """Experiment base class.

    Defines the common execution logic for a single experiment.
    Concrete playgrounds can inherit and override relevant methods.
    """

    def __init__(self, agent, config):
        """Initialize the experiment.

        Args:
            agent: Agent instance.
            config: EvoMasterConfig instance.
        """
        self.agent = agent
        self.config = config
        self.results = []
        self.logger = logging.getLogger(self.__class__.__name__)
        self.run_dir = None

    @property
    def exp_name(self) -> str:
        """Get the Exp name (automatically inferred from the class name).

        Example: SolverExp -> Solver, CriticExp -> Critic.
        Subclasses can override this property to customize the name.
        """
        class_name = self.__class__.__name__
        # Remove the "Exp" suffix
        if class_name.endswith('Exp'):
            return class_name[:-3]
        return class_name

    def set_run_dir(self, run_dir: str | Path) -> None:
        """Set the run directory.

        Args:
            run_dir: Run directory path.
        """
        self.run_dir = Path(run_dir)

    def run(self, task_description: str, task_id: str = "exp_001", images: list[str] | None = None, on_step=None) -> dict:
        """Run a single experiment.

        Args:
            task_description: Task description.
            task_id: Task ID.
            images: List of image file paths (optional, for multimodal tasks).
            on_step: Step callback, signature (StepRecord, step_number, max_steps) -> None.

        Returns:
            Run result dictionary.
        """
        # Create a task instance
        task = TaskInstance(
            task_id=task_id,
            task_type="discovery",
            description=task_description,
            images=images or [],
        )

        # Run the Agent
        self.logger.debug(f"Running task: {task_id}")
        trajectory = self.agent.run(task, on_step=on_step)

        # Save results
        result = {
            "task_id": task_id,
            "status": trajectory.status,
            "steps": len(trajectory.steps),
            "trajectory": trajectory,
        }
        self.results.append(result)

        return {
            "trajectory": trajectory,
            "status": trajectory.status,
            "steps": len(trajectory.steps),
        }

    def save_results(self, output_file: str):
        """Save experiment results.

        Args:
            output_file: Output file path.
        """
        output_data = []
        for result in self.results:
            output_data.append({
                "task_id": result["task_id"],
                "status": result["status"],
                "steps": result["steps"],
                "trajectory": result["trajectory"].model_dump(),
            })

        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=2, default=str, ensure_ascii=False)

        self.logger.info(f"Results saved to {output_file}")


    def _extract_agent_response(self, trajectory: Any) -> str:
        """Extract the agent's final response from a trajectory.

        Args:
            trajectory: Execution trajectory.

        Returns:
            The agent's response text.
        """
        return extract_agent_response(trajectory)
