import json
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel

from gepa import EvaluationBatch, GEPAAdapter


class TerminalBenchTask(BaseModel):
    task_id: str
    model_name: str
    api_base: str | None = None


def run_agent_tb(
    task_ids: str | list[str],
    run_id: str,
    model_name: str,
    instruction_prompt: str,
    dataset_name: str = "terminal-bench-core",
    dataset_version: str = "head",
    dataset_path: str | None = None,
    agent_import_path: str = "train:Terminus2Wrapper",
    n_concurrent: int = 6,
    prompt_template_path: str = "prompt-templates/instruction_prompt.txt",
    api_base: str | None = None,
):
    """Run the replay agent for multiple task IDs using tb run command."""

    env = os.environ.copy()
    original_path = Path(prompt_template_path)
    stem = original_path.stem 
    suffix = original_path.suffix  
    parent = original_path.parent  
    new_filename = f"{stem}_{suffix}"
    actual_prompt_path = parent / new_filename
    with open(actual_prompt_path, "w") as f:
        f.write(instruction_prompt)

    cmd = [
        "tb",
        "run",
    ]

    # Use either dataset path or dataset name/version
    if dataset_path:
        cmd.extend(["--dataset-path", dataset_path])
    else:
        cmd.extend([
            "--dataset-name",
            dataset_name,
            "--dataset-version",
            dataset_version,
        ])

    cmd.extend([
        "--agent-import-path",
        agent_import_path,
        "--model",
        model_name,
        "--run-id",
        run_id,
        "--n-concurrent",
        str(n_concurrent),
        "--output-path",
        str(Path(os.getcwd()) / "runs"),
        "--global-timeout-multiplier",
        str(10)
    ])

    # Add api_base if provided
    if api_base is not None:
        cmd.extend(["--api-base", api_base])

    if isinstance(task_ids, list):
        for task_id in task_ids:
            cmd.extend(["--task-id", task_id])
    else:
        cmd.extend(["--task-id", task_ids])

    print(f"Running command: {' '.join(cmd)}")

    try:
        # Use Popen for better control over stdout/stderr and to avoid blocking
        import sys
        process = subprocess.Popen(
            cmd,
            env=env,
            cwd=os.getcwd(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Merge stderr into stdout
            text=True,
            bufsize=1  # Line buffered
        )

        # Read output line by line to prevent buffer overflow and show progress
        for line in process.stdout:
            print(line, end='')
            sys.stdout.flush()

        # Wait for process to complete
        return_code = process.wait()

        if return_code != 0:
            print(f"Command failed with return code: {return_code}")
            return return_code

        print(f"Command completed successfully with return code: {return_code}")
        return return_code

    except subprocess.CalledProcessError as e:
        print(f"Command failed with return code: {e.returncode}")
        return e.returncode
    except Exception as e:
        print(f"Error running command: {e}")
        return 1


def get_results(task_id: str, run_id: str) -> tuple[bool, float, str, list]:

    def _read_episode_response(episode_dir: Path) -> str | None:
        """Helper method to read response.txt from an episode directory."""
        response_file = episode_dir / "response.txt"
        if response_file.exists():
            try:
                return response_file.read_text()
            except Exception:
                pass
        return None

    def _get_logging_dir(task_id: str, run_id: str):
        logging_dir_base = Path("runs") / run_id / task_id
        for dir in logging_dir_base.iterdir():
            if dir.is_dir() and dir.name.startswith(task_id):
                return dir
        raise ValueError(
            f"No logging directory found for task {task_id} and run {run_id}"
        )

    def _extract_judge_score_from_tests_log(logging_dir: Path) -> float:
        """Threshold based judge is used
        """
        tests_log_path = logging_dir / "sessions" / "tests.log"
        if not tests_log_path.exists():
            print(f"Warning: tests.log not found at {tests_log_path}")
            return 0.0

        try:
            with open(tests_log_path, 'r') as f:
                content = f.read()

            # Extract ALL <judge_score> tags (should be 2: report score + reproducibility score)
            score_matches = re.findall(r'<judge_score>([\d.]+)</judge_score>', content)

            if len(score_matches) == 2:
                report_score = float(score_matches[0])
                reproducibility_score = float(score_matches[1])
                final_score = report_score if reproducibility_score > 0.95 else 0.0
                print(f"  - Final score (report if repro > 0.95 else 0): {final_score:.4f}")

                return final_score

            elif len(score_matches) == 1:
                # Fallback: only one score found (old format or partial execution)
                judge_score = float(score_matches[0])
                print(f"Warning: Only 1 judge_score found (expected 2) for {logging_dir.name}, using: {judge_score:.4f}")
                return judge_score

            else:
                # No scores found, try to calculate from grade JSON as fallback
                print(f"Warning: Expected 2 judge_score tags but found {len(score_matches)} for {logging_dir.name}")

                grade_match = re.search(r'<grade>\s*(\{[\s\S]*?\})\s*</grade>', content)
                if grade_match:
                    try:
                        grade_json = grade_match.group(1).strip()
                        grade_data = json.loads(grade_json)

                        # Calculate average score from applicable criteria
                        applicable_scores = [
                            criterion['score']
                            for criterion in grade_data.values()
                            if isinstance(criterion, dict) and criterion.get('is_applicable', True)
                        ]

                        if applicable_scores:
                            judge_score = sum(applicable_scores) / len(applicable_scores)
                            return judge_score
                    except (json.JSONDecodeError, Exception) as e:
                        print(f"Warning: Failed to calculate score from grade JSON: {e}")

                print(f"Warning: No judge_score or grade data found in tests.log for {logging_dir.name}")
                return 0.0

        except Exception as e:
            print(f"Error extracting judge_score from tests.log: {e}")
            return 0.0

    logging_dir = _get_logging_dir(task_id, run_id)
    result_json = logging_dir / "results.json"
    with open(result_json) as f:
        result = json.load(f)

    parser_results = result.get("parser_results", {})
    test_grade_report = parser_results.get("test_grade_report", "failed")
    success = test_grade_report == "passed"
    score = _extract_judge_score_from_tests_log(logging_dir)

    failed_reason = result.get("failure_mode", "unknown") if not success else "none"

    trajectory_path = logging_dir / "agent-logs"
    episode_dirs = []
    for dir in trajectory_path.iterdir():
        if dir.is_dir() and dir.name.startswith("episode-"):
            episode_dirs.append(dir)

    if episode_dirs:
        # Sort by episode number to get the last one
        episode_dirs.sort(key=lambda x: int(x.name.split("-")[1]))
        last_episode_dir = episode_dirs[-1]

    last_episode_dir_trajectory = last_episode_dir / "debug.json"
    with open(last_episode_dir_trajectory) as f:
        trajectory = json.load(f)

        if "input" in trajectory and isinstance(trajectory["input"], list):
            messages = trajectory["input"]
        response_text = _read_episode_response(last_episode_dir)

        if response_text:
            assistant_message = {
                "role": "assistant",
                "content": response_text,
            }
            messages.append(assistant_message)

    return success, score, failed_reason, messages


class Terminus2Adapter(GEPAAdapter):

    def __init__(
        self,
        n_concurrent: int = 6,
        instruction_prompt_path: str = "prompt-templates/instruction_prompt.txt",
        dataset_path: str | None = None,
    ):
        self.n_concurrent = n_concurrent
        self.instruction_prompt_path = instruction_prompt_path
        self.dataset_path = dataset_path

    def evaluate(
        self,
        batch: list[TerminalBenchTask],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch:
        outputs = []
        scores = []
        trajectories = []
        example_run_id = "temp_gepa_run2" + "_" + datetime.now().strftime("%Y%m%d%H%M%S")
        example_model_name = batch[0].model_name
        example_api_base = batch[0].api_base  # Get api_base from first task

        run_agent_tb(
            [task.task_id for task in batch],
            example_run_id,
            example_model_name,
            instruction_prompt=candidate["instruction_prompt"],
            dataset_path=self.dataset_path,
            n_concurrent=self.n_concurrent,
            prompt_template_path=self.instruction_prompt_path,
            api_base=example_api_base,
        )

        for example in batch:
            try:
                success, score, failed_reason, messages = get_results(
                    example.task_id, example_run_id
                )
            except Exception as e:
                print(f"Error running example {example.task_id} {example_run_id}: {e}")
                success = False
                score = 0
                failed_reason = str(e)
                messages = []

            outputs.append(
                f"Terminal Bench outputs are omitted. Please see runs/{example_run_id}/{example.task_id}/ for detailed logging."
            )
            scores.append(score)
            trajectories.append(
                {
                    "messages": messages,
                    "instruction_prompt": candidate["instruction_prompt"],
                    "failed_reason": failed_reason,
                    "success": success,
                }
            )
        return EvaluationBatch(
            outputs=outputs,
            scores=scores,
            trajectories=trajectories,
        )

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch,
        components_to_update: list[str],
    ):
        reflective_dataset = {"instruction_prompt": []}
        for score, trajectory in zip(eval_batch.scores, eval_batch.trajectories, strict=False):
            if score == 0:
                feedback = f"Weak or wrong reproducibility. Scored {score:.2f}/1.0."
            elif score >= 0.65:
                feedback = f"Task passed. Evaluation score = {score:.2f}/1.0."
            else:
                feedback = f"Task failed. Evaluation score = {score:.2f}/1.0."

            reflective_dataset["instruction_prompt"].append(
                {
                    "Message History": trajectory["messages"],
                    "Instruction Prompt": candidate["instruction_prompt"],
                    "Feedback": feedback,
                }
            )
        return reflective_dataset
