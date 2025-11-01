import argparse
import json
import random
from pathlib import Path

import litellm
from terminal_bench.agents.terminus_2 import Terminus2
from terminal_bench.agents.base_agent import AgentResult
from terminal_bench.agents.failure_mode import FailureMode
from terminal_bench.terminal.tmux_session import TmuxSession
from dataset import Dataset, DatasetConfig

from gepa import optimize
from adapter import (
    TerminalBenchTask,
    Terminus2Adapter,
)

from terminal_bench.registry.client import RegistryClient



INSTRUCTION_PROMPT_PATH = "initial.txt"


class Terminus2Wrapper(Terminus2):
    def __init__(
        self,
        model_name: str,
        max_episodes: int = 50,
        parser_name: str = "json",
        api_base: str | None = None,
        temperature: float = 0.7,  ## Orig 0.7, various on 0.2
        **kwargs,
    ):
        # Check for optimized prompt file first (with _ suffix)
        original_path = Path(INSTRUCTION_PROMPT_PATH)
        optimized_path = original_path.parent / f"{original_path.stem}_{original_path.suffix}"

        # Use optimized prompt if it exists, otherwise fall back to original
        if optimized_path.exists():
            self.instruction_prompt = optimized_path.read_text()
        else:
            self.instruction_prompt = original_path.read_text()

        super().__init__(model_name, max_episodes, parser_name, api_base, temperature, **kwargs)

    def perform_task(
        self,
        instruction: str,
        session: TmuxSession,
        logging_dir: Path | None = None,
        time_limit_seconds: float | None = None,
    ):
        from terminal_bench.llms.chat import Chat

        chat = Chat(self._llm)

        # Get the base prompt template and format it
        base_prompt = self._prompt_template.format(
            instruction=instruction,
            terminal_state=self._limit_output_length(session.get_incremental_output()),
        )

        # Prepend the instruction prompt
        initial_prompt = self.instruction_prompt + "\n\n" + base_prompt

        self._run_agent_loop(initial_prompt, session, chat, logging_dir, instruction)

        return AgentResult(
            total_input_tokens=chat.total_input_tokens,
            total_output_tokens=chat.total_output_tokens,
            failure_mode=FailureMode.NONE,
            timestamped_markers=self._timestamped_markers,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="anthropic/claude-sonnet-4-5-20250929",
                        help="Model name to use for the agent")
    parser.add_argument("--api_base", type=str, default=None,
                        help="API base URL for custom providers (e.g., https://api.deepinfra.com/)")
    parser.add_argument("--n_concurrent", type=int, default=40,
                        help="Number of concurrent tasks to run")
    parser.add_argument("--task_directory", type=str, default=None,
                        help="Absolute path to the task directory")
    parser.add_argument("--train_size", type=int, default=7,
                        help="Number of tasks in training set")
    parser.add_argument("--val_size", type=int, default=8,
                        help="Number of tasks in validation set")
    parser.add_argument("--test_size", type=int, default=7,
                        help="Number of tasks in test set")
    parser.add_argument("--random_seed", type=int, default=42,
                        help="Random seed for reproducible task splitting")
    parser.add_argument("--output_dir", type=str, default="gepa_output",
                        help="Output directory for GEPA results and logs")
    parser.add_argument("--optimized_prompt_file", type=str, default="optimized.txt",
                        help="Filename to save the optimized prompt")
    parser.add_argument("--max_metric_calls", type=int, default=100,
                        help="Maximum number of metric evaluation calls during optimization")
    parser.add_argument("--reflection_minibatch_size", type=int, default=3,
                        help="Number of examples to use in each reflection minibatch")
    parser.add_argument("--perfect_score", type=float, default=1.0,
                        help="Perfect score threshold for optimization")
    parser.add_argument("--skip_perfect_score", action="store_true",
                        help="Skip early stopping when perfect score is reached")
    parser.add_argument("--use_wandb", action="store_true", default=True,
                        help="Enable Weights & Biases logging")
    args = parser.parse_args()


    initial_prompt_from_terminus = """
You are a data science agent. Perform proper data analysis, hypothesis testing, and visualization.

CRITICAL REQUIREMENTS:
1. You MUST generate the report in /app/report/report.md (you get ZERO points without it)
2. You MUST return valid JSON matching the EXACT format provided in the template
3. Return valid JSON even when you believe that other keys are not needed, you can have an empty array in commands, but it is necessary to have all required keys in the output ALWAYS

JSON FORMAT REQUIREMENTS (ALL fields are REQUIRED):
{
  "analysis": "string describing current terminal state and what you observe",
  "plan": "string explaining what commands will do and your strategy",
  "commands": [
    {
      "keystrokes": "command to execute (e.g., 'ls\\n')",
      "duration": 0.1
    }
  ],
  "task_complete": true/false
}

IMPORTANT:
- 'commands' must be an ARRAY (list), never a string
- ALL required fields must be present in EVERY response
- Don't skip any fields or you'll get validation errors
- Append \\n to commands that need execution
- Set duration based on expected command completion time (0.1 for quick, 1.0 for normal, higher for slow commands)
- Never exceed 60 seconds duration; use polling instead
"""

    if args.task_directory:
        terminal_bench_dataset = Dataset(path=Path(args.task_directory))
    else:
        terminal_bench_dataset = Dataset(name="terminal-bench-core", version="head")
    all_tasks = terminal_bench_dataset._tasks
    random.seed(args.random_seed)
    shuffled_tasks = all_tasks.copy()
    random.shuffle(shuffled_tasks)

    # Calculate total required tasks
    total_required = args.train_size + args.val_size + args.test_size

    # Verify we have enough tasks
    if len(shuffled_tasks) < total_required:
        raise ValueError(
            f"Not enough tasks in dataset. Required {total_required} "
            f"(train={args.train_size}, val={args.val_size}, test={args.test_size}), "
            f"but only {len(shuffled_tasks)} tasks available."
        )

    # Split tasks into train/val/test
    train_tasks = shuffled_tasks[:args.train_size]
    val_tasks = shuffled_tasks[args.train_size:args.train_size + args.val_size]
    test_tasks = shuffled_tasks[args.train_size + args.val_size:args.train_size + args.val_size + args.test_size]

    # Convert to TerminalBenchTask objects
    trainset = [
        TerminalBenchTask(task_id=task.name, model_name=args.model_name, api_base=args.api_base)
        for task in train_tasks
    ]
    valset = [
        TerminalBenchTask(task_id=task.name, model_name=args.model_name, api_base=args.api_base)
        for task in val_tasks
    ]
    testset = [
        TerminalBenchTask(task_id=task.name, model_name=args.model_name, api_base=args.api_base)
        for task in test_tasks
    ]

    # Print split information
    print(f"\n=== Dataset Split (seed={args.random_seed}) ===")
    print(f"Train set ({len(trainset)}): {[t.task_id for t in trainset]}")
    print(f"Val set ({len(valset)}): {[t.task_id for t in valset]}")
    print(f"Test set ({len(testset)}): {[t.task_id for t in testset]}")
    print(f"{'='*50}\n")

    reflection_lm = (
        lambda prompt: litellm.completion(
            model="openai/gpt-5",
            messages=[{"role": "user", "content": prompt}],
            reasoning_effort="medium",
        )
        .choices[0]
        .message.content
    )

    adapter = Terminus2Adapter(
        n_concurrent=args.n_concurrent,
        instruction_prompt_path=INSTRUCTION_PROMPT_PATH,
        dataset_path=args.task_directory
    )

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Evaluate testset with initial instruction prompt BEFORE optimization
    testset_results_before_opt = adapter.evaluate(
        testset,
        {"instruction_prompt": initial_prompt_from_terminus},
        capture_traces=True,
    )

    with open(f"{args.output_dir}/testset_results_before_opt.json", "w") as f:
        json.dump(
            {
                "score": sum(trajectory["success"] for trajectory in testset_results_before_opt.trajectories),
                "trajectories": testset_results_before_opt.trajectories,
            },
            f,
            indent=4,
        )

    optimized_results = optimize(
        seed_candidate={"instruction_prompt": initial_prompt_from_terminus},
        trainset=trainset,
        valset=valset,
        adapter=adapter,
        reflection_lm=reflection_lm,
        use_wandb=args.use_wandb,
        max_metric_calls=args.max_metric_calls,
        reflection_minibatch_size=args.reflection_minibatch_size,
        perfect_score=args.perfect_score,
        skip_perfect_score=args.skip_perfect_score,
        run_dir=args.output_dir,
    )

    with open(args.optimized_prompt_file, "w") as f:
        f.write(optimized_results.best_candidate["instruction_prompt"])
    print(f"Saved optimized prompt to {args.optimized_prompt_file}")

    testset_results_after_opt = adapter.evaluate(
        testset,
        {"instruction_prompt": optimized_results.best_candidate["instruction_prompt"]},
        capture_traces=True,
    )

    with open(f"{args.output_dir}/optimized_results.json", "w") as f:
        json.dump(
            {
                "score": sum(trajectory["success"] for trajectory in testset_results_after_opt.trajectories),
                "trajectories": testset_results_after_opt.trajectories,
            },
            f,
            indent=4,
        )
