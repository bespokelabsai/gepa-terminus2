# GEPA + Terminus 2

To run, first install gepa and terminal-bench:

`pip install terminal-bench`

`pip install gepa`

### Env exports
```
export ANTHROPIC_API_KEY=""
export OPENAI_API_KEY=""  # For the reflection model
export DEEPINFRA_API_KEY="" # In case you want to use an open source model on DeepInfra (e.g., model_name deepinfra/zai-org/GLM-4.6)
```

Training command example:

```bash
python train.py --model_name "anthropic/claude-sonnet-4-5-20250929" --train_size 7 --val_size 8 --test_size 7 --n_concurrent 10
```

### Optional Arguments

**Dataset Configuration:**
- `--task_directory`: Absolute path to custom task directory (default: uses Terminal Bench dataset)
- `--train_size`: Number of tasks in training set (default: 7)
- `--val_size`: Number of tasks in validation set (default: 8)
- `--test_size`: Number of tasks in test set (default: 7)
- `--random_seed`: Random seed for reproducible task splitting (default: 42)

**Model Configuration:**
- `--model_name`: Model name for the agent (default: anthropic/claude-sonnet-4-5-20250929)
- `--api_base`: API base URL for custom providers (e.g., https://api.deepinfra.com/)
- `--n_concurrent`: Number of concurrent tasks to run (default: 40)

**GEPA Optimization Parameters:**
- `--output_dir`: Output directory for GEPA results and logs (default: gepa_output)
- `--optimized_prompt_file`: Filename to save the optimized prompt (default: optimized.txt)
- `--max_metric_calls`: Maximum number of metric evaluation calls (default: 100)
- `--reflection_minibatch_size`: Number of examples in each reflection minibatch (default: 3)
- `--perfect_score`: Perfect score threshold for optimization (default: 1.0)
- `--skip_perfect_score`: Skip early stopping when perfect score is reached (default: false)
- `--use_wandb`: Enable Weights & Biases logging (default: true)

### Notes

- The reflection model is hardcoded to `openai/gpt-5` with `reasoning_effort="medium"`.
- `make_reflective_dataset` and `get_results` would need to be changed based on the specific judge. Currently, it is based on our task-specific judge, but the base implementations [here](https://github.com/gepa-ai/gepa/blob/main/src/gepa/adapters/terminal_bench_adapter/terminal_bench_adapter.py) should work.
- `initial.txt` contains the initial prompt that we want to optimize with GEPA, and the current prompt is rewritten to `initial_.txt` as the optimization continues.
- All gepa checkpoints are saved to the directory specified by `--output_dir` (default: `gepa_output/`).
- The agent runs are stored in runs/ directory