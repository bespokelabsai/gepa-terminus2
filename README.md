# GEPA + Terminus 2

To run, first install gepa and terminal-bench
Training command example :

```
python train2.py --model_name "anthropic/claude-sonnet-4-5-20250929" --task_directory "/home/shivank/work/terminal-bench-replay/tasks_judge" --train_size 7 --val_size 8 --test_size 7 --n_concurrent 10
```

Task directory --> Directory with terminal bench tasks if none takes the [terminal bench tasks dataset](https://github.com/laude-institute/terminal-bench/tree/main/tasks) by default

**Note:**
`make_reflective_dataset` and `get_results` would need to be changed based on the specific judge (Currently its based on our task specific judge, but the base implementations [here](https://github.com/gepa-ai/gepa/blob/main/src/gepa/adapters/terminal_bench_adapter/terminal_bench_adapter.py) should work)