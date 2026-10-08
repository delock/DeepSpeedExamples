# Model-tensor CPU offload

Pin vs pageable step time for ZeRO CPU offload of parameters and/or optimizer
states. Default `--zero-stage 3` (param + optimizer). `--zero-stage 1` or `2`
offloads the optimizer only.

See the [parent README](../README.md).

H200 (ZeRO-3 param+optim, pin on/off):

```bash
ulimit -l unlimited
python bench.py --model Qwen/Qwen3-8B --batch 4 --seq 512
python bench.py --model Qwen/Qwen3.5-9B --batch 4 --seq 512
```

Use `Qwen/Qwen3-4B` if host RAM is under ~256 GB. Qwen3.5 is instantiated from the text config when the hub card is multimodal. MoE/FP8 ids are rejected.
