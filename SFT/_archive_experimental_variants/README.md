# Archived Experimental Variants

这些版本是 V12 之后的探索版，不作为当前主线结果使用。

- `auditor_agent_sft_v12_gnn_distill_package`
- `auditor_agent_sft_v12_gnn_strong_distill_package`
- `auditor_agent_sft_v12_hq_package`
- `auditor_agent_sft_v12_hq_final_package`

归档原因：这些版本尝试引入 GNN distillation 或额外 high-quality filtering，但在 common50 对比中没有稳定超过 V12 主版本，且部分版本降低了 run-level 判别或引入定位噪声。

当前主线请看：

```text
../auditor_agent_sft_v12_package
../auditor_agent_gnn_baselines_package
```

