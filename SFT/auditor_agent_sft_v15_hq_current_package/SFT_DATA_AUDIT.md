# V15-HQ Data Audit

## 结论

V15-HQ 已按 V14-HQ 标准重构：

```text
schema: Graph-grounded-Candidate-SFT/v12
all:   2904
train: 2319
test:   585
```

标签分布：

```text
clean_safe:      777
attack_failed:  647
attack_success: 1480
```

## 泄露检查

全量 SFT 可见 messages 中未发现：

```text
ACI_*
END_NEGOTIATION
attack_id
success_marker / success_markers
attack_metadata
private_user_information
Injected instruction / Injected profile / Injected message
Benchmark addendum
offline verifier / attack-success index
```

结果：

```text
leak_check all/train/test: {}
```

## 结构检查

```text
invalid evidence refs: 0
invalid component refs: 0
invalid localization component ids: 0
manual quality sample 50 problem counts: {}
```

## 筛选原则

保留：

```text
1. attacked 且存在 marker row-level signal
2. attack_success 且能抽到可见输出中的脱敏成功证据片段
3. clean
4. private_control 且无 natural marker leakage
```

排除：

```text
attacked missing marker signal:            534
private_control natural marker leakage:     68
private_control missing signal:            165
no visible run evidence:                  1071
attack_success missing visible HQ snippet:  49
```

## 重要限制

同事 zip 中：

```text
data/trajectories/*.jsonl 全部为空
research marble_results 也为空
```

因此本版是“基于 MARBLE result 可见输出”的高质量 SFT 数据，不是完整 trajectory 版本。
