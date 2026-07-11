# Current Common50 Results

数据集：balanced common50，`safe=31`，`unsafe=19`。

## strict_agent

只把 `N::agentX` 算作 agent label。

| Method | Binary Acc | Safe F1 | Unsafe F1 | Agent P | Agent R | Agent F1 | Agent Hit | Top1 | Top2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| V8-SFT | 78.0 | 80.0 | 75.6 | 5.6 | 15.4 | 8.2 | 18.2 | N/A | N/A |
| V12-SFT | 76.0 | 79.3 | 71.4 | 38.5 | 38.5 | 38.5 | 36.4 | N/A | N/A |
| G-Safeguard | 70.0 | 79.5 | 44.4 | 35.3 | 46.2 | 40.0 | 45.5 | 36.4 | 72.7 |
| BlindGuard | 44.0 | 17.6 | 57.6 | 12.3 | 61.5 | 20.5 | 63.6 | 63.6 | 72.7 |

## agent_or_tool_owner

把 `N::agentX` 和 `T::agentX` 算作 agent label。

| Method | Binary Acc | Safe F1 | Unsafe F1 | Agent P | Agent R | Agent F1 | Agent Hit | Top1 | Top2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| V8-SFT | 78.0 | 80.0 | 75.6 | 8.3 | 21.4 | 12.0 | 25.0 | N/A | N/A |
| V12-SFT | 76.0 | 79.3 | 71.4 | 28.0 | 50.0 | 35.9 | 50.0 | N/A | N/A |
| G-Safeguard | 70.0 | 80.0 | 40.0 | 35.7 | 35.7 | 35.7 | 33.3 | 58.3 | 75.0 |
| BlindGuard | 42.0 | 17.1 | 55.4 | 13.1 | 57.1 | 21.3 | 58.3 | 58.3 | 66.7 |

## Takeaway

V12 is the best current trade-off: it is close to G-Safeguard on agent-level localization while still producing full MAS audit JSON. V8 is stronger on run-level binary judgment but much weaker on localization. GNN baselines are reported only on their native/shared metrics; unsupported audit metrics are `N/A`.

