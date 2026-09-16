# Fable Premium Long-Trace Report

Raw traces are not truncated. Extra-long traces are excluded from the 1M candidate unless a future reviewed segmentation pass proves turn-boundary and tool-pair integrity.

- rows inspected: `6,365`
- invalid trace rows: `210,292`
- validation errors: `{"assistant_missing": 248, "tool_without_preceding_assistant": 206394, "trace_ends_with_tool_result": 3650}`

| Bucket | Rows |
| --- | ---: |
| `short` | 1,306 |
| `medium` | 50 |
| `long` | 4,729 |
| `extra_long` | 280 |

- segmentation states: `{"long_trace_unsegmented": 5009}`
