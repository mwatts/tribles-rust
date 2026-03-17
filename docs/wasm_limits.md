# WASM Formatter Limits

`WasmLimits` defines the resource caps used by `WasmValueFormatter` when running
value formatter modules. The defaults are a stable contract within a major
version:

- `max_memory_pages`: 8
- `max_fuel`: 5_000_000
- `max_output_bytes`: 8 * 1024

`WasmValueFormatter::format_value` uses the defaults. Use
`format_value_with_limits` if you need to tighten or loosen the caps for a
specific workload.
