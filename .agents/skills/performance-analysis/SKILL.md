---
name: performance-analysis
description: >
  Performance profiling, benchmarking, and optimization. Includes CPU profiling, memory analysis, latency measurement, and automated optimization suggestions.
  Use when: slow operations detected, memory usage issues, optimization needed, pre-release performance validation, database query optimization, API latency concerns, bundle size analysis.
  Skip when: early feature development, documentation updates, prototyping phase, configuration changes.
---

# Performance Analysis Skill

## Purpose
Performance profiling, benchmarking, and optimization. Includes CPU profiling, memory analysis, latency measurement, and automated optimization suggestions.

## When to Trigger
- slow operations detected
- memory usage issues
- optimization needed
- pre-release performance validation
- database query optimization
- API latency concerns
- bundle size analysis

## When to Skip
- early feature development
- documentation updates
- prototyping phase
- configuration changes

## Commands

### Run Benchmark Suite
Execute all performance benchmarks

```bash
npx @claude-flow/cli performance benchmark --suite all
```

**Example:**
```bash
npx @claude-flow/cli performance benchmark --suite all --iterations 100 --output bench-results.json
```

### Profile Code
Profile code execution for CPU and memory

```bash
npx @claude-flow/cli performance profile --target ./src
```

**Example:**
```bash
npx @claude-flow/cli performance profile --target ./src/api --duration 60s
```

### Memory Analysis
Analyze memory usage patterns

```bash
npx @claude-flow/cli performance metrics --metric memory
```

**Example:**
```bash
npx @claude-flow/cli performance metrics --metric memory --threshold 100MB
```

### Latency Analysis
Measure and analyze latency

```bash
npx @claude-flow/cli performance metrics --metric latency
```

### Optimize Suggestions
Get automated optimization suggestions

```bash
npx @claude-flow/cli performance optimize --analyze
```

**Example:**
```bash
npx @claude-flow/cli performance optimize --analyze --apply-safe
```

### Performance Report
Generate performance report

```bash
npx @claude-flow/cli performance report
```

**Example:**
```bash
npx @claude-flow/cli performance report --format html --output perf-report.html
```

### Compare Benchmarks
Compare benchmark results

```bash
npx @claude-flow/cli performance benchmark --compare baseline.json current.json
```

### WASM Benchmark
Run WASM-specific benchmarks

```bash
npx @claude-flow/cli performance benchmark --suite wasm
```


## Scripts

| Script | Path | Description |
|--------|------|-------------|
| `perf-baseline` | `.agents/scripts/perf-baseline.sh` | Capture performance baseline |
| `perf-regression` | `.agents/scripts/perf-regression.sh` | Check for performance regressions |


## References

| Document | Path | Description |
|----------|------|-------------|
| `Performance Guide` | `docs/performance.md` | Performance optimization guide |
| `Benchmark Reference` | `docs/benchmarks.md` | Benchmark configuration reference |

## Best Practices
1. Check memory for existing patterns before starting
2. Use hierarchical topology for coordination
3. Store successful patterns after completion
4. Document any new learnings
