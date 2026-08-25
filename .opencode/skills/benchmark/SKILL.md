---
name: benchmark
description: Runs all benchmarks in ./benchmarks and compares BENCHMARK*.md results against their corresponding BENCHMARK*_baseline.md files.
---

# Benchmark Analysis

Analyze how the current code changes affect the project's overall performance.

## Procedure

1. Verify that the `./benchmarks/` directory exists.
   - If it does not exist, stop and report a clear error.

2. Read the following files, if available:
   - `README.md`
   - `./benchmarks/README.md`
   - Relevant build and configuration files such as `package.json`, `Makefile`,
     `pyproject.toml`, `Cargo.toml`, `go.mod`, or equivalent files.

3. Determine how the benchmarks must be executed.

   Prefer, in this order:

   1. A documented benchmark command
   2. A dedicated benchmark runner script
   3. A benchmark target defined by the project's build system
   4. Individual executable benchmark files in `./benchmarks/`

4. Run all benchmarks associated with `./benchmarks/`.

   Requirements:

   - Use the project root as the working directory unless the project specifies otherwise.
   - Do not execute Markdown result files or baseline files.
   - Never overwrite files ending in `_baseline.md`.
   - Do not stop after the first failure; collect all benchmark failures.
   - Do not invent, estimate, or infer benchmark results that were not produced by the benchmark tools.
   - Record the command and exit status for each benchmark run.

5. After execution, recursively find all current result files matching:

   ```text
   ./BENCHMARK*.md
   ```

   Ignore every file whose name ends with `_baseline.md`.

6. Match each current result file with its corresponding baseline.

   Mapping rule:

   ```text
   <name>.md -> <name>_baseline.md
   ```

   Example:

   ```text
   BENCHMARK_api.md
   BENCHMARK_api_baseline.md
   ```

   The current result and its baseline must be located in the same directory
   unless the benchmark documentation explicitly defines another convention.

7. Compare each current result with its baseline semantically.

   Consider, when available:

   - Execution time
   - Latency
   - Throughput
   - Operations per second
   - CPU usage
   - Memory usage
   - Allocations
   - File or artifact size
   - Error rates
   - Scores
   - Number of iterations
   - Variance
   - Standard deviation
   - Confidence intervals

8. Determine the direction of each metric correctly.

   Usually, lower is better for:

   - Execution time
   - Latency
   - CPU usage
   - Memory usage
   - Allocations
   - Artifact size
   - Error rates

   Usually, higher is better for:

   - Throughput
   - Operations per second
   - Performance scores

   If the direction is ambiguous, classify the result as `unclear` rather than
   making an unsupported assumption.

9. For comparable numeric values, calculate:

   ```text
   absolute change = current - baseline
   relative change = ((current - baseline) / baseline) * 100
   ```

   Round percentages sensibly, normally to two decimal places.

   Do not calculate a relative percentage when the baseline value is zero.
   In that case, report only the absolute change and explain why the relative
   change is undefined.

10. Normalize units only when the conversion is unambiguous.

    Examples:

    - Seconds and milliseconds may be converted.
    - Bytes, KiB, MiB, and GiB may be converted with explicit binary units.
    - Do not silently compare incompatible or ambiguous units.

11. Classify every comparable metric as one of:

    - `improvement`
    - `regression`
    - `unchanged`
    - `unclear`
    - `not comparable`

12. Account for measurement noise.

    - Do not automatically treat very small differences as meaningful.
    - Use variance, standard deviation, confidence intervals, or documented
      tolerances when available.
    - If no statistical information or tolerance is available, explicitly state
      that small differences may be measurement noise.
    - Do not claim statistical significance unless the available data supports it.

13. Explicitly report all incomplete or invalid comparisons, including:

    - Current result without a baseline
    - Baseline without a current result
    - Failed benchmark execution
    - Missing metric
    - Invalid or non-numeric value
    - Incompatible units
    - Changed benchmark configuration
    - Different iteration counts that make results unreliable
    - Result format that cannot be interpreted safely

## Overall Assessment

Determine whether the overall performance:

- improved,
- regressed,
- remained effectively unchanged, or
- cannot be determined reliably.

Base this assessment on the importance and consistency of the available metrics.
Do not simply average unrelated percentages.

Highlight:

- The most critical regression
- The largest meaningful improvement
- Results that should be rerun because of noise or insufficient data
- Missing baselines or failed benchmarks that weaken the conclusion

## Final Report

Return a concise report using this structure:

# Benchmark Report

## Execution

| Benchmark | Command | Status | Notes |
|---|---|---|---|
| ... | ... | successful/failed | ... |

## Comparison

| Benchmark | Metric | Baseline | Current | Absolute Change | Relative Change | Assessment |
|---|---|---:|---:|---:|---:|---|
| ... | ... | ... | ... | ... | ... | improvement/regression/unchanged/unclear/not comparable |

## Missing or Invalid Results

List:

- Missing baselines
- Missing current results
- Failed benchmarks
- Invalid values
- Incompatible units
- Any other issue preventing a reliable comparison

If there are no issues, state:

```text
No missing or invalid results.
```

## Summary

- Improvements:
- Regressions:
- Unchanged metrics:
- Unclear or non-comparable metrics:
- Missing baselines:
- Failed benchmarks:

## Conclusion

Clearly answer:

1. Did overall performance improve, regress, remain effectively unchanged, or
   is the result inconclusive?
2. What is the most critical regression?
3. What is the largest meaningful improvement?
4. Which results should be rerun or investigated?

## Restrictions

- Do not modify application source code.
- Do not create, update, rename, or overwrite baseline files unless the user
  explicitly requests it.
- Do not compare a baseline file with itself.
- Do not run commands outside the repository.
- Do not install dependencies without explicit user approval.
- Do not hide failed benchmarks.
- Do not fabricate missing values.
- Report unsupported assumptions instead of presenting them as facts.
