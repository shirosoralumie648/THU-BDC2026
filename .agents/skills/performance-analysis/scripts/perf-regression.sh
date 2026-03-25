#!/bin/bash
# Performance Analysis - Regression Check Script
# Check for performance regressions

set -e

BASELINE_FILE="${1:-baseline.json}"
CURRENT_FILE="current.json"
THRESHOLD="${2:-10}"

echo "Running current benchmarks..."
npx @claude-flow/cli performance benchmark \
  --suite all \
  --iterations 100 \
  --output "$CURRENT_FILE"

echo "Comparing against baseline..."
npx @claude-flow/cli performance benchmark \
  --compare "$BASELINE_FILE" "$CURRENT_FILE" \
  --threshold "$THRESHOLD"

rm "$CURRENT_FILE"
