#!/bin/bash
# Performance Analysis - Baseline Script
# Capture performance baseline

set -e

BASELINE_FILE="${1:-baseline.json}"

echo "Capturing performance baseline..."
npx @claude-flow/cli performance benchmark \
  --suite all \
  --iterations 100 \
  --output "$BASELINE_FILE"

echo "Baseline saved to $BASELINE_FILE"
