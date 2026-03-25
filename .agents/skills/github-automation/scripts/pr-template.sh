#!/bin/bash
# GitHub Automation - PR Template Script
# Generate PR from template

set -e

TITLE="${1:-Update}"
BRANCH=$(git rev-parse --abbrev-ref HEAD)

echo "Creating PR for branch: $BRANCH"

gh pr create \
  --title "$TITLE" \
  --body "$(cat <<EOF
## Summary
<!-- Describe your changes -->

## Test Plan
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Documentation updated
- [ ] No breaking changes

Generated with claude-flow
EOF
)"

echo "PR created successfully"
