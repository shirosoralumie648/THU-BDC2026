#!/bin/bash
# GitHub Automation - Release Prep Script
# Prepare release with changelog

set -e

VERSION="${1:-patch}"

echo "Preparing release..."

# Bump version
npm version "$VERSION" --no-git-tag-version

NEW_VERSION=$(node -p "require('./package.json').version")

echo "Creating release v$NEW_VERSION..."
gh release create "v$NEW_VERSION" \
  --generate-notes \
  --draft

echo "Draft release v$NEW_VERSION created"
