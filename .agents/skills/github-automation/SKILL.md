---
name: github-automation
description: >
  GitHub workflow automation including PR management, CI/CD, issue tracking, and release management. Integrates with GitHub CLI for seamless automation.
  Use when: creating pull requests, setting up CI/CD pipelines, release management, issue tracking automation, branch management, code review workflows, repository maintenance.
  Skip when: local-only development, prototyping without commits, non-GitHub repositories, offline work.
---

# Github Automation Skill

## Purpose
GitHub workflow automation including PR management, CI/CD, issue tracking, and release management. Integrates with GitHub CLI for seamless automation.

## When to Trigger
- creating pull requests
- setting up CI/CD pipelines
- release management
- issue tracking automation
- branch management
- code review workflows
- repository maintenance

## When to Skip
- local-only development
- prototyping without commits
- non-GitHub repositories
- offline work

## Commands

### Create Pull Request
Create a new pull request with summary

```bash
gh pr create --title "[title]" --body "[description]"
```

**Example:**
```bash
gh pr create --title "feat: add OAuth2 authentication" --body "## Summary\n- Implemented OAuth2 flow\n- Added token refresh\n\n## Test Plan\n- Run auth tests"
```

### View PR
View pull request details

```bash
gh pr view [number]
```

**Example:**
```bash
gh pr view 123 --comments
```

### Merge PR
Merge a pull request

```bash
gh pr merge [number] --squash
```

**Example:**
```bash
gh pr merge 123 --squash --delete-branch
```

### Run Workflow
Trigger a GitHub Actions workflow

```bash
gh workflow run [workflow]
```

**Example:**
```bash
gh workflow run ci.yml --ref feature-branch
```

### View Workflow Runs
List recent workflow runs

```bash
gh run list --limit 10
```

### Create Issue
Create a new issue

```bash
gh issue create --title "[title]" --body "[body]"
```

**Example:**
```bash
gh issue create --title "Bug: login fails on mobile" --body "## Description\n..." --label bug
```

### Create Release
Create a new release

```bash
gh release create [tag] --notes "[notes]"
```

**Example:**
```bash
gh release create v1.0.0 --notes "Initial release" --generate-notes
```

### View Checks
View PR check status

```bash
gh pr checks [number]
```

**Example:**
```bash
gh pr checks 123 --watch
```

### Review PR
Submit a PR review

```bash
gh pr review [number] --approve --body "[comment]"
```

**Example:**
```bash
gh pr review 123 --approve --body "LGTM! Great work on the tests."
```


## Scripts

| Script | Path | Description |
|--------|------|-------------|
| `pr-template` | `.agents/scripts/pr-template.sh` | Generate PR from template |
| `release-prep` | `.agents/scripts/release-prep.sh` | Prepare release with changelog |


## References

| Document | Path | Description |
|----------|------|-------------|
| `GitHub CLI Reference` | `docs/gh-cli.md` | GitHub CLI command reference |
| `PR Guidelines` | `docs/pr-guidelines.md` | Pull request best practices |
| `CI/CD Setup` | `docs/ci-cd.md` | CI/CD pipeline configuration |

## Best Practices
1. Check memory for existing patterns before starting
2. Use hierarchical topology for coordination
3. Store successful patterns after completion
4. Document any new learnings
