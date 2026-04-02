# Week 1 Stabilization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stabilize the current contest delivery pipeline by isolating test-side effects, documenting the new factor-graph/dataset-build path, and verifying the real local data flow up to reproducible dataset artifacts.

**Architecture:** Keep the existing Python ranking pipeline intact and avoid broad refactors. Contain immediate risk by removing top-level execution from Docker validation scripts, make the YAML-driven data/factor path the documented primary workflow, and verify the new path with local commands against current repo data.

**Tech Stack:** Python 3.12, unittest, argparse, pandas, existing CLI scripts, Markdown docs

---

## File Map

- [`test/test.py`](/home/shirosora/THU-BDC2026/test/test.py)
  Linux Docker batch validation script. Must stop executing on import so `unittest discover` stays clean.
- [`test/test_windows.py`](/home/shirosora/THU-BDC2026/test/test_windows.py)
  Windows Docker batch validation script. Already has a `main()` guard; use as the reference shape.
- [`test/test_batch_validation_import.py`](/home/shirosora/THU-BDC2026/test/test_batch_validation_import.py)
  New regression test proving importing the batch validation script has no side effects.
- [`README.md`](/home/shirosora/THU-BDC2026/README.md)
  Primary project documentation. Needs one canonical “recommended workflow” that includes pipeline config validation, factor graph build, dataset build, train, and predict.
- [`GUIDE.md`](/home/shirosora/THU-BDC2026/GUIDE.md)
  Operator guide. Needs the same workflow reflected in task-oriented language.
- [`docs/superpowers/plans/2026-04-02-week1-stabilization.md`](/home/shirosora/THU-BDC2026/docs/superpowers/plans/2026-04-02-week1-stabilization.md)
  This execution plan.

## Task 1: Stop `unittest discover` from executing Docker validation side effects

**Files:**
- Create: `test/test_batch_validation_import.py`
- Modify: `test/test.py`
- Reference: `test/test_windows.py`

- [ ] **Step 1: Write the failing regression test**

```python
import os
import subprocess
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class BatchValidationImportTests(unittest.TestCase):
    def test_importing_linux_batch_validation_script_has_no_runtime_side_effects(self):
        command = [
            sys.executable,
            '-c',
            'import importlib; importlib.import_module("test.test")',
        ]
        result = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertNotIn('tar_files已成功从文件中读取', result.stdout)
        self.assertNotIn('docker load -i', result.stdout)


if __name__ == '__main__':
    unittest.main()
```

- [ ] **Step 2: Run the regression test to verify it fails for the right reason**

Run:
```bash
./.venv/bin/python -m unittest test.test_batch_validation_import -v
```

Expected:
- `FAIL`
- stdout still contains `tar_files已成功从文件中读取` because `test/test.py` executes on import

- [ ] **Step 3: Refactor `test/test.py` so import is side-effect free**

Target structure:
```python
from pathlib import Path


def read_tar_files(input_file: Path) -> list[str]:
    with open(input_file, 'r', encoding='utf-8') as file:
        return [line.strip() for line in file.readlines() if line.strip()]


def main() -> None:
    input_file = Path('./test/tar_files_list.txt')
    tar_files = read_tar_files(input_file)

    print('tar_files已成功从文件中读取：')
    print(tar_files)

    for tar_file in tar_files:
        ...


if __name__ == '__main__':
    main()
```

Rules:
- Do not change the operational Docker logic yet.
- Only move top-level execution into `main()` and keep behavior the same when run as a script.
- Keep existing helper functions intact unless a small rename or signature cleanup is required.

- [ ] **Step 4: Re-run the regression test and targeted suite**

Run:
```bash
./.venv/bin/python -m unittest test.test_batch_validation_import -v
./.venv/bin/python -m unittest discover -s test -p 'test_*.py' -v
```

Expected:
- regression test `PASS`
- no Docker batch execution noise during discovery
- existing factor/factor-graph/hf tests remain green

- [ ] **Step 5: Commit**

```bash
git add test/test.py test/test_batch_validation_import.py
git commit -m "test: isolate docker batch validation script import side effects"
```

## Task 2: Make the YAML-driven build flow the documented primary workflow

**Files:**
- Modify: `README.md`
- Modify: `GUIDE.md`

- [ ] **Step 1: Update `README.md` workflow order**

Required content to add or rewrite:
```md
## 推荐执行流程

1. 校验 pipeline 配置
```bash
python code/src/manage_data.py validate-pipeline-config --config-dir ./config
```

2. 构建宽表因子
```bash
python code/src/manage_data.py build-factor-graph \
  --pipeline-config-dir ./config \
  --feature-set-version v1
```

3. 构建训练/测试集
```bash
python code/src/manage_data.py build-dataset \
  --pipeline-config-dir ./config \
  --feature-set-version v1
```

4. 训练模型
```bash
sh train.sh
```

5. 生成预测结果
```bash
sh test.sh
```
```

Also add one short note stating:
- legacy `data/train.csv` / `data/test.csv` remains compatible
- the preferred path is now `pipeline config -> factor graph -> dataset build -> train/predict`

- [ ] **Step 2: Update `GUIDE.md` to match the same operator sequence**

Required content to add or rewrite:
```md
# 推荐流程（当前默认）

1. 校验配置
2. 构建因子宽表
3. 构建训练集/测试集
4. 训练
5. 预测
6. 自评与 Docker 验证
```

Rules:
- Keep the screenshot-heavy style, but make the ordered flow unambiguous.
- Do not remove legacy instructions; demote them to compatibility or fallback notes.

- [ ] **Step 3: Verify docs mention the new flow consistently**

Run:
```bash
rg -n "validate-pipeline-config|build-factor-graph|build-dataset|train.sh|test.sh" README.md GUIDE.md
```

Expected:
- both docs mention all five commands
- neither doc presents direct `train.sh` as the only primary path anymore

- [ ] **Step 4: Commit**

```bash
git add README.md GUIDE.md
git commit -m "docs: promote yaml-driven factor and dataset build workflow"
```

## Task 3: Verify the real local data path with current repo data

**Files:**
- Read/Verify: `config/datasets.yaml`
- Read/Verify: `config/factors.yaml`
- Read/Verify: `config/storage.yaml`
- Read/Verify: `data/stock_data.csv`
- Output: `data/datasets/features/train_features_v1.csv` or rendered equivalent
- Output: `data/train.csv`
- Output: `data/test.csv`
- Output: `data/data_manifest_dataset_build.json`

- [ ] **Step 1: Verify config validity before build commands**

Run:
```bash
./.venv/bin/python code/src/manage_data.py validate-pipeline-config --config-dir ./config
```

Expected:
- exit code `0`
- output contains `pipeline 配置校验: valid=True`

- [ ] **Step 2: Build factor graph on current local data**

Run:
```bash
./.venv/bin/python code/src/manage_data.py build-factor-graph \
  --pipeline-config-dir ./config \
  --feature-set-version v1 \
  --base-input ./data/stock_data.csv
```

Expected:
- factor build completes without traceback
- output path is printed
- manifest includes `factor_fingerprint`

- [ ] **Step 3: Build dataset from current local data and factor output**

Run:
```bash
./.venv/bin/python code/src/manage_data.py build-dataset \
  --pipeline-config-dir ./config \
  --feature-set-version v1 \
  --base-input ./data/stock_data.csv \
  --train-start 2015-01-01 \
  --train-end 2026-03-06 \
  --test-start 2026-03-09 \
  --test-end 2026-03-13
```

Expected:
- `data/train.csv` and `data/test.csv` are regenerated
- `data/data_manifest_dataset_build.json` exists
- manifest records `build_id`, `feature_set_version`, and `factor_fingerprint`

- [ ] **Step 4: Spot-check the generated manifest and output sizes**

Run:
```bash
ls -lh data/train.csv data/test.csv data/data_manifest_dataset_build.json
python - <<'PY'
import json
from pathlib import Path
path = Path('data/data_manifest_dataset_build.json')
payload = json.loads(path.read_text(encoding='utf-8'))
print(payload['build_id'])
print(payload['feature_set_version'])
print(payload['factor_fingerprint'])
print(payload['stats'])
PY
```

Expected:
- all files exist
- manifest fields are populated
- row counts are non-zero

- [ ] **Step 5: Commit**

```bash
git add data/data_manifest_dataset_build.json
# Add generated dataset artifacts only if they are intended to stay versioned in this repository.
git commit -m "chore: verify yaml-driven factor and dataset pipeline locally"
```

## Self-Review

- Spec coverage:
  - Week-1 stabilization asks for test-boundary cleanup, workflow documentation, and real-path verification. Tasks 1-3 map directly to those items.
- Placeholder scan:
  - No `TODO` or `TBD` markers remain.
- Type consistency:
  - The new regression test imports `test.test`, and Task 1 explicitly restructures `test/test.py` around a `main()` guard to satisfy that contract.
