import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class EntrypointScriptTests(unittest.TestCase):
    def _run_script(self, script_name, *args):
        env = os.environ.copy()
        env['THU_BDC_PYTHON_BIN'] = '/bin/echo'
        result = subprocess.run(
            ['sh', script_name, *args],
            cwd=PROJECT_ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        return result

    def test_train_wrapper_uses_configured_python_and_forwards_args(self):
        result = self._run_script('train.sh', '--dry-run')

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn('code/src/train.py --dry-run', result.stdout.strip())

    def test_predict_wrapper_uses_configured_python_and_forwards_args(self):
        result = self._run_script('test.sh', '--dry-run')

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn('code/src/predict.py --dry-run', result.stdout.strip())

    def test_train_wrapper_prefers_parent_repo_venv_in_worktree_layout(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            repo_root = tmp_root / 'repo'
            worktree_root = repo_root / '.worktrees' / 'stage-x'
            python_bin = repo_root / '.venv' / 'bin' / 'python'
            entrypoint = worktree_root / 'code' / 'src' / 'train.py'
            script_path = worktree_root / 'train.sh'

            python_bin.parent.mkdir(parents=True, exist_ok=True)
            entrypoint.parent.mkdir(parents=True, exist_ok=True)
            script_path.parent.mkdir(parents=True, exist_ok=True)

            python_bin.write_text('#!/bin/sh\necho "$0 $@"\n', encoding='utf-8')
            python_bin.chmod(0o755)
            entrypoint.write_text('print("placeholder")\n', encoding='utf-8')
            shutil.copy2(PROJECT_ROOT / 'train.sh', script_path)

            env = os.environ.copy()
            env.pop('THU_BDC_PYTHON_BIN', None)
            env['PATH'] = '/bin'
            result = subprocess.run(
                ['/bin/sh', str(script_path), '--dry-run'],
                cwd=worktree_root,
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, msg=result.stderr)
            self.assertIn(str(python_bin), result.stdout)
            self.assertIn('code/src/train.py --dry-run', result.stdout)

    def test_predict_wrapper_prefers_parent_repo_venv_in_worktree_layout(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            repo_root = tmp_root / 'repo'
            worktree_root = repo_root / '.worktrees' / 'stage-x'
            python_bin = repo_root / '.venv' / 'bin' / 'python'
            entrypoint = worktree_root / 'code' / 'src' / 'predict.py'
            script_path = worktree_root / 'test.sh'

            python_bin.parent.mkdir(parents=True, exist_ok=True)
            entrypoint.parent.mkdir(parents=True, exist_ok=True)
            script_path.parent.mkdir(parents=True, exist_ok=True)

            python_bin.write_text('#!/bin/sh\necho "$0 $@"\n', encoding='utf-8')
            python_bin.chmod(0o755)
            entrypoint.write_text('print("placeholder")\n', encoding='utf-8')
            shutil.copy2(PROJECT_ROOT / 'test.sh', script_path)

            env = os.environ.copy()
            env.pop('THU_BDC_PYTHON_BIN', None)
            env['PATH'] = '/nonexistent'
            result = subprocess.run(
                ['/bin/sh', str(script_path), '--dry-run'],
                cwd=worktree_root,
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, msg=result.stderr)
            self.assertIn(str(python_bin), result.stdout)
            self.assertIn('code/src/predict.py --dry-run', result.stdout)

    def test_predict_wrapper_prefers_parent_repo_venv_in_worktree_layout(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            repo_root = tmp_root / 'repo'
            worktree_root = repo_root / '.worktrees' / 'stage-x'
            python_bin = repo_root / '.venv' / 'bin' / 'python'
            entrypoint = worktree_root / 'code' / 'src' / 'predict.py'
            script_path = worktree_root / 'test.sh'

            python_bin.parent.mkdir(parents=True, exist_ok=True)
            entrypoint.parent.mkdir(parents=True, exist_ok=True)
            script_path.parent.mkdir(parents=True, exist_ok=True)

            python_bin.write_text('#!/bin/sh\necho "$0 $@"\n', encoding='utf-8')
            python_bin.chmod(0o755)
            entrypoint.write_text('print("placeholder")\n', encoding='utf-8')
            shutil.copy2(PROJECT_ROOT / 'test.sh', script_path)

            env = os.environ.copy()
            env.pop('THU_BDC_PYTHON_BIN', None)
            env['PATH'] = '/nonexistent'
            result = subprocess.run(
                ['/bin/sh', str(script_path), '--dry-run'],
                cwd=worktree_root,
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, msg=result.stderr)
            self.assertIn(str(python_bin), result.stdout)
            self.assertIn('code/src/predict.py --dry-run', result.stdout)


if __name__ == '__main__':
    unittest.main()
