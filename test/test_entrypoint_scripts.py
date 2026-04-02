import os
import subprocess
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


if __name__ == '__main__':
    unittest.main()
