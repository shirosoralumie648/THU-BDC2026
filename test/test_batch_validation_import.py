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
