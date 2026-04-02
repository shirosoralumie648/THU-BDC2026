import os
import sys
import unittest


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC_ROOT = os.path.join(PROJECT_ROOT, 'code', 'src')
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)


class IngestionImportTests(unittest.TestCase):
    def test_ingestion_package_exports(self):
        import ingestion

        self.assertTrue(hasattr(ingestion, '__all__'))
        self.assertIn('IngestionService', ingestion.__all__)

    def test_api_package_exports(self):
        import ingestion.api

        self.assertTrue(hasattr(ingestion.api, '__all__'))
        self.assertIn('create_app', ingestion.api.__all__)


if __name__ == '__main__':
    unittest.main()
