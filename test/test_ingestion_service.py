import os
import sys
import tempfile
import unittest

import pandas as pd


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC_ROOT = os.path.join(PROJECT_ROOT, 'code', 'src')
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from ingestion.models import DatasetSpec
from ingestion.models import IngestionRequest
from ingestion.quality import QualityGate
from ingestion.service import IngestionService


class _FakeAdapter:
    adapter_name = 'fake_adapter'

    def fetch(self, request, spec):
        return pd.DataFrame(
            [
                {'instrument_id': '000001', 'trade_date': '2024-01-02', 'open': 10.0, 'high': 11.0, 'low': 9.0, 'close': 10.5},
                {'instrument_id': '000002', 'trade_date': '2024-01-02', 'open': 20.0, 'high': 21.0, 'low': 19.0, 'close': 20.5},
            ]
        )


class _EmptyAdapter:
    adapter_name = 'fake_adapter'

    def fetch(self, request, spec):
        return pd.DataFrame(columns=['instrument_id', 'trade_date', 'close'])


class _FailingAdapter:
    adapter_name = 'fake_adapter'

    def fetch(self, request, spec):
        raise RuntimeError('malformed provider payload')


class _RetryableAdapter:
    adapter_name = 'fake_adapter'

    def fetch(self, request, spec):
        raise TimeoutError('provider timeout')


class _BadSchemaAdapter:
    adapter_name = 'fake_adapter'

    def fetch(self, request, spec):
        return pd.DataFrame(
            [
                {'instrument_id': '000001', 'trade_date': '2024-01-02'},
            ]
        )


class IngestionServiceTests(unittest.TestCase):
    def test_run_sync_executes_adapter_quality_and_job_persistence(self):
        with tempfile.TemporaryDirectory() as tmp:
            spec = DatasetSpec(
                dataset='market_bar_1d',
                domain='market',
                granularity='1d',
                source_name='fake',
                adapter_name='fake_adapter',
                request_spec={},
                schema_spec={'primary_key': ['instrument_id', 'trade_date'], 'columns': {'instrument_id': {'source': 'instrument_id'}}},
                quality_spec={'required_columns': ['instrument_id', 'trade_date', 'close']},
                storage_spec={'raw_uri': 'data/raw/{dataset}/{run_id}.csv', 'curated_uri': 'data/curated/{dataset}/{run_id}.csv'},
            )
            service = IngestionService.for_testing(
                specs={'market_bar_1d': spec},
                adapters={'fake_adapter': _FakeAdapter()},
                runtime_root=tmp,
            )

            job = service.create_job(IngestionRequest(dataset='market_bar_1d', start='2024-01-01', end='2024-01-31'))
            finished = service.run_job(job.job_id)

            self.assertEqual(finished.status, 'succeeded')
            self.assertTrue(finished.manifest_path.endswith('.json'))
            self.assertTrue(os.path.exists(finished.manifest_path))
            self.assertEqual(finished.result.get('row_count'), 2)
            self.assertEqual(finished.result.get('status'), 'succeeded')
            self.assertTrue(finished.result.get('finished_at'))

    def test_run_sync_marks_fetch_failures_without_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            spec = DatasetSpec(
                dataset='market_bar_1d',
                domain='market',
                granularity='1d',
                source_name='fake',
                adapter_name='fake_adapter',
                request_spec={},
                schema_spec={'primary_key': ['instrument_id', 'trade_date']},
                quality_spec={'required_columns': ['instrument_id', 'trade_date', 'close']},
                storage_spec={'raw_uri': 'data/raw/{dataset}/{run_id}.csv', 'curated_uri': 'data/curated/{dataset}/{run_id}.csv'},
            )
            service = IngestionService.for_testing(
                specs={'market_bar_1d': spec},
                adapters={'fake_adapter': _FailingAdapter()},
                runtime_root=tmp,
            )

            job = service.create_job(IngestionRequest(dataset='market_bar_1d', start='2024-01-01', end='2024-01-31'))

            with self.assertRaisesRegex(RuntimeError, 'malformed provider payload'):
                service.run_job(job.job_id)

            failed = service.get_job(job.job_id)
            self.assertEqual(failed.status, 'fetch_failed')
            self.assertEqual(failed.errors, ['malformed provider payload'])
            self.assertEqual(failed.manifest_path, '')

    def test_run_sync_marks_retryable_failures_for_provider_timeouts(self):
        with tempfile.TemporaryDirectory() as tmp:
            spec = DatasetSpec(
                dataset='market_bar_1d',
                domain='market',
                granularity='1d',
                source_name='fake',
                adapter_name='fake_adapter',
                request_spec={},
                schema_spec={'primary_key': ['instrument_id', 'trade_date']},
                quality_spec={'required_columns': ['instrument_id', 'trade_date', 'close']},
                storage_spec={'raw_uri': 'data/raw/{dataset}/{run_id}.csv', 'curated_uri': 'data/curated/{dataset}/{run_id}.csv'},
            )
            service = IngestionService.for_testing(
                specs={'market_bar_1d': spec},
                adapters={'fake_adapter': _RetryableAdapter()},
                runtime_root=tmp,
            )

            job = service.create_job(IngestionRequest(dataset='market_bar_1d', start='2024-01-01', end='2024-01-31'))

            with self.assertRaisesRegex(TimeoutError, 'provider timeout'):
                service.run_job(job.job_id)

            failed = service.get_job(job.job_id)
            self.assertEqual(failed.status, 'retryable_failed')
            self.assertEqual(failed.errors, ['provider timeout'])
            self.assertEqual(failed.manifest_path, '')

    def test_run_sync_marks_quality_failures_before_manifest_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            spec = DatasetSpec(
                dataset='market_bar_1d',
                domain='market',
                granularity='1d',
                source_name='fake',
                adapter_name='fake_adapter',
                request_spec={},
                schema_spec={'primary_key': ['instrument_id', 'trade_date']},
                quality_spec={'required_columns': ['instrument_id', 'trade_date', 'close']},
                storage_spec={'raw_uri': 'data/raw/{dataset}/{run_id}.csv', 'curated_uri': 'data/curated/{dataset}/{run_id}.csv'},
            )
            service = IngestionService.for_testing(
                specs={'market_bar_1d': spec},
                adapters={'fake_adapter': _BadSchemaAdapter()},
                runtime_root=tmp,
            )

            job = service.create_job(IngestionRequest(dataset='market_bar_1d', start='2024-01-01', end='2024-01-31'))

            with self.assertRaisesRegex(ValueError, 'missing required columns'):
                service.run_job(job.job_id)

            failed = service.get_job(job.job_id)
            self.assertEqual(failed.status, 'quality_failed')
            self.assertIn('missing required columns', failed.errors[0])
            self.assertEqual(failed.manifest_path, '')

    def test_run_sync_keeps_empty_canonical_result_with_warning(self):
        with tempfile.TemporaryDirectory() as tmp:
            spec = DatasetSpec(
                dataset='market_bar_1d',
                domain='market',
                granularity='1d',
                source_name='fake',
                adapter_name='fake_adapter',
                request_spec={},
                schema_spec={'primary_key': ['instrument_id', 'trade_date']},
                quality_spec={'required_columns': ['instrument_id', 'trade_date', 'close']},
                storage_spec={'raw_uri': 'data/raw/{dataset}/{run_id}.csv', 'curated_uri': 'data/curated/{dataset}/{run_id}.csv'},
            )
            service = IngestionService.for_testing(
                specs={'market_bar_1d': spec},
                adapters={'fake_adapter': _EmptyAdapter()},
                runtime_root=tmp,
            )

            job = service.create_job(IngestionRequest(dataset='market_bar_1d', start='2024-01-01', end='2024-01-31'))
            finished = service.run_job(job.job_id)

            self.assertEqual(finished.status, 'succeeded')
            self.assertEqual(finished.result.get('row_count'), 0)
            self.assertEqual(finished.warnings, ['empty_result'])
            self.assertEqual(finished.result.get('warnings'), ['empty_result'])


class QualityGateTests(unittest.TestCase):
    def test_validate_supports_unique_expression_and_coverage_rules(self):
        rows = []
        for trade_date, count in [('2024-01-02', 10), ('2024-01-03', 9), ('2024-01-04', 7)]:
            for offset in range(count):
                base = 10.0 + offset
                rows.append(
                    {
                        'instrument_id': f'{offset + 1:06d}',
                        'trade_date': trade_date,
                        'open': base,
                        'high': base + 1.0,
                        'low': base - 1.0,
                        'close': base + 0.5,
                        'volume': 1000 + offset,
                        'amount': 10000 + offset,
                    }
                )
        df = pd.DataFrame(rows)
        spec = DatasetSpec(
            dataset='market_bar_1d',
            domain='market',
            granularity='1d',
            source_name='fake',
            adapter_name='fake_adapter',
            request_spec={},
            schema_spec={'primary_key': ['instrument_id', 'trade_date']},
            quality_spec={
                'required_columns': ['instrument_id', 'trade_date', 'open', 'high', 'low', 'close', 'volume', 'amount'],
                'rules': [
                    {'name': 'no_duplicate_pk', 'type': 'unique_key', 'key': ['instrument_id', 'trade_date']},
                    {'name': 'valid_price_range', 'type': 'expression', 'expr': 'open > 0 and high >= low and close > 0'},
                    {'name': 'non_negative_liquidity', 'type': 'expression', 'expr': 'volume >= 0 and amount >= 0'},
                    {'name': 'coverage_floor', 'type': 'stock_coverage_vs_trade_days', 'min_ratio_p50': 0.85, 'min_ratio_p10': 0.65},
                ],
            },
            storage_spec={},
        )

        summary = QualityGate().validate(df, spec)

        self.assertEqual(summary['row_count'], len(df))
        self.assertEqual(len(summary['rules']), 4)
        self.assertTrue(all(rule['passed'] for rule in summary['rules']))
        self.assertEqual(summary['rules'][0]['name'], 'no_duplicate_pk')

    def test_validate_rejects_out_of_session_bars(self):
        df = pd.DataFrame(
            [
                {
                    'instrument_id': '000001',
                    'ts': '2024-01-02 12:05:00',
                    'trade_date': '2024-01-02',
                    'open': 10.0,
                    'high': 10.2,
                    'low': 9.9,
                    'close': 10.1,
                    'volume': 1000,
                    'amount': 10100,
                }
            ]
        )
        spec = DatasetSpec(
            dataset='market_bar_1m',
            domain='market',
            granularity='1m',
            source_name='fake',
            adapter_name='fake_adapter',
            request_spec={},
            schema_spec={'primary_key': ['instrument_id', 'ts']},
            quality_spec={
                'required_columns': ['instrument_id', 'ts', 'close'],
                'rules': [
                    {'name': 'no_duplicate_pk', 'type': 'unique_key', 'key': ['instrument_id', 'ts']},
                    {'name': 'ts_in_session', 'type': 'market_session_check', 'calendar': 'CN_STOCK'},
                ],
            },
            storage_spec={},
        )

        with self.assertRaisesRegex(ValueError, 'market session'):
            QualityGate().validate(df, spec)


if __name__ == '__main__':
    unittest.main()
