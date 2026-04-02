from __future__ import annotations


class QualityGate:
    def _require_pandas(self):
        try:
            import pandas as pd
        except ModuleNotFoundError as exc:
            raise RuntimeError('pandas is required for ingestion quality checks') from exc
        return pd

    def _rule_name(self, rule, *, default: str) -> str:
        return str(rule.get('name', '') or default)

    def _validate_unique_key(self, df, rule, summary):
        key = rule.get('key', [])
        key = list(key) if isinstance(key, list) else []
        if not key:
            return
        missing = [column for column in key if column not in df.columns]
        if missing:
            raise ValueError(f'missing columns for unique key rule: {missing}')
        duplicate_rows = int(df.duplicated(subset=key).sum())
        summary['key'] = list(key)
        summary['duplicate_rows'] = duplicate_rows
        summary['passed'] = duplicate_rows == 0
        if duplicate_rows:
            raise ValueError(f'duplicate rows for unique key {key}')

    def _validate_expression(self, df, rule, summary):
        pd = self._require_pandas()
        expr = str(rule.get('expr', '') or '').strip()
        if not expr:
            summary['passed'] = True
            summary['failed_rows'] = 0
            return
        result = df.eval(expr, engine='python')
        if isinstance(result, pd.Series):
            failed = int((~result.fillna(False)).sum())
        else:
            failed = 0 if bool(result) else int(len(df))
        summary['expr'] = expr
        summary['failed_rows'] = failed
        summary['passed'] = failed == 0
        if failed:
            raise ValueError(f'quality expression failed for rule {summary["name"]}')

    def _validate_stock_coverage_vs_trade_days(self, df, rule, summary):
        required = ['instrument_id', 'trade_date']
        missing = [column for column in required if column not in df.columns]
        if missing:
            raise ValueError(f'missing columns for coverage rule: {missing}')
        if df.empty:
            summary['passed'] = True
            summary['day_count'] = 0
            summary['ratio_p50'] = 1.0
            summary['ratio_p10'] = 1.0
            return
        counts = df.groupby('trade_date')['instrument_id'].nunique()
        max_count = float(counts.max()) if not counts.empty else 0.0
        ratio_p50 = float(counts.quantile(0.5) / max_count) if max_count else 1.0
        ratio_p10 = float(counts.quantile(0.1) / max_count) if max_count else 1.0
        min_ratio_p50 = float(rule.get('min_ratio_p50', 0.0) or 0.0)
        min_ratio_p10 = float(rule.get('min_ratio_p10', 0.0) or 0.0)
        summary['day_count'] = int(len(counts))
        summary['ratio_p50'] = ratio_p50
        summary['ratio_p10'] = ratio_p10
        summary['passed'] = ratio_p50 >= min_ratio_p50 and ratio_p10 >= min_ratio_p10
        if not summary['passed']:
            raise ValueError(
                f'stock coverage rule failed: p50={ratio_p50:.3f}, p10={ratio_p10:.3f}'
            )

    def _validate_market_session_check(self, df, rule, summary):
        pd = self._require_pandas()
        if 'ts' not in df.columns:
            raise ValueError('missing columns for market session rule: [\'ts\']')
        if df.empty:
            summary['passed'] = True
            summary['invalid_rows'] = 0
            return
        timestamps = pd.to_datetime(df['ts'], errors='coerce')
        minutes = timestamps.dt.hour * 60 + timestamps.dt.minute
        in_morning = (minutes >= 9 * 60 + 30) & (minutes <= 11 * 60 + 30)
        in_afternoon = (minutes >= 13 * 60) & (minutes <= 15 * 60)
        valid = timestamps.notna() & (in_morning | in_afternoon)
        invalid_rows = int((~valid).sum())
        summary['calendar'] = str(rule.get('calendar', '') or '')
        summary['invalid_rows'] = invalid_rows
        summary['passed'] = invalid_rows == 0
        if invalid_rows:
            raise ValueError(f'market session check failed for {invalid_rows} rows')

    def validate(self, df, spec):
        required = spec.quality_spec.get('required_columns', [])
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f'missing required columns: {missing}')

        primary_key = spec.primary_key
        if primary_key and df.duplicated(subset=primary_key).any():
            raise ValueError(f'duplicate primary key rows for {primary_key}')

        summary = {
            'row_count': int(len(df)),
            'required_columns': list(required),
            'primary_key': list(primary_key),
            'rules': [],
        }
        for rule in spec.quality_spec.get('rules', []) or []:
            rule_type = str(rule.get('type', '') or '').strip()
            rule_summary = {
                'name': self._rule_name(rule, default=rule_type or 'rule'),
                'type': rule_type,
                'passed': True,
            }
            if rule_type == 'unique_key':
                self._validate_unique_key(df, rule, rule_summary)
            elif rule_type == 'expression':
                self._validate_expression(df, rule, rule_summary)
            elif rule_type == 'stock_coverage_vs_trade_days':
                self._validate_stock_coverage_vs_trade_days(df, rule, rule_summary)
            elif rule_type == 'market_session_check':
                self._validate_market_session_check(df, rule, rule_summary)
            else:
                rule_summary['skipped'] = True
            summary['rules'].append(rule_summary)
        return summary
