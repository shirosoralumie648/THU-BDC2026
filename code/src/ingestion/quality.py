class QualityGate:
    def validate(self, df, spec):
        required = spec.quality_spec.get('required_columns', [])
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f'missing required columns: {missing}')

        primary_key = spec.primary_key
        if primary_key and df.duplicated(subset=primary_key).any():
            raise ValueError(f'duplicate primary key rows for {primary_key}')
