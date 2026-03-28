import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class PipelineConfigError(ValueError):
    """配置加载或校验错误。"""


@dataclass
class ValidationReport:
    valid: bool
    errors: List[str]
    warnings: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            'valid': bool(self.valid),
            'errors': list(self.errors),
            'warnings': list(self.warnings),
        }


def _project_root() -> Path:
    # code/src/pipeline_config.py -> project root
    return Path(__file__).resolve().parents[2]


def _default_config_dir() -> Path:
    return _project_root() / 'config'


def _resolve_path(path_str: str) -> Path:
    p = Path(path_str).expanduser()
    if p.is_absolute():
        return p
    return (_project_root() / p).resolve()


def load_yaml_file(path: str, *, required: bool = True) -> Dict[str, Any]:
    target = _resolve_path(path)
    if not target.exists():
        if required:
            raise PipelineConfigError(f'未找到配置文件: {target}')
        return {}

    try:
        import yaml
    except Exception as exc:  # pragma: no cover - runtime dependency failure
        raise PipelineConfigError(
            '缺少 PyYAML 依赖，请安装 `pyyaml` 后再加载 YAML 配置'
        ) from exc

    try:
        with open(target, 'r', encoding='utf-8') as f:
            payload = yaml.safe_load(f) or {}
    except Exception as exc:
        raise PipelineConfigError(f'YAML 解析失败: {target} | {exc}') from exc

    if not isinstance(payload, dict):
        raise PipelineConfigError(f'YAML 顶层必须是对象: {target}')
    return payload


def _require_keys(payload: Dict[str, Any], required_keys: List[str], prefix: str) -> List[str]:
    errors = []
    for key in required_keys:
        if key not in payload:
            errors.append(f'{prefix} 缺少字段: {key}')
    return errors


def validate_datasets_config(payload: Dict[str, Any]) -> ValidationReport:
    errors = _require_keys(payload, ['version', 'datasets'], 'datasets.yaml')
    warnings: List[str] = []

    datasets = payload.get('datasets', {})
    if datasets is None:
        datasets = {}
    if not isinstance(datasets, dict):
        errors.append('datasets.yaml 字段 `datasets` 必须为对象')
        datasets = {}

    for name, spec in datasets.items():
        if not isinstance(spec, dict):
            errors.append(f'datasets.yaml dataset `{name}` 配置必须为对象')
            continue
        errors.extend(
            _require_keys(spec, ['source', 'schema', 'storage'], f'datasets.yaml dataset `{name}`')
        )
        schema = spec.get('schema', {})
        if isinstance(schema, dict):
            if 'columns' not in schema:
                errors.append(f'datasets.yaml dataset `{name}` 的 schema 缺少 columns')
        else:
            errors.append(f'datasets.yaml dataset `{name}` 的 schema 必须为对象')

        storage = spec.get('storage', {})
        if not isinstance(storage, dict):
            errors.append(f'datasets.yaml dataset `{name}` 的 storage 必须为对象')
        else:
            if ('raw_uri' not in storage) and ('curated_uri' not in storage):
                warnings.append(f'datasets.yaml dataset `{name}` 未声明 raw_uri/curated_uri')

    return ValidationReport(valid=len(errors) == 0, errors=errors, warnings=warnings)


def validate_factors_config(payload: Dict[str, Any]) -> ValidationReport:
    errors = _require_keys(payload, ['version', 'layer_order', 'factor_nodes'], 'factors.yaml')
    warnings: List[str] = []

    layer_order = payload.get('layer_order', [])
    if not isinstance(layer_order, list):
        errors.append('factors.yaml `layer_order` 必须为数组')
        layer_order = []

    nodes = payload.get('factor_nodes', [])
    if not isinstance(nodes, list):
        errors.append('factors.yaml `factor_nodes` 必须为数组')
        nodes = []

    seen_ids = set()
    known_ids = set()
    for node in nodes:
        if not isinstance(node, dict):
            errors.append('factors.yaml 中 factor_nodes 元素必须为对象')
            continue
        node_id = str(node.get('id', '')).strip()
        if not node_id:
            errors.append('factors.yaml 存在缺少 id 的因子节点')
            continue
        if node_id in seen_ids:
            errors.append(f'factors.yaml 因子节点重复 id: {node_id}')
        seen_ids.add(node_id)
        known_ids.add(node_id)

        layer = str(node.get('layer', '')).strip()
        if layer and layer_order and (layer not in layer_order):
            errors.append(f'factors.yaml 节点 `{node_id}` 引用了未定义 layer: {layer}')
        compute = node.get('compute', {})
        if not isinstance(compute, dict):
            errors.append(f'factors.yaml 节点 `{node_id}` 的 compute 必须为对象')
            continue
        if 'engine' not in compute:
            errors.append(f'factors.yaml 节点 `{node_id}` 缺少 compute.engine')
        if ('expression' not in compute) and (compute.get('engine') in {'expression', 'meta_expression'}):
            errors.append(f'factors.yaml 节点 `{node_id}` 缺少 compute.expression')

    for node in nodes:
        if not isinstance(node, dict):
            continue
        node_id = str(node.get('id', '')).strip()
        deps = node.get('dependencies', [])
        if deps is None:
            deps = []
        if not isinstance(deps, list):
            errors.append(f'factors.yaml 节点 `{node_id}` 的 dependencies 必须为数组')
            continue
        for dep in deps:
            dep_name = str(dep).strip()
            if dep_name and dep_name not in known_ids:
                warnings.append(f'factors.yaml 节点 `{node_id}` 依赖 `{dep_name}` 未在 factor_nodes 中找到')

    return ValidationReport(valid=len(errors) == 0, errors=errors, warnings=warnings)


def validate_storage_config(payload: Dict[str, Any]) -> ValidationReport:
    errors = _require_keys(payload, ['version', 'layers'], 'storage.yaml')
    warnings: List[str] = []

    layers = payload.get('layers', {})
    if not isinstance(layers, dict):
        errors.append('storage.yaml `layers` 必须为对象')
        layers = {}

    for required_layer in ['raw', 'curated', 'feature_long', 'feature_wide', 'datasets', 'manifests']:
        if required_layer not in layers:
            errors.append(f'storage.yaml 缺少层定义: {required_layer}')

    for layer_name, spec in layers.items():
        if not isinstance(spec, dict):
            errors.append(f'storage.yaml layer `{layer_name}` 必须为对象')
            continue
        if 'uri_template' not in spec:
            warnings.append(f'storage.yaml layer `{layer_name}` 未配置 uri_template')

    return ValidationReport(valid=len(errors) == 0, errors=errors, warnings=warnings)


def validate_pipeline_configs(
    datasets_payload: Dict[str, Any],
    factors_payload: Dict[str, Any],
    storage_payload: Dict[str, Any],
) -> ValidationReport:
    reports = [
        validate_datasets_config(datasets_payload),
        validate_factors_config(factors_payload),
        validate_storage_config(storage_payload),
    ]
    errors: List[str] = []
    warnings: List[str] = []
    for report in reports:
        errors.extend(report.errors)
        warnings.extend(report.warnings)
    return ValidationReport(valid=len(errors) == 0, errors=errors, warnings=warnings)


def load_pipeline_configs(
    *,
    config_dir: Optional[str] = None,
    strict: bool = True,
) -> Tuple[Dict[str, Any], ValidationReport]:
    base_dir = _resolve_path(config_dir) if config_dir else _default_config_dir()
    datasets_path = os.path.join(str(base_dir), 'datasets.yaml')
    factors_path = os.path.join(str(base_dir), 'factors.yaml')
    storage_path = os.path.join(str(base_dir), 'storage.yaml')

    datasets_payload = load_yaml_file(datasets_path, required=True)
    factors_payload = load_yaml_file(factors_path, required=True)
    storage_payload = load_yaml_file(storage_path, required=True)

    report = validate_pipeline_configs(datasets_payload, factors_payload, storage_payload)
    if strict and not report.valid:
        raise PipelineConfigError('配置校验失败:\n- ' + '\n- '.join(report.errors))

    return {
        'config_dir': str(base_dir),
        'datasets': datasets_payload,
        'factors': factors_payload,
        'storage': storage_payload,
    }, report


def get_dataset_spec(configs: Dict[str, Any], dataset_name: str) -> Dict[str, Any]:
    datasets_cfg = configs.get('datasets', {})
    datasets = datasets_cfg.get('datasets', {}) if isinstance(datasets_cfg, dict) else {}
    if dataset_name not in datasets:
        raise PipelineConfigError(f'datasets.yaml 中未找到 dataset: {dataset_name}')
    spec = datasets[dataset_name]
    if not isinstance(spec, dict):
        raise PipelineConfigError(f'dataset `{dataset_name}` 配置不是对象')
    return spec


def _is_truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {'1', 'true', 'yes', 'y', 'on'}


def derive_hf_builder_defaults(factors_cfg: Dict[str, Any]) -> Dict[str, Any]:
    out = {
        'tail_minutes': None,
        'min_bars': None,
        'resample_minutes': [],
        'skip_raw': None,
        'force_suffix': None,
    }
    builder_defaults = factors_cfg.get('builder_defaults', {}) if isinstance(factors_cfg, dict) else {}
    hf_defaults = builder_defaults.get('hf_daily_aggregation', {}) if isinstance(builder_defaults, dict) else {}
    if not isinstance(hf_defaults, dict):
        return out

    if 'tail_minutes' in hf_defaults:
        out['tail_minutes'] = int(hf_defaults['tail_minutes'])
    if 'min_bars' in hf_defaults:
        out['min_bars'] = int(hf_defaults['min_bars'])
    if 'resample_minutes' in hf_defaults and isinstance(hf_defaults['resample_minutes'], list):
        out['resample_minutes'] = [int(v) for v in hf_defaults['resample_minutes']]
    if 'skip_raw' in hf_defaults:
        out['skip_raw'] = _is_truthy(hf_defaults['skip_raw'])
    if 'force_suffix' in hf_defaults:
        out['force_suffix'] = _is_truthy(hf_defaults['force_suffix'])
    return out


def render_feature_csv_compat_uri(
    factors_cfg: Dict[str, Any],
    *,
    feature_set_version: str,
) -> str:
    views = factors_cfg.get('factor_views', [])
    if not isinstance(views, list):
        return ''
    for view in views:
        if not isinstance(view, dict):
            continue
        if view.get('layout') != 'wide':
            continue
        export_spec = view.get('export', {})
        if not isinstance(export_spec, dict):
            continue
        uri_tpl = str(export_spec.get('csv_compat_uri', '')).strip()
        if not uri_tpl:
            continue
        return uri_tpl.format(feature_set_version=feature_set_version)
    return ''
