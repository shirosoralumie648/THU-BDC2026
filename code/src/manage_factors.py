import argparse
import json
from pathlib import Path

from config import config
from factor_store import (
    activate_only_factors,
    clear_builtin_override,
    delete_custom_factor,
    get_factor_spec,
    resolve_factor_pipeline,
    set_factors_enabled,
    set_factor_enabled,
    set_group_enabled,
    upsert_builtin_override,
    upsert_custom_factor,
)


def _parse_inputs_arg(raw_inputs):
    if raw_inputs is None:
        return None
    raw_inputs = raw_inputs.strip()
    if not raw_inputs:
        return {}
    try:
        parsed = json.loads(raw_inputs)
    except json.JSONDecodeError as exc:
        raise ValueError(f'--inputs 不是合法 JSON: {exc}') from exc
    if not isinstance(parsed, dict):
        raise ValueError('--inputs 必须是 JSON 对象，例如 {"input_price":"收盘","window":20}')
    return parsed


def _build_parser():
    parser = argparse.ArgumentParser(description='因子管理工具')
    parser.add_argument(
        '--feature-set',
        default=config['feature_num'],
        choices=['39', '158+39'],
        help='要操作的因子集合',
    )
    parser.add_argument(
        '--store-path',
        default=config['factor_store_path'],
        help='因子存储文件路径',
    )

    subparsers = parser.add_subparsers(dest='command', required=True)

    list_parser = subparsers.add_parser('list', help='列出因子')
    list_parser.add_argument('--enabled-only', action='store_true', help='只显示启用中的因子')
    list_parser.add_argument('--json', action='store_true', help='以 JSON 输出')

    list_groups_parser = subparsers.add_parser('list-groups', help='按分组查看因子统计')
    list_groups_parser.add_argument('--json', action='store_true', help='以 JSON 输出')

    show_parser = subparsers.add_parser('show', help='查看单个因子详情')
    show_parser.add_argument('name', help='因子名')

    enable_parser = subparsers.add_parser('enable', help='启用因子')
    enable_parser.add_argument('name', help='因子名')

    disable_parser = subparsers.add_parser('disable', help='关闭因子')
    disable_parser.add_argument('name', help='因子名')

    enable_many_parser = subparsers.add_parser('enable-many', help='批量启用因子')
    enable_many_parser.add_argument('names', nargs='+', help='因子名列表')
    enable_many_parser.add_argument('--ignore-missing', action='store_true', help='忽略不存在的因子')

    disable_many_parser = subparsers.add_parser('disable-many', help='批量关闭因子')
    disable_many_parser.add_argument('names', nargs='+', help='因子名列表')
    disable_many_parser.add_argument('--ignore-missing', action='store_true', help='忽略不存在的因子')

    enable_group_parser = subparsers.add_parser('enable-group', help='按分组启用因子')
    enable_group_parser.add_argument('group', help='分组名')
    enable_group_parser.add_argument('--source', choices=['all', 'builtin', 'custom'], default='all')

    disable_group_parser = subparsers.add_parser('disable-group', help='按分组关闭因子')
    disable_group_parser.add_argument('group', help='分组名')
    disable_group_parser.add_argument('--source', choices=['all', 'builtin', 'custom'], default='all')

    activate_only_parser = subparsers.add_parser('activate-only', help='仅保留指定因子为启用状态')
    activate_only_parser.add_argument(
        '--names',
        default='',
        help='逗号分隔因子名，例如 "RSQR60,ROC20,return_20"',
    )
    activate_only_parser.add_argument(
        '--from-file',
        default='',
        help='从文件读取因子名（每行一个，支持逗号分隔）',
    )
    activate_only_parser.add_argument('--ignore-missing', action='store_true', help='忽略不存在的因子')

    create_parser = subparsers.add_parser('create', help='新建自定义因子')
    create_parser.add_argument('name', help='因子名')
    create_parser.add_argument('--expression', required=True, help='因子表达式')
    create_parser.add_argument('--group', default='custom', help='因子分组')
    create_parser.add_argument('--description', default='', help='因子说明')
    create_parser.add_argument('--inputs', default=None, help='输入映射(JSON)，例如 {"input_price":"收盘","window":20}')
    create_parser.add_argument('--author', default=None, help='作者标识')
    create_parser.add_argument('--disabled', action='store_true', help='创建后默认关闭')

    update_parser = subparsers.add_parser('update', help='编辑自定义因子')
    update_parser.add_argument('name', help='因子名')
    update_parser.add_argument('--expression', required=True, help='新的因子表达式')
    update_parser.add_argument('--group', default='custom', help='因子分组')
    update_parser.add_argument('--description', default='', help='因子说明')
    update_parser.add_argument('--inputs', default=None, help='输入映射(JSON)，例如 {"input_price":"收盘","window":20}')
    update_parser.add_argument('--author', default=None, help='作者标识')
    update_parser.add_argument('--disabled', action='store_true', help='更新后关闭')

    delete_parser = subparsers.add_parser('delete', help='删除自定义因子')
    delete_parser.add_argument('name', help='因子名')

    reset_parser = subparsers.add_parser('reset', help='恢复内置因子的默认公式')
    reset_parser.add_argument('name', help='内置因子名')

    return parser


def _group_summary_rows(pipeline):
    rows = {}
    for spec in pipeline['all_specs']:
        group = str(spec.get('group', 'unknown'))
        source = str(spec.get('source', 'unknown'))
        key = (group, source)
        if key not in rows:
            rows[key] = {
                'group': group,
                'source': source,
                'total': 0,
                'enabled': 0,
                'disabled': 0,
            }
        rows[key]['total'] += 1
        if spec.get('enabled', True):
            rows[key]['enabled'] += 1
        else:
            rows[key]['disabled'] += 1
    return sorted(rows.values(), key=lambda row: (row['group'], row['source']))


def _parse_activate_only_names(raw_names: str, file_path: str):
    names = []
    raw_names = str(raw_names or '').strip()
    if raw_names:
        names.extend([part.strip() for part in raw_names.split(',') if part.strip()])

    file_path = str(file_path or '').strip()
    if file_path:
        payload = Path(file_path).read_text(encoding='utf-8')
        for line in payload.splitlines():
            for part in line.split(','):
                name = part.strip()
                if name:
                    names.append(name)

    return list(dict.fromkeys(names))


def _print_factor_list(pipeline, enabled_only=False):
    specs = pipeline['all_specs']
    if enabled_only:
        specs = [spec for spec in specs if spec.get('enabled', True)]

    print(
        f'feature_set={pipeline["feature_set"]} '
        f'active={pipeline["summary"]["active_total"]} '
        f'cross={pipeline["summary"].get("cross_sectional_total", 0)} '
        f'builtin={pipeline["summary"]["builtin_enabled"]}/{pipeline["summary"]["builtin_total"]} '
        f'builtin_overridden={pipeline["summary"].get("builtin_overridden", 0)} '
        f'custom={pipeline["summary"]["custom_enabled"]}/{pipeline["summary"]["custom_total"]}'
    )
    print('enabled  source   override  group       name')
    print('-------  -------  --------  ----------  ------------------------------')
    for spec in specs:
        enabled = 'on ' if spec.get('enabled', True) else 'off'
        override = 'yes' if spec.get('overridden', False) else 'no'
        print(
            f'{enabled:<7}  '
            f'{spec.get("source", "unknown"):<7}  '
            f'{override:<8}  '
            f'{spec.get("group", "unknown"):<10}  '
            f'{spec["name"]}'
        )


def main():
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == 'enable':
        set_factor_enabled(args.store_path, args.feature_set, args.name, True)
        print(f'已启用因子: {args.name}')
        return

    if args.command == 'disable':
        set_factor_enabled(args.store_path, args.feature_set, args.name, False)
        print(f'已关闭因子: {args.name}')
        return

    if args.command == 'enable-many':
        result = set_factors_enabled(
            args.store_path,
            args.feature_set,
            args.names,
            enabled=True,
            strict=not bool(args.ignore_missing),
        )
        print(
            f'批量启用完成: updated={result["updated_total"]} '
            f'(builtin={result["updated_builtin"]}, custom={result["updated_custom"]})'
        )
        if result['missing']:
            print(f'忽略不存在因子: {result["missing"]}')
        return

    if args.command == 'disable-many':
        result = set_factors_enabled(
            args.store_path,
            args.feature_set,
            args.names,
            enabled=False,
            strict=not bool(args.ignore_missing),
        )
        print(
            f'批量关闭完成: updated={result["updated_total"]} '
            f'(builtin={result["updated_builtin"]}, custom={result["updated_custom"]})'
        )
        if result['missing']:
            print(f'忽略不存在因子: {result["missing"]}')
        return

    if args.command == 'enable-group':
        result = set_group_enabled(
            args.store_path,
            args.feature_set,
            args.group,
            enabled=True,
            source=args.source,
        )
        print(
            f'分组启用完成: group={result["group"]}, source={result["source"]}, '
            f'matched={result["matched"]}, updated={result["updated_total"]}'
        )
        return

    if args.command == 'disable-group':
        result = set_group_enabled(
            args.store_path,
            args.feature_set,
            args.group,
            enabled=False,
            source=args.source,
        )
        print(
            f'分组关闭完成: group={result["group"]}, source={result["source"]}, '
            f'matched={result["matched"]}, updated={result["updated_total"]}'
        )
        return

    if args.command == 'activate-only':
        names = _parse_activate_only_names(args.names, args.from_file)
        if not names:
            raise ValueError('activate-only 至少需要通过 --names 或 --from-file 提供一个因子名')
        result = activate_only_factors(
            args.store_path,
            args.feature_set,
            names,
            strict=not bool(args.ignore_missing),
        )
        print(
            f'activate-only 完成: active={result["active_count"]} '
            f'(builtin={result["builtin_active"]}, custom={result["custom_active"]})'
        )
        if result['unknown']:
            print(f'忽略未知因子: {result["unknown"]}')
        return

    if args.command == 'create':
        inputs = _parse_inputs_arg(args.inputs)
        upsert_custom_factor(
            args.store_path,
            args.feature_set,
            args.name,
            args.expression,
            group=args.group,
            description=args.description,
            enabled=not args.disabled,
            inputs=inputs,
            author=args.author,
        )
        print(f'已创建自定义因子: {args.name}')
        return

    if args.command == 'update':
        inputs = _parse_inputs_arg(args.inputs)
        spec = get_factor_spec(args.feature_set, args.store_path, args.name)
        if spec.get('source') == 'custom':
            upsert_custom_factor(
                args.store_path,
                args.feature_set,
                args.name,
                args.expression,
                group=args.group,
                description=args.description,
                enabled=not args.disabled,
                inputs=inputs,
                author=args.author,
            )
            print(f'已更新自定义因子: {args.name}')
            return

        upsert_builtin_override(
            args.store_path,
            args.feature_set,
            args.name,
            args.expression,
            group=args.group,
            description=args.description,
            inputs=inputs,
            author=args.author,
        )
        if args.disabled:
            set_factor_enabled(args.store_path, args.feature_set, args.name, False)
        print(f'已更新内置因子 override: {args.name}')
        return

    if args.command == 'delete':
        spec = get_factor_spec(args.feature_set, args.store_path, args.name)
        if spec.get('source') != 'custom':
            raise ValueError(f'内置因子不能删除，请使用 reset 恢复默认公式: {args.name}')
        delete_custom_factor(args.store_path, args.feature_set, args.name)
        print(f'已删除自定义因子: {args.name}')
        return

    if args.command == 'reset':
        clear_builtin_override(args.store_path, args.feature_set, args.name)
        print(f'已恢复内置因子默认公式: {args.name}')
        return

    if args.command == 'show':
        spec = get_factor_spec(args.feature_set, args.store_path, args.name)
        print(json.dumps(spec, indent=2, ensure_ascii=False))
        return

    if args.command == 'list':
        pipeline = resolve_factor_pipeline(args.feature_set, args.store_path)
        if args.json:
            print(json.dumps(pipeline, indent=2, ensure_ascii=False))
        else:
            _print_factor_list(pipeline, enabled_only=args.enabled_only)
        return

    if args.command == 'list-groups':
        pipeline = resolve_factor_pipeline(args.feature_set, args.store_path)
        rows = _group_summary_rows(pipeline)
        if args.json:
            print(json.dumps(rows, indent=2, ensure_ascii=False))
            return
        print('group                         source    enabled/total')
        print('----------------------------  --------  -------------')
        for row in rows:
            group_label = row['group'][:28]
            print(f'{group_label:<28}  {row["source"]:<8}  {row["enabled"]}/{row["total"]}')
        return

    raise ValueError(f'未知命令: {args.command}')


if __name__ == '__main__':
    main()
