import argparse
import json

from config import config
from factor_store import (
    clear_builtin_override,
    delete_custom_factor,
    get_factor_spec,
    resolve_factor_pipeline,
    set_factor_enabled,
    upsert_builtin_override,
    upsert_custom_factor,
)


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

    show_parser = subparsers.add_parser('show', help='查看单个因子详情')
    show_parser.add_argument('name', help='因子名')

    enable_parser = subparsers.add_parser('enable', help='启用因子')
    enable_parser.add_argument('name', help='因子名')

    disable_parser = subparsers.add_parser('disable', help='关闭因子')
    disable_parser.add_argument('name', help='因子名')

    create_parser = subparsers.add_parser('create', help='新建自定义因子')
    create_parser.add_argument('name', help='因子名')
    create_parser.add_argument('--expression', required=True, help='因子表达式')
    create_parser.add_argument('--group', default='custom', help='因子分组')
    create_parser.add_argument('--description', default='', help='因子说明')
    create_parser.add_argument('--disabled', action='store_true', help='创建后默认关闭')

    update_parser = subparsers.add_parser('update', help='编辑自定义因子')
    update_parser.add_argument('name', help='因子名')
    update_parser.add_argument('--expression', required=True, help='新的因子表达式')
    update_parser.add_argument('--group', default='custom', help='因子分组')
    update_parser.add_argument('--description', default='', help='因子说明')
    update_parser.add_argument('--disabled', action='store_true', help='更新后关闭')

    delete_parser = subparsers.add_parser('delete', help='删除自定义因子')
    delete_parser.add_argument('name', help='因子名')

    reset_parser = subparsers.add_parser('reset', help='恢复内置因子的默认公式')
    reset_parser.add_argument('name', help='内置因子名')

    return parser


def _print_factor_list(pipeline, enabled_only=False):
    specs = pipeline['all_specs']
    if enabled_only:
        specs = [spec for spec in specs if spec.get('enabled', True)]

    print(
        f'feature_set={pipeline["feature_set"]} '
        f'active={pipeline["summary"]["active_total"]} '
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

    if args.command == 'create':
        upsert_custom_factor(
            args.store_path,
            args.feature_set,
            args.name,
            args.expression,
            group=args.group,
            description=args.description,
            enabled=not args.disabled,
        )
        print(f'已创建自定义因子: {args.name}')
        return

    if args.command == 'update':
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

    raise ValueError(f'未知命令: {args.command}')


if __name__ == '__main__':
    main()
