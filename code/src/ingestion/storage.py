def render_storage_target(template: str, **kwargs) -> str:
    return str(template).format(**kwargs)
