from typing import Protocol


class BaseAdapter(Protocol):
    adapter_name: str

    def fetch(self, request, spec):
        ...
