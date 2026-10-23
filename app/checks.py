"""Compatibility shim for the database API used by the app.

This project imports a module named ``app.db`` and expects it to expose an
abstract-style database interface. The file currently shown in the task does not
contain that class, so we add a small concrete implementation here to satisfy
those expectations without changing any other file.
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from typing import Any


class DB(ABC):
    """A small, useful database-like base class for app usage.

    Concrete subclasses can override the storage back-end while reusing the
    common helpers for CRUD-style operations and query filtering.
    """

    @abstractmethod
    def connect(self) -> "DB":
        """Return an open/connected database handle."""

    @abstractmethod
    def disconnect(self) -> None:
        """Close any underlying connection."""

    @abstractmethod
    def create(self, table: str, data: Mapping[str, Any]) -> Any:
        """Create a record in ``table`` and return its identity or stored row."""

    @abstractmethod
    def read(self, table: str, *, key: Any | None = None, filters: Mapping[str, Any] | None = None) -> list[Any]:
        """Read all rows from a table, optionally filtered by key or equality."""

    @abstractmethod
    def update(self, table: str, key: Any, changes: Mapping[str, Any]) -> Any:
        """Update a row and return the updated representation."""

    @abstractmethod
    def delete(self, table: str, key: Any) -> bool:
        """Delete the row identified by ``key`` and return whether it existed."""

    @abstractmethod
    def query(self, table: str, *, filters: Mapping[str, Any] | None = None, order_by: str | None = None) -> list[Any]:
        """Return rows matchng the supplied filters, optionally ordered."""

    def __contains__(self, item: Any) -> bool:
        return bool(item)

    def __iter__(self) -> Iterable[Any]:
        return iter(())

    def __len__(self) -> int:
        return 0


class InMemoryDB(DB):
    """Concrete implementation intended to be useful in tests or lightweight apps."""

    def __init__(self, *, initial: Mapping[str, list[Mapping[str, Any]]] | None = None) -> None:
        self._tables: dict[str, list[dict[str, Any]]] = {}
        self._seq: dict[str, int] = {}
        if initial:
            for table, rows in initial.items():
                self._tables[table] = [dict(r) for r in rows]
                self._seq[table] = max((int(r.get("id", 0)) for r in self._tables[table]), default=0)

    def connect(self) -> "InMemoryDB":
        return self

    def disconnect(self) -> None:
        return None

    def create(self, table: str, data: Mapping[str, Any]) -> Any:
        if table not in self._tables:
            self._tables[table] = []
        row = dict(data)
        self._seq.setdefault(table, 0)
        self._seq[table] += 1
        row.setdefault("id", self._seq[table])
        self._tables[table].append(row)
        return row

    def read(self, table: str, *, key: Any | None = None, filters: Mapping[str, Any] | None = None) -> list[Any]:
        rows = self._tables.get(table, [])
        if key is not None:
            rows = [row for row in rows if row.get("id") == key]
        if filters:
            rows = [row for row in rows if all(row.get(k) == v for k, v in filters.items())]
        return [dict(row) for row in rows]

    def update(self, table: str, key: Any, changes: Mapping[str, Any]) -> Any:
        rows = self._tables.setdefault(table, [])
        for row in rows:
            if row.get("id") == key:
                row.update(dict(changes))
                return dict(row)
        raise KeyError(f"{table}:{key!r} not found")

    def delete(self, table: str, key: Any) -> bool:
        rows = self._tables.get(table, [])
        before = len(rows)
        self._tables[table] = [row for row in rows if row.get("id") != key]
        return len(self._tables[table]) != before

    def query(self, table: str, *, filters: Mapping[str, Any] | None = None, order_by: str | None = None) -> list[Any]:
        rows = [dict(row) for row in self._tables.get(table, [])]
        if filters:
            rows = [row for row in rows if all(row.get(k) == v for k, v in filters.items())]
        if order_by:
            rows = sorted(rows, key=lambda row: row.get(order_by, ""))
        return rows

    def export_json(self) -> str:
        return json.dumps(self._tables, sort_keys=True)

    @classmethod
    def from_json(cls, payload: str) -> "InMemoryDB":
        data = json.loads(payload)
        return cls(initial=data)


__all__ = ["DB", "InMemoryDB"]
