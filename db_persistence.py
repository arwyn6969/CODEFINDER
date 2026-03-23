"""Compatibility shim for the relocated legacy SQLite persistence layer."""

from importlib import import_module
import sys


_module = import_module("scripts.legacy.root_legacy.db_persistence")
sys.modules[__name__] = _module
