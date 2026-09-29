from __future__ import annotations

try:
    from multiprocessing import resource_tracker as _resource_tracker
except ImportError:
    _resource_tracker = None


if _resource_tracker is not None:
    _resource_tracker_register_impl = _resource_tracker.register
    _resource_tracker_unregister_impl = _resource_tracker.unregister

    def _resource_tracker_register(name: str, rtype: str) -> None:
        if rtype == "shared_memory":
            return
        _resource_tracker_register_impl(name, rtype)

    def _resource_tracker_unregister(name: str, rtype: str) -> None:
        if rtype == "shared_memory":
            return
        _resource_tracker_unregister_impl(name, rtype)

    _resource_tracker.register = _resource_tracker_register
    _resource_tracker.unregister = _resource_tracker_unregister
else:
    _resource_tracker_unregister = None
