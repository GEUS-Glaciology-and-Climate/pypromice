from typing import Callable, Mapping, Any, Optional

import attrs

__all__ = [
    "Task",
    "TaskAware",
    "KwargsMap",
]


@attrs.define
class Task:
    value: Any
    stid: Any
    failed: bool = False
    exception: Optional[str] = None


@attrs.define
class TaskAware:
    function: Callable

    def __call__(self, task: Task) -> Task:
        try:
            output = self.function(task.value)
            return Task(value=output, stid=task.stid)
        except Exception as exception:
            return Task(
                value=None,
                stid=task.stid,
                exception=str(exception),
                failed=True,
            )


@attrs.define
class KwargsMap:
    function: Callable

    def __call__(self, kwargs: Mapping):
        return self.function(**kwargs)
