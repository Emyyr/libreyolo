"""Keep unguarded ``train()`` / ``val()`` scripts from crashing DataLoader workers.

Under the ``spawn`` and ``forkserver`` start methods (macOS and Windows, and
Linux from Python 3.14) every DataLoader worker re-imports the ``__main__``
script before it runs. A ``model.train()`` or ``model.val()`` call at the top
level of that script, outside ``if __name__ == "__main__":``, then runs again
inside each worker, which tries to start workers of its own and dies with
"An attempt has been made to start a new process before the current process
has finished its bootstrapping phase".

Whether a script is guarded cannot be known in general. This module answers
only the case it can prove: a file-backed ``__main__`` whose module-level
statement that led to the call is reached by a worker re-running the module,
with no ``__name__`` test around it and none before it that could stop the
worker. Everything else (guarded scripts, early exits, ``python -c``, stdin,
notebooks, frozen apps, unusual guard forms) is treated as guarded and left
alone.
"""

from __future__ import annotations

import ast
import inspect
import logging
import multiprocessing
import os
import sys
import tokenize

logger = logging.getLogger(__name__)

# Start methods whose workers re-import the ``__main__`` script.
_REIMPORTING_START_METHODS = frozenset({"spawn", "forkserver"})


def effective_start_method() -> str:
    """The start method DataLoader workers will use, without fixing it."""
    method = multiprocessing.get_start_method(allow_none=True)
    # The first entry is the platform default (documented by multiprocessing).
    return method or multiprocessing.get_all_start_methods()[0]


def _main_module_statement() -> tuple[str, int] | None:
    """Source file and line of the ``__main__`` module-level statement running now."""
    frame = inspect.currentframe()
    try:
        while frame is not None:
            namespace = frame.f_globals
            if (
                frame.f_code.co_name == "<module>"
                and namespace.get("__name__") == "__main__"
            ):
                path = namespace.get("__file__")
                if not isinstance(path, str) or not os.path.isfile(path):
                    return None
                if os.path.realpath(path) != os.path.realpath(frame.f_code.co_filename):
                    return None
                return path, frame.f_lineno
            frame = frame.f_back
        return None
    finally:
        del frame


def _tests_name(node: ast.AST) -> bool:
    return any(
        isinstance(child, ast.Name) and child.id == "__name__"
        for child in ast.walk(node)
    )


def _worker_branch(node: ast.If) -> list[ast.stmt] | None:
    """The branch a re-importing worker takes through a standard main guard.

    ``None`` for any test other than ``__name__ == "__main__"`` or
    ``__name__ != "__main__"`` (either operand order).
    """
    test = node.test
    if not (
        isinstance(test, ast.Compare)
        and len(test.ops) == 1
        and isinstance(test.ops[0], (ast.Eq, ast.NotEq))
    ):
        return None
    operands = [test.left, test.comparators[0]]
    has_name = any(isinstance(o, ast.Name) and o.id == "__name__" for o in operands)
    has_main = any(
        isinstance(o, ast.Constant) and o.value == "__main__" for o in operands
    )
    if not (has_name and has_main):
        return None
    # Workers run with ``__name__ == "__mp_main__"``.
    return node.orelse if isinstance(test.ops[0], ast.Eq) else node.body


def _module_level_nodes(tree: ast.Module):
    """Nodes that run when the module executes (function bodies do not)."""
    stack: list[ast.AST] = list(reversed(tree.body))
    while stack:
        node = stack.pop()
        yield node
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            continue
        stack.extend(reversed(list(ast.iter_child_nodes(node))))


def _is_unguarded(tree: ast.Module, line: int) -> bool:
    """True when a worker re-running the module provably reaches ``line``.

    A ``__name__`` test around the call means guarded (or undecidable). An
    earlier standard guard whose worker branch is empty is passed through;
    any other earlier ``__name__`` test (an early exit, an unusual guard, an
    ``assert``, a bare ``cond and sys.exit()``) is undecidable.
    """
    if not any(stmt.lineno <= line <= (stmt.end_lineno or 0) for stmt in tree.body):
        return False
    # Main-only branches of guards the worker passes through: their contents
    # never run in a worker, so they cannot stop it.
    skipped: list[tuple[int, int]] = []
    for node in _module_level_nodes(tree):
        lineno = getattr(node, "lineno", None)
        if lineno is None or any(start <= lineno <= end for start, end in skipped):
            continue
        if isinstance(node, (ast.If, ast.Assert, ast.While)):
            test = node.test
        elif isinstance(node, ast.Match):
            test = node.subject
        elif isinstance(node, ast.Expr):
            test = node.value
        else:
            continue
        if node.lineno > line or not _tests_name(test):
            continue
        if line <= (node.end_lineno or 0):
            return False
        branch = _worker_branch(node) if isinstance(node, ast.If) else None
        if branch is None or not all(isinstance(s, ast.Pass) for s in branch):
            return False
        main_branch = node.body if branch is node.orelse else node.orelse
        if main_branch:
            skipped.append(
                (main_branch[0].lineno, main_branch[-1].end_lineno or node.lineno)
            )
    return True


def main_script_reruns_in_workers() -> bool:
    """True when DataLoader workers would re-run this call from ``__main__``."""
    if effective_start_method() not in _REIMPORTING_START_METHODS:
        return False
    if getattr(sys, "frozen", False):
        return False
    location = _main_module_statement()
    if location is None:
        return False
    path, line = location
    try:
        with tokenize.open(path) as handle:
            tree = ast.parse(handle.read(), path)
    except (OSError, SyntaxError, ValueError):
        return False
    return _is_unguarded(tree, line)


def resolve_dataloader_workers(
    workers: int,
    *,
    explicit: bool,
    action: str,
    device=None,
) -> int:
    """The DataLoader worker count to use for an unguarded-script-safe call.

    Args:
        workers: The requested or default worker count.
        explicit: Whether the caller set ``workers`` themselves.
        action: The public call, for the warning (``"train()"``, ``"val()"``).
        device: The training device spec; multi-device specs launch DDP, whose
            ranks do not re-import the user's script.
    """
    if not workers or workers <= 0:
        return workers
    from ..training.distributed import has_torchrun_env, parse_device_arg

    if has_torchrun_env() or len(parse_device_arg(device)) > 1:
        return workers
    if not main_script_reruns_in_workers():
        return workers
    method = effective_start_method()
    if explicit:
        logger.warning(
            "%s runs at the top level of the script, outside "
            '`if __name__ == "__main__":`. With the %r multiprocessing start '
            "method every DataLoader worker re-runs that script, so workers=%d "
            "will likely fail. Move the call under "
            '`if __name__ == "__main__":`, or pass workers=0.',
            action,
            method,
            workers,
        )
        return workers
    logger.warning(
        "%s runs at the top level of the script, outside "
        '`if __name__ == "__main__":`. With the %r multiprocessing start '
        "method DataLoader workers re-run that script, which fails for an "
        "unguarded call, so data loads in the main process (workers=0). Move "
        'the call under `if __name__ == "__main__":` to load data with '
        "parallel workers again.",
        action,
        method,
    )
    return 0


__all__ = [
    "effective_start_method",
    "main_script_reruns_in_workers",
    "resolve_dataloader_workers",
]
