"""Machine learning package for the load-forecasting project.

This file exists so that `machine_learning` is unambiguously a PACKAGE.
Without it, `from machine_learning.machine_learning import ...` (see
apps/routes.py) still works at runtime thanks to implicit namespace packages,
but mypy cannot decide whether the file is a top-level module or a submodule
and aborts with:

    Source file found twice under different module names:
    "machine_learning" and "machine_learning.machine_learning"

That aborts type checking entirely (exit code 2), which is why the CI `types`
job failed on something that is not a type error at all.

Having this file makes the package declaration match the import style that is
already in use.
"""
