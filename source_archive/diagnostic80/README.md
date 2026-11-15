# WORKSPACE_ONLY source evidence

Exact original v16 reward/environment/trainer, runner and audit/table-builder
sources are supplied for review of what produced the diagnostic result. Their
original workspace evidence/import closure and external Gazebo stack are not
included here. These files are not a self-contained v16 training command.

The small portable guarded capsule reproduces the earlier v8 preparation only;
the 80-input policy has a separate pure inference entry point:
`python scripts/inspect_diagnostic80.py`. It is not the default 77-input actor.
