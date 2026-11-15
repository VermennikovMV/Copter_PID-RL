"""Competent PID and PID+DOB share exactly one classical implementation.

The main PID disables the disturbance observer. Gain selection is development
work and must be frozen before final held-out testing.
"""
import json
from pathlib import Path
from reference.dob import DOBConfig,DOBReference


def read_pid_config(path=None):
    values=json.loads(Path(path or Path(__file__).with_name("pid_config.json")).read_text())
    values.pop("status",None)
    values.pop("selection",None)
    values["enabled"]=False
    return DOBConfig(**values)


class CascadedPID(DOBReference):
    architecture="pid"
    def __init__(self,dt=.02,config=None):
        if config is not None and config.enabled:
            raise ValueError("main PID must disable the disturbance observer; use DOBReference for PID+DOB")
        super().__init__(dt,config or read_pid_config())

