"""Exact historical Direct111 NumPy controller; emits actions, not live PWM."""
from pathlib import Path
import hashlib
from quadrl.policy import NumpyPolicy

ROOT=Path(__file__).resolve().parents[1]
MODEL_PATH='artifacts/training/main_direct_seed111_v2/best_policy.npz'
MODEL_SHA='5d0d19e02c30c62ed3a0efdcf003d8b6b635ae1eea9b1d32b4737836c46ff900'

class DirectController111(NumpyPolicy):
    def __init__(self,path=None):
        path=Path(path)if path is not None else ROOT/MODEL_PATH
        if hashlib.sha256(path.read_bytes()).hexdigest()!=MODEL_SHA:raise ValueError('Exact historical Direct111 policy required')
        super().__init__(path)
        if self.architecture!='direct':raise ValueError('Direct architecture required')
