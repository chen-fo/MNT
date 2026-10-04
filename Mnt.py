"""Import entry for MNT v0. Install with: python -m pip install -e .

The new environment uses the PettingZoo Parallel API (per-agent dictionaries).
The original navigation prototype is preserved in legacy/Mnt_navigation.py.
"""

from mnt_env import MntConfig, MntEnv, parallel_env

Mnt = MntEnv

__all__ = ["Mnt", "MntEnv", "MntConfig", "parallel_env"]
