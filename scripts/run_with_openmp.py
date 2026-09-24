"""Run project Python using the existing sklearn-bundled macOS OpenMP runtime.

Avoid modifying global system libraries. Linux uses its normal loader path.
"""
import importlib.util
import os
from pathlib import Path
import sys
if sys.platform=='darwin':
    runtime=Path(importlib.util.find_spec('sklearn').origin).parent/'.dylibs'
    if not (runtime/'libomp.dylib').exists():raise FileNotFoundError('Install an OpenMP runtime required by LightGBM')
    old=os.environ.get('DYLD_LIBRARY_PATH','');os.environ['DYLD_LIBRARY_PATH']=str(runtime)+(os.pathsep+old if old else '')
os.execv(sys.executable,[sys.executable,*sys.argv[1:]])
