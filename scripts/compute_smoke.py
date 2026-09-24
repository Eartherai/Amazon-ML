"""Print remote runtime capability; reads no competition data and trains nothing."""
import json,platform,shutil,subprocess
print(json.dumps({'python':platform.python_version(),'platform':platform.platform(),'nvidia_smi':bool(shutil.which('nvidia-smi'))}))
if shutil.which('nvidia-smi'):
    subprocess.run(['nvidia-smi','--query-gpu=name,memory.total','--format=csv,noheader'],check=True,timeout=20)
