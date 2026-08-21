#!/bin/bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate xtts
cd /root
exec python -m uvicorn xtts_server:app --host 0.0.0.0 --port 8020
