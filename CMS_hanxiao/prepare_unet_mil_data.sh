#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/zhaona/CMS_hanxiao
exec /home/zhaona/anaconda3/envs/ljhtorch/bin/python "$ROOT/prepare_unet_mil_data.py" "$@"
