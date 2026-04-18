#!/usr/bin/env bash
set -euo pipefail

cd /workspace/mlagent
export PYTHONPATH="/workspace/mlagent:${PYTHONPATH:-}"

python -m agent.agent_framework "$@"
