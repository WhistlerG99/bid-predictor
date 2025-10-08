#!/usr/bin/env bash
set -euo pipefail

# Fix ownership on volumes that SageMaker mounts at container start
chown -R trainer:trainer /opt/ml || true
chmod -R a+rwx /opt/ml || true  # be generous; or tighten if your org requires

# Drop privileges and exec the standard sagemaker-training entrypoint
exec gosu trainer /usr/local/bin/train "$@"
# If gosu isn't installed, use:
# su -s /bin/bash -c "/usr/local/bin/train $*" trainer
