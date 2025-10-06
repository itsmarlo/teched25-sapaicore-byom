cat > serving/serve.sh <<'SH'
#!/usr/bin/env bash
set -e
echo "MODEL_PATH=${MODEL_PATH:-/mnt/models}"
echo "PORT=${PORT:-8080}"
exec uvicorn serving:app --host 0.0.0.0 --port "${PORT:-8080}"
SH
chmod +x serving/serve.sh
