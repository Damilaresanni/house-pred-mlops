set -e
cd /opt/house-pred-api/house-pred-mlops/
git fetch origin main
git reset --hard origin/main
source .venv/bin/activate
pip install -r requirements.txt
sudo systemctl restart house-pred-api.service
sleep 2
curl -f http://localhost:8000/health || (echo "Health check failed" && exit 1)

