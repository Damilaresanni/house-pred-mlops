set -e
cd /opt/house-pred-api/house-pred-mlops/
git fetch origin main
git reset --hard origin/main
python3 -m venv server
source server/bin/activate
pip install -r requirements.txt
sudo systemctl restart house-pred-api
sleep 2
curl -f http://localhost:8000/health || (echo "Health check failed" && exit 1)
echo "This deployment script run success"

