set -e
cd /opt/house-pred-api/house-pred-mlops/
git fetch origin main
git reset --hard origin/main
python3 -m venv deploy
source deploy/bin/activate
pip install -r requirements.txt
which python
pip list
sudo systemctl restart house-pred-api.service
sleep 2
curl -f http://localhost:8000/health || (echo "Health check failed" && exit 1)

