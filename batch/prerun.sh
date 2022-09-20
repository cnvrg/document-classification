pip3 install --no-build-isolation -U pypdfium2==1.0.0

pip install --upgrade pip
apt-get -y update
apt-get install -y git
apt-get install -y libgl1-mesa-dev
pip uninstall -y pillow
pip install --no-cache-dir pillow
apt-get install -y libpangocairo-1.0-0