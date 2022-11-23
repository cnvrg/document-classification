pip install --upgrade pip
apt-get -y update
apt-get install -y git
apt-get install -y libgl1-mesa-dev
pip uninstall -y pillow
pip install --no-cache-dir pillow
apt-get install -y libpangocairo-1.0-0
apt-get -y install python-dev libxml2-dev libxslt1-dev antiword unrtf poppler-utils pstotext tesseract-ocr \
flac ffmpeg lame libmad0 libsox-fmt-mp3 sox libjpeg-dev swig libpulse-dev