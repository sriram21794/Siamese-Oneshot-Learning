echo "Downloading..."

wget -nc https://github.com/brendenlake/omniglot/raw/master/python/images_background.zip  -O data/images_background.zip
wget -nc https://github.com/brendenlake/omniglot/raw/master/python/images_evaluation.zip -O data/images_evaluation.zip


echo "unzipping..."
unzip data/images_background.zip -d data;
unzip data/images_evaluation.zip -d data;