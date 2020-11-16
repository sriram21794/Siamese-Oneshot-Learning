# Siamese-Oneshot-Learning
Implementation of [ Siamese Neural Networks for One-shot Image Recognition](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf)


## Steps for running this project

#### Download Omniglot Dataset

All relevant background and evaluation alphabets will be downloaded to `./data` directory. All you've to do is run below shell script  

```
sh download.sh
```

#### Install all the relevant libraries

```
pip install -r requirements.txt
```

#### Define right configurations in  `params.json`

```json
{
    "target_size": [150, 150, 1],
    "batch_size": 64,
    "white_list_formats": ["png", "jpg", "jpeg", "bmp", "ppm", "tif", "tiff"],
    "nb_epochs": 10,
    "seed": 22,
    "log_dir": "./logs",
    "background_samples": 19280, 
    "evaluation_samples": 13180
}
```

#### Run the training script


```
python main.py train -c params.json \
                     -b ./data/images_background/ \ 
                     -e ./data/images_evaluation/ \ 
                     -m ./saved_model
```


