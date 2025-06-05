# Datasets

## UCF-101

Download [UCF-101](https://www.crcv.ucf.edu/data/UCF101.php) dataset.

## Kinetics-600

1. Download Kinetics-600:

```
bash data/k600_downloader.sh
bash data/k600_extractor.sh
```

2. Resize the videos to $128 \times 128$ resolution, which can significantly save the training time.

```
python data/resize_videos.py \
        data/k600/train_raw data/k600/train \
        --dense --level 1 --scale 128 --num-worker 256

python data/resize_videos.py \
        data/k600/val_raw data/k600/val \
        --dense --level 1 --scale 128 --num-worker 256
```

## Set up the datasets

We put all data into the `./data` directory, as follows:

```
├── data
    ├── metadata
    ├── ucf101
    |   ├── ApplyEyeMakeup
    |   ├── ApplyLipstick
    |   └── ...
    └── k600
        ├── train
        └── val
```

