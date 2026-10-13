# Data

This project uses the BDD100K dataset (Kaggle mirror):
https://www.kaggle.com/datasets/marquis03/bdd100k

The dataset is not included in this repository because of its size.

## Download

```python
import kagglehub
path = kagglehub.dataset_download("marquis03/bdd100k")
```

## Prepare

Run `python src/prepare_data.py` to create the image links and convert
the labels to YOLO format.

## Classes

car, truck, bus, motor, bike, person, rider, train
