# L2CL: Embarrassingly Simple Layer-to-Layer Contrastive Learning for Graph Collaborative Filtering

## Requirements
```
python>=3.9.18
pytorch>=1.13.1
torch-geometric>=2.4.0
torch-sparse>=0.6.17+pt113cu117
numpy>=1.26.1
pandas>=2.1.2
CUDA 11.7
```

## Installation
```
conda create -n L2CL python=3.9
conda activate L2CL

pip install torch==1.13.1+cu117 -f https://download.pytorch.org/whl/torch/
pip install torch-sparse==0.6.17 -f https://pytorch-geometric.com/whl/torch-1.13.1+cu117.html
pip install torch-scatter==2.1.1 -f https://pytorch-geometric.com/whl/torch-1.13.1+cu117.html
pip install torch-geometric
pip install -r requirements.txt
```

## Dataset

| Datasets    | #Users | #Items | #Interactions | Density  |
|-------------|--------|--------|---------------|----------|
| Kindle      | 60,468 | 57,212 | 880,859       | 0.00025  |
| Yelp        | 45,477 | 30,708 | 1,777,765     | 0.00127  |
| Books       | 58,144 | 58,051 | 2,517,437     | 0.00075  |
| QB-video    | 30,323 | 25,730 | 1,581,136     | 0.00203  |

For `amazon-kindle-store`, `yelp`, `amazon-books`, they will be automatically downloaded via RecBole once you run the main program.

For `QB-video`, we provide it under dataset/
```
cd dataset
unzip QB-video.zip
```


## Reproduction

We integrate our [L2CL](./recbole_gnn/model/general_recommender/l2cl.py) method into the [RecBole](https://recbole.io/) and [RecoBole-GNN](https://github.com/RUCAIBox/RecBole-GNN) framework.

You can reproduct our experiment results through below instructions:

#### Amazon-kindle-store
```
bash scripts/run_kindle.sh
```

#### Yelp
```
bash scripts/run_yelp.sh
```

#### Amazon-books
```
bash scripts/run_books.sh
```

#### QB-video
```
bash scripts/run_video.sh
```

