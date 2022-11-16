# Fraud Detection with GNN

The code of this project is based on [Streaming Graph Neural Networks via Continual Learning](https://dl.acm.org/doi/abs/10.1145/3340531.3411963)（CIKM 2020).

## Usage 

### Installation
Install dependencies
```
pip install -r requirements.txt
```

Install PyTorch 1.12.1 (CPU only)
```bash
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 -c pytorch
```

With CUDA support
```bash
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge
```

### Generate dataset
Please first download the Amazon Review Dataset (2014) from [here](http://jmcauley.ucsd.edu/data/amazon/index_2014.html), and place the json files under `/data/origin/`.

Generate the preprocessed dataset. Here, we are using `reviews_Amazon_Instant_Video_5.json` as an example.
```
python preprocess.py --filename=reviews_Amazon_Instant_Video_5.json --dataset-name=amazon_instant_video --num-streams=15 --corpus-sim-percentile=95 --usu-interval=259200 --total-features=2000 --num-features=1000 --fradulent-threshold=0.5 --feature-schema=sentence_embeddings
```

### Training
```
python main_stream.py --data=amazon_instant_video --new-ratio=0.6 --memory-size=1000 --ewc-lambda=80.0 --max-detect-size=16 --batch-size=64 --num-epochs=75
```

## Cora

ContinualGNN (proposed model) on Cora:
```
cd src/
python main_stream.py --data=cora --new-ratio=0.8 --memory-size=250 --ewc-lambda=80.0 
```
OnlineGNN (lower bound) on Cora:
```
python main_stream.py --data=cora
```

If using cuda, set `--cuda`.