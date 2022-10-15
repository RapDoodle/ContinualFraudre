# Graud Detection with GNN

The code of this project is based on [Streaming Graph Neural Networks via Continual Learning](https://dl.acm.org/doi/abs/10.1145/3340531.3411963)（CIKM 2020).

### Usages

* ContinualGNN (proposed model) on Cora:
```
cd src/
python main_stream.py --data=cora --new_ratio=0.8 --memory_size=250 --ewc_lambda=80.0 
```
* OnlineGNN (lower bound) on Cora:
```
python main_stream.py --data=cora
```

If using cuda, set `--cuda`.
