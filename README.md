

GraphPPepIS

############################################################################

The source code, training and test datasets of paper 'Simultaneous prediction of interaction sites on both protein and peptide sides of complexes through multi-layer graph convolutional networks'

############################################################################


### install dependency:

- conda create -n ppis python==3.7.10 -y

- conda activate ppis

- pip install Bio


### Prerequisites:

* `python`: 3.7.10

* `CUDA`: 10.1

* `pytorch`: 1.2.0


### All relevant inputs：

Due to the limitation of Github, some inputs larger than 25MB are not uploaded. Please contact me directly at 20204227031@stu.suda.edu.cn.


############################################################################

### Reproduce experimental results: 

(settrain: Train4094, settest: Test169, settest2: Test53)

## Test our model GraphPPepIS

Step 1: ```cd code```

Step 2: ```python predict_graph.py'``

## Test our model SeqPPepIS

Step 1: ```cd code```

Step 2: ```python predict_seq.py'``

## Train your own model GraphPPepIS

Step 1: ```cd code```

Step 2: ```python train_graph.py --layers 8 --units 512 --epochs 300 --pw 0.8 --lw 0.1```

## Train your own model SeqPPepIS

Step 1: ```cd code```

Step 2: ```python train_seq.py --layers 8 --units 512 --epochs 300 --pw 0.8 --lw 1.0```




