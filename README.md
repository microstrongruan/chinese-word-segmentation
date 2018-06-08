# Chinese Word Segmentation

Class project

# Requirement
1. tensorflow 1.8

# Usage

1. set train, validation, test file in train.sh, infer.sh and score.sh
2. set pythonpath and cuda path in train.sh, infer.sh and score.sh

```
sh train.sh
chmod +x infer.sh
./infer
chmod +x score.sh
./score.sh
```

# Model
please reference to 
Long short-term memory neural networks for chinese word segmentation
http://www.aclweb.org/anthology/D15-1141

model5 is an extendtion to bi-directional lstm.

# Results
parameters:  
vocab_size = hidden_size = 1000  
learning_rate = 1e-4 without decay  
optimization = Adam  
dropout = 0.2  

PKU:  

|Model|R|P|F| 
|-|-|-|-| 
|Model1|75.1|76.3|75.7|
|Model2|75.6|77.5|76.5|
|Model3|86.5|87.9|87.2|
|Model4|90.5|92.3|91.4|
|Model5|90.5|92.3|91.4|
  
MSRA:  

|Model|R|P|F|
|-|-|-|-|
|Model1|79.8|79.5|79.7|
|Model2|79.9|79.9|79.9|
|Model3|93.3|92.9|93.1|
|Model4|95.3|95.1|95.2|
|Model5|93.1|93.8|93.5|
  
CTB6:  

|Model|R|P|F|
|-|-|-|-|
|Model1|79.2|77.7|78.4|
|Model2|79.2|78.9|79.1|
|Model3|91.1|90.5|90.8|
|Model4|93.9|93.6|93.7|
|Model5|91.6|91.1|91.4|

# Checkpoints and more results
https://pan.baidu.com/s/15Lr0CXmSIVEG2peFangL9w
