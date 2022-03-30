> A simple algorithm to learn embeddings of entities in knowledge graph.

## What is it
TripleWalk is an algorithm to learn vector embeddings of entities in a knowledge graph by performing random walks on triples.

## Installation

***Please not this package is only available for python 3.8+ and Pytorch >= 1.9.0***

#### Install from PyPI
``` Python
pip install triple_walk
```

#### Install from Github
``` bash
pip install git+https://github.com/udel-cbcb/triple_walk.git#egg=triple_walk
```


# Triple Walk
Author : Sachin Gavali

## Requirements
```
1. Pytorch >= 1.9.0
2. NVIDIA-GPU (Cuda Toolkit >= 11.4
3. AMD-GPU (ROCM == 4.0.1)
4. Python == 3.8
```

## Examples
* SkipGram Triple Walk model : [SkipGramTriple](examples/skipgram_triple_walk.py)
* CBOW Triple Walk model : [CBOWTriple](examples/skipgram_triple_walk.py)


