from tkinter import W
from triple_walk import utils
from triple_walk import rw
from triple_walk.model import CBOWTriple, SkipGramTriple
import torch
import numpy as np
import pandas as pd

# params
walk_length = 100
window_size = 20
walks_per_node = 100
amplify_triples = 100000
num_negatives = 20
epochs = 2

# triples
triples_list = [
    ("A","r1","B"),
    ("B","r2","D"),
    ("A","r1","C"),
    ("C","r2","E"),
    ("C","r3","B"),
    ("A","r2","D"),
    ("D","r3","A"),
    ("D","r2","C")
]

# load the triple list into a dataframe
triple_list_pd = pd.DataFrame(data=triples_list,columns=["head","relation","tail"])

# convert to indexed triples
triples_index, entities_map,relations_map = utils.to_indexed_triples(triple_list_pd)

# convert to torch tensor
triples_index_tensor = torch.from_numpy(triples_index)
triples_index_tensor = triples_index_tensor.repeat_interleave(amplify_triples,0)

# get target nodes
target_entities_list = list(set(triples_index_tensor[:,0].tolist()+triples_index_tensor[:,2].tolist()))
target_entities_tensor = torch.Tensor(target_entities_list).to(int)

target_entities_tensor = target_entities_tensor.repeat_interleave(walks_per_node)

# build node edge index
relation_tail_index,triples_index_tensor_sorted = utils.build_relation_tail_index(triples_index_tensor,target_entities_tensor)

# create a list of all entities
all_entities_list = list(entities_map.values()) + list(relations_map.values())

# sort the list
all_entities_list.sort()

# create the padding index
padding_idx = all_entities_list[-1] + 1

# move tensors to gpu
triples_index_tensor_sorted = triples_index_tensor_sorted.cuda()
relation_tail_index = relation_tail_index.cuda()
target_entities_tensor = target_entities_tensor.cuda()

# training loop
from tqdm import tqdm
for i in tqdm(range(epochs)):

    # perform walk
    walks = rw.walk_triples(triples_indexed=triples_index_tensor_sorted,
                            relation_tail_index=relation_tail_index,
                            target_nodes=target_entities_tensor,
                            walk_length=walk_length,
                            seed=10,
                            padding_idx=padding_idx,
                            restart=False
                            )

    # amplify walks
    walks = walks.repeat_interleave(num_negatives,0)

    # split walk to windows
    target_triples,pos_context,neg_context = rw.to_windows_triples_sg(walks=walks,
                                                                    window_size=window_size,
                                                                    num_nodes=len(all_entities_list),
                                                                    padding_idx=padding_idx,
                                                                    triples=triples_index_tensor_sorted,
                                                                    seed=20)