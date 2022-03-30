from triple_walk import utils
from triple_walk import rw
from triple_walk.model import CBOWTriple, SkipGramTriple
import torch
import numpy as np
import pandas as pd

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

# get target nodes
target_entities_list = list(set(triples_index_tensor[:,0].tolist()+triples_index_tensor[:,2].tolist()))
target_entities_tensor = torch.Tensor(target_entities_list).to(int)

# build node edge index
relation_tail_index,triples_index_tensor_sorted = utils.build_relation_tail_index(triples_index_tensor,target_entities_tensor)

# create a list of all entities
all_entities_list = list(entities_map.values()) + list(relations_map.values())

# sort the list
all_entities_list.sort()

# create the padding index
padding_idx = all_entities_list[-1] + 1

# perform walk
walks = rw.walk_triples(triples_indexed=triples_index_tensor_sorted,
                        relation_tail_index=relation_tail_index,
                        target_nodes=target_entities_tensor,
                        walk_length=6,
                        seed=10,
                        padding_idx=padding_idx,
                        restart=False
                        )

# split walk to windows
target_triples,pos_context,neg_context = rw.to_windows_triples_sg(walks=walks,
                                                                window_size=4,
                                                                num_nodes=30,
                                                                padding_idx=padding_idx,
                                                                triples=triples_index_tensor_sorted,
                                                                seed=20)


# create the model
model = SkipGramTriple(num_nodes=len(all_entities_list),
                                    embedding_dim=32,
                                    padding_index=padding_idx
                                )

# train the model for one step
loss = model(target_triples,pos_context,neg_context)

# get head embeddings
head_embedding = model.get_head_embedding(to_cpu=False)

# to data frame
head_embedding_filtered = head_embedding[list(entities_map.values())]
head_embedding_df = pd.DataFrame(data=head_embedding_filtered)
head_embedding_df.insert(0,"entity",list(entities_map.keys()))


# get tail embeddings
tail_embedding = model.get_tail_embedding(to_cpu=False)

# to data frame
tail_embedding_filtered = tail_embedding[list(entities_map.values())]
tail_embedding_df = pd.DataFrame(data=tail_embedding_filtered)
tail_embedding_df.insert(0,"entity",list(entities_map.keys()))