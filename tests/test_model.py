import unittest
from triple_walk import utils
from triple_walk import rw
from triple_walk.model import CBOWTriple, SkipGramTriple
import torch
import numpy as np
import pandas as pd


class ModelTest(unittest.TestCase):
    
    def test_model_cbow(self):
    
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
        pos_triples, neg_triples, context = rw.to_windows_triples_cbow(walks=walks,
                                                                        window_size=4,
                                                                        num_nodes=30,
                                                                        padding_idx=padding_idx,
                                                                        triples=triples_index_tensor_sorted,
                                                                        seed=20)


        # create the model
        model = CBOWTriple(num_nodes=len(all_entities_list),
                                            embedding_dim=32,
                                            padding_index=padding_idx
                                        )

        # train the model for one step
        loss = model(pos_triples,neg_triples,context)

        assert loss != 0, "loss cannot be zero"


    def test_model_skipgram(self):
    
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

        assert loss != 0, "loss cannot be zero"

        


