import triple_walk_native

def walk_triples(triples_indexed, relation_tail_index,target_nodes, walk_length,padding_idx,seed,restart=True):
    return triple_walk_native.walk_triples(triples_indexed,
                                          relation_tail_index,
                                          target_nodes,
                                          walk_length,
                                          padding_idx,
                                          restart,
                                          seed
                                        )                            

def to_windows_cbow(walks, window_size, num_nodes,seed):
    return triple_walk_native.to_windows_cbow(walks, window_size, num_nodes,seed)

def to_windows_triples_sg(walks, window_size, num_nodes,padding_idx,triples,seed):
    return triple_walk_native.to_windows_triples(walks, window_size,num_nodes,padding_idx,triples,seed)

def to_windows_triples_cbow(walks, window_size, num_nodes,padding_idx,triples,seed):
    return triple_walk_native.to_windows_triples_cbow(walks, window_size,num_nodes,padding_idx,triples,seed)

