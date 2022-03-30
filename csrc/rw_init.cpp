#include <torch/extension.h>
#include "cpu/rw_cpu.h"
#include "cpu/rw_cpu_edge_list.h"
#include "cuda/rw_cuda_edge_list.h"
#include "cuda/rw_cuda.h"
#include "cpu/windows_cpu.h"
#include "cuda/windows_cuda.h"
#include "cpu/rw_cpu_triples.h"
#include "cuda/rw_cuda_triples.h"


torch::Tensor walk_triples(const torch::Tensor *triples_indexed,
                  const torch::Tensor *relation_tail_index,
                  const torch::Tensor *target_nodes,
                  const int walk_length,
                  const int64_t padding_idx,
                  const bool restart,
                  const int seed
                )
{

  if(target_nodes->device().is_cuda()) {
    return triples::walk_triples_gpu(triples_indexed,
                                    relation_tail_index,
                                    target_nodes,
                                    walk_length,
                                    padding_idx,
                                    restart,
                                    seed);
  }else{
    return triples::walk_triples_cpu(triples_indexed,
                                     relation_tail_index,
                                     target_nodes,
                                     walk_length,
                                     padding_idx,
                                     restart,
                                     seed);
  }
  
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> to_windows_triples(const torch::Tensor *walks,
                                      const int window_size,
                                      const int64_t num_nodes,
                                      const int64_t padding_idx,
                                      const torch::Tensor *triples,
                                      const int seed
                                    )
{
  if(walks->device().is_cuda()) {
    return to_windows_triples_gpu(walks,window_size,num_nodes,padding_idx,triples,seed);
  }else{
    return to_windows_triples_cpu(walks,window_size,num_nodes,padding_idx,triples,seed);
  }
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> to_windows_triples_cbow(const torch::Tensor *walks,
                                      const int window_size,
                                      const int64_t num_nodes,
                                      const int64_t padding_idx,
                                      const torch::Tensor *triples,
                                      const int seed
                                    )
{
  if(walks->device().is_cuda()) {
    return to_windows_triples_cbow_gpu(walks,window_size,num_nodes,padding_idx,triples,seed);
  }else{
    return to_windows_triples_cbow_cpu(walks,window_size,num_nodes,padding_idx,triples,seed);
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("walk_triples", &walk_triples, "walk_triples");
  m.def("to_windows_triples",&to_windows_triples,"to_windows_triples");
  m.def("to_windows_triples_cbow",&to_windows_triples_cbow,"to_windows_triples_cbow");
}
