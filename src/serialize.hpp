#pragma once

#include <torch/torch.h>
#include <Position.h>

namespace hydra {
    /**
     * Serialize a board position to a tensor.
     * @param {libchess::Position} pos - board position.
     * @returns {torch::Tensor} tensor.
     */ 
    torch::Tensor serialize(libchess::Position pos);
}