#pragma once

#include <torch/torch.h>
#include "config.hpp"

namespace hydra {
    /**
     * The actual value network. Takes in a 12x64+4 board state as input
     * and returns a single scalar from [-1, 1] as the predicted score.
     */ 
    struct EvalImpl : public torch::nn::Module {
        EvalImpl()
            : fc1(12 * 64 + 4, 2048),
              drop1(torch::nn::DropoutOptions().p(0.2)),
              fc2(2048, 2048),
              drop2(torch::nn::DropoutOptions().p(0.2)),
              fc3(2048, 2048),
              drop3(torch::nn::DropoutOptions().p(0.2)),
              fc4(2048, 1)
        {
            register_module("fc1", fc1);
            register_module("drop1", drop1);
            register_module("fc2", fc2);
            register_module("drop2", drop2);
            register_module("fc3", fc3);
            register_module("drop3", drop3);
            register_module("fc4", fc4);
        }

        torch::Tensor forward(torch::Tensor x) {
            x = drop1(torch::relu(fc1(x)));
            x = drop2(torch::relu(fc2(x)));
            x = drop3(torch::relu(fc3(x)));
            x = torch::tanh(fc4(x));
            return x;
        }
        
        torch::nn::Linear fc1, fc2, fc3, fc4;
        torch::nn::Dropout drop1, drop2, drop3;
    };
    TORCH_MODULE(Eval);
}