#pragma once

namespace pegasus {
    namespace config {
        constexpr const char*   ENGINE_NAME     = "Hydra";
        constexpr const char*   ENGINE_AUTHOR   = "a.big.pickle@gmail.com";
        constexpr int           MCTS_ITERATIONS = 4000;
        constexpr int           OMP_ROLLOUT_CNT = 8;
    }
}