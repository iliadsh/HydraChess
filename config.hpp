#pragma once

namespace pegasus {
    namespace config {
        constexpr const char*   ENGINE_NAME     = "PegaZero";
        constexpr const char*   ENGINE_AUTHOR   = "a.big.pickle@gmail.com";
        constexpr int           MCTS_ITERATIONS = 2000;
        constexpr int           OMP_ROLLOUT_CNT = 8;
    }
}