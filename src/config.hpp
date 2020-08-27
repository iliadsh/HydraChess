#pragma once

namespace hydra {
    namespace config {
        //UCI identification parameters.
        constexpr const char*   ENGINE_NAME     = "Hydra";
        constexpr const char*   ENGINE_AUTHOR   = "a.big.pickle@gmail.com";
        //MCTS search parameters.
        constexpr int           MCTS_ITERATIONS = 1600;
        constexpr int           THREAD_CNT      = 8;
        //NN training parameters.
        constexpr int           NUM_EPOCH       = 20;
        constexpr int           BATCH_SIZE      = 1024;
        constexpr int           LOG_INTERVAL    = 1;
        constexpr bool          LOAD_CHECKPOINT = false;
        constexpr const char*   WEIGHTS_PATH    = "./data/model.pt";
    }
}