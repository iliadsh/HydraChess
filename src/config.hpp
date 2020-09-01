#pragma once

namespace hydra {
    namespace config {
        //UCI identification parameters.
        constexpr const char*   ENGINE_NAME     = "Hydra";
        constexpr const char*   ENGINE_AUTHOR   = "a.big.pickle@gmail.com";
        //MCTS search parameters.
        constexpr int           MCTS_ITERATIONS = 16000;
        constexpr int           THREAD_CNT      = 2;
        constexpr float         C_PUCT          = 0.01;
        //NN training parameters.
        constexpr int           NUM_EPOCH       = 25;
        constexpr int           BATCH_SIZE      = 1024;
        constexpr int           LOG_INTERVAL    = 10;
        constexpr bool          LOAD_CHECKPOINT = false;
        constexpr const char*   WEIGHTS_PATH    = ".\\data\\";
    }
}