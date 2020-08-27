#pragma once

#include "neural.hpp"
#include "dataset.hpp"
#include "config.hpp"

namespace hydra {
    /**
     * Train the value network on game results.
     * @param {ValueNetwork} net - the value network to train.
     * @param {const std::string&} path - path to the dataset.
     */
    void train(ValueNetwork net, const std::string& path);
}