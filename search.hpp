#pragma once

#include <unordered_map>
#include <memory>
#if defined(_OPENMP)
    #include <omp.h>
#endif
#include <Position.h>

namespace pegasus {
    /**
     * A node in the search tree. Holds total evaluation, number of visits, and the child node pointers.
     */
    struct MCTS_Node {
        bool init   { false };
        float w     {   0   };
        float n     {   0   };
        std::unordered_map<libchess::Move::value_type, std::unique_ptr<MCTS_Node>> children;
    };

    /**
     * A MCTS rollout determines the value of a node through random play.
     * @param {libchess::Position} pos - A copy of the current board state.
     * @returns {float} The value of the node.
     */
    float rollout(libchess::Position pos);

    /**
     * One iteration of MCTS goes through 4 stages.
     * 1) selection: traverse down tree nodes which maximize UCT.
     * 2) expansion: if a selected node is unexplored, add it to the search tree.
     * 3) simulation: perform a rollout on the the new node to determine its value.
     * 4) back-propogation: send the statistics up the search three.
     * @param {libchess::Position&} pos - The current board state.
     * @param {std::unique_ptr<MCTS_Node>&} search_node - The search node in the tree.
     * @returns {float} Terminal node evaluation.
     */
    float mcts_search(libchess::Position& pos, std::unique_ptr<MCTS_Node>& search_node);

    /**
     * Performs several iterations of MCTS and then chooses the optimal move.
     * @param {libchess::Position&} pos - The current position.
     * @param {const bool&} stopped_flag - A flag used for terminating the search early.
     * @param {int&} out_score - The predicted score from the network.
     * @returns {libchess::Move} The optimal move.
     */
    libchess::Move choose_best_move(libchess::Position& pos, std::unique_ptr<MCTS_Node>& root_node, const bool& stopped_flag, int& out_score);
}