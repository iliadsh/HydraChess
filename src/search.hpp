#pragma once

#include <unordered_map>
#include <memory>
#include <random>
#include <Position.h>
#include "neural.hpp"

namespace hydra {
    /**
     * A node in the search tree. Holds total evaluation, number of visits, and the child node pointers.
     */
    struct MCTS_Node {
        bool visited                                                                        {  false  }; //visited flag
        float w                                                                             {    0    }; //total action
        float n                                                                             {    0    }; //visit count
        MCTS_Node* parent                                                                   { nullptr }; //parent node*
        libchess::MoveList move_list;                                                                    //move list cache
        libchess::Position::hash_type position_hash;                                                     //position hash
        std::unordered_map<libchess::Move::value_type, std::unique_ptr<MCTS_Node>> children;             //child nodes

        /**
         * Calculates the UCT value of the search node.
         */ 
        float UCT() const {
            float Q = w / n;
            float U = config::C_PUCT * sqrtf(parent->n) / (n + 1);
            return Q + U;
        }
    };

    class MCTSearch {
        private:
            /**
             * Search tree root.
             */
            std::unique_ptr<MCTS_Node> search_root;

            /**
             * Value network.
             */ 
            Eval value_net;

            /**
             * Random engine.
             */
            std::mt19937 mt_eng;

            /**
             * Determines the value of the current node statically using the value network.
             * @param {libchess::Position&} pos - The current board state.
             * @returns {float} The value of the node.
             */
            float rollout(libchess::Position& pos);

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
            float mcts_search(libchess::Position& pos, std::unique_ptr<MCTS_Node>& search_node, bool explore);

        public:
            /**
             * Performs several iterations of MCTS and then chooses the optimal move.
             * @param {libchess::Position&} pos - The current position.
             * @param {const bool&} stopped_flag - A flag used for terminating the search early.
             * @param {int&} out_score - The predicted score from the network.
             * @returns {libchess::Move} The optimal move.
             */
            libchess::Move choose_best_move(libchess::Position& pos, const bool& stopped_flag, int& out_score);

            /**
             * Shift tree root down and destroys all other branches of tree.
             * @param {libchess::Move::value_type} move - the move to shift tree down by.
             * @returns {bool} true on success.
             */ 
            bool shift_tree_down(libchess::Move::value_type move);

            MCTSearch();

            ~MCTSearch();
    };
}