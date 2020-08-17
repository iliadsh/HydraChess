#include "search.hpp"
#include "config.hpp"

namespace pegasus {
    std::unique_ptr<MCTS_Node> search_root = std::make_unique<MCTS_Node>();
    std::random_device rd;
    std::mt19937 mt_eng{rd()};

    float rollout(libchess::Position pos) {
        libchess::Color curr_player = pos.side_to_move();
        while (pos.game_state() == libchess::Position::GameState::IN_PROGRESS) {
            //make random moves until termination
            libchess::MoveList move_list = pos.legal_move_list();
            bool found_capture{false};
            for (const auto& move : move_list) {
                if (pos.is_capture_move(move)) {
                    found_capture = true;
                    pos.make_move(move);
                    break;
                }
            }
            if (!found_capture) {
                std::uniform_int_distribution<int> distribution(0, static_cast<int>(move_list.size() - 1));
                pos.make_move(*(move_list.begin() + distribution(mt_eng)));
            }
        }
        if (pos.game_state() == libchess::Position::GameState::CHECKMATE) {
            //checkmated player is the side to move
            return (pos.side_to_move() == curr_player ? -1.0f : 1.0f);
        }
        else {
            return 0;
        }
    }

    float mcts_search(libchess::Position& pos, std::unique_ptr<MCTS_Node>& search_node) {
        //static evaluations
        if (pos.game_state() == libchess::Position::GameState::CHECKMATE) {
            return -1;
        }
        else if (pos.game_state() != libchess::Position::GameState::IN_PROGRESS) {
            return 0;
        }

        //newly expanded node, setup stats and perform rollout
        if (!search_node->init) {
            search_node->init = true;

            //perform rollouts. does several in parallel if OpenMP is supported
            #if defined(_OPENMP)
                float v_s[config::OMP_ROLLOUT_CNT];
                #pragma omp parallel for
                for (int i = 0; i < config::OMP_ROLLOUT_CNT; i++) {
                    v_s[i] = -rollout(pos);
                }
                float v_avg{0};
                for (int i = 0; i < config::OMP_ROLLOUT_CNT; i++) {
                    v_avg += v_s[i];
                }
                v_avg /= config::OMP_ROLLOUT_CNT;
            #else
                float v_avg = -rollout(pos);
            #endif
            
            search_node->w = v_avg;
            search_node->n = 1;
            search_node->position_hash = pos.hash();
            search_node->move_list = pos.legal_move_list();
            return -search_node->w;
        }

        //choose next move which maximizes the UCT
        float max_uct = -INFINITY;
        libchess::Move best_move;
        for (const auto& move : search_node->move_list) {
            auto& child = search_node->children.find(move.value_sans_type());
            //if the node is unexplored
            if (child == search_node->children.end()) {
                max_uct = INFINITY;
                best_move = move;
                break;
            }
            float uct = child->second->UCT();
            if (uct > max_uct) {
                max_uct = uct;
                best_move = move;
            }
        }

        //continue selection (or expansion if leaf node)
        auto& next_node = search_node->children[best_move.value_sans_type()];
        if (next_node == nullptr) {
            next_node = std::make_unique<MCTS_Node>();
            next_node->parent = search_node.get();
        }
        pos.make_move(best_move);
        float v = mcts_search(pos, next_node);
        pos.unmake_move();

        //back propogate stats
        search_node->w += v;
        search_node->n++;
        return -v;
    }

    libchess::Move choose_best_move(libchess::Position& pos, std::unique_ptr<MCTS_Node>& root_node, const bool& stopped_flag, int& out_score) {
        //validate tree cache
        if (root_node->position_hash != pos.hash()) {
            root_node = std::make_unique<MCTS_Node>();
        }
        //perfrom iterations of MCTS
        for (int iter = 0; iter < config::MCTS_ITERATIONS && !stopped_flag; iter++) {
            mcts_search(pos, root_node);
        }
        //choose move with highest visit count
        float max_n = -INFINITY;
        libchess::Move best_move;
        for (auto& child : root_node->children) {
            float n = child.second->n;
            if (n > max_n) {
                max_n = n;
                best_move = libchess::Move(child.first);
            }
        }

        //calculate predicted score
        out_score = static_cast<int>(-root_node->w / root_node->n * 100);

        //move search tree down to chosen node
        root_node = std::move(root_node->children[best_move.value_sans_type()]);
        return best_move;
    }
}