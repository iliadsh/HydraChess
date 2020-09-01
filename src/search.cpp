#include "search.hpp"
#include "config.hpp"
#include "serialize.hpp"

namespace hydra {
    MCTSearch::MCTSearch() {
        search_root = std::make_unique<MCTS_Node>();
        std::random_device rd;
        mt_eng = std::mt19937{rd()};
        torch::load(value_net, std::string(config::WEIGHTS_PATH) + "evaluator.pt");
        value_net->eval();
        value_net->to(at::kCUDA);
    }

    MCTSearch::~MCTSearch() {}

    float MCTSearch::rollout(libchess::Position& pos) {
        //NN evaluation
        torch::Tensor pos_tensor = serialize(pos).unsqueeze(0).to(at::kCUDA);
        float val = value_net->forward(pos_tensor)[0][0].item<float>();
        return val;
    }

    float MCTSearch::mcts_search(libchess::Position& pos, std::unique_ptr<MCTS_Node>& search_node, bool explore) {
        //static evaluations
        if (pos.game_state() != libchess::Position::GameState::IN_PROGRESS) {
            float v{0};
            if (pos.game_state() == libchess::Position::GameState::CHECKMATE) {
                v = 10;
            }
            else {
                v = 0;
            }
            search_node->w += v;
            search_node->n++;
            return -v;
        }

        //newly expanded node, setup stats and perform rollout
        if (!search_node->visited) {
            search_node->visited = true; 
            search_node->w = -rollout(pos);
            search_node->n = 1;
            search_node->position_hash = pos.hash();
            search_node->move_list = pos.legal_move_list();
            return -search_node->w;
        }

        //choose next move which maximizes the UCT
        float max_uct = -INFINITY;
        libchess::Move best_move;
        std::uniform_real_distribution<> dist(0.0, 1.0);
        for (const auto& move : search_node->move_list) {
            auto& child = search_node->children.find(move.value_sans_type());
            //if the node is unexplored
            if (child == search_node->children.end()) {
                max_uct = INFINITY;
                best_move = move;
                break;
            }
            float uct = child->second->UCT();
            //add some random noise to the search if it is exploratory.
            if (explore)
                uct += dist(mt_eng);
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

        //recurse down tree
        pos.make_move(best_move);
        float v = mcts_search(pos, next_node, explore);
        pos.unmake_move();

        //back propogate stats (and remove virtual loss)
        search_node->w += v;
        search_node->n += 1;
        return -v;
    }

    libchess::Move MCTSearch::choose_best_move(libchess::Position& pos, const bool& stopped_flag, int& out_score) {
        //validate tree cache
        if (search_root->position_hash != pos.hash()) {
            std::cout << "Cache miss.\n";
            search_root = std::make_unique<MCTS_Node>();
        }
        
        //perfrom iterations of MCTS
        //exploration passes (background thread(s)):
        std::vector<std::unique_ptr<MCTS_Node>> r_search_roots(config::THREAD_CNT - 1); 
        std::vector<std::thread> search_threads(config::THREAD_CNT - 1);
        for (int thread_id = 0; thread_id < config::THREAD_CNT - 1; thread_id++) {
            r_search_roots[thread_id] = std::move(std::make_unique<MCTS_Node>());
            auto& r_search_root = r_search_roots[thread_id];
            search_threads[thread_id] = std::move(std::thread([&]() {
                for (int iter = 0; iter < config::MCTS_ITERATIONS && !stopped_flag; iter++) {
                    mcts_search(libchess::Position{pos}, r_search_root, true);
                }
            }));
        }
        //greedy pass (main thread):
        for (int iter = 0; iter < config::MCTS_ITERATIONS && !stopped_flag; iter++) {
            mcts_search(libchess::Position{pos}, search_root, false);
        }
        //join background threads
        for (auto& thread : search_threads) {
            thread.join();
        }
        //choose move with highest visit count
        float max_n = -INFINITY;
        libchess::Move best_move;
        for (auto& move : pos.legal_move_list()) {
            //get greedy node stats
            auto& child_greedy = search_root->children[move.value_sans_type()];
            float greedy_n = child_greedy != nullptr ? child_greedy->n : 0;

            //get all explore node stats
            float local_max_n = greedy_n;
            for (auto& r_search_root : r_search_roots) {
               auto& child_explore = r_search_root->children[move.value_sans_type()];
               float explore_n = child_explore != nullptr ? child_explore->n : 0;
               local_max_n = std::max(local_max_n, explore_n);
            }

            if (local_max_n > max_n) {
                max_n = local_max_n;
                best_move = move;
            }
        }

        //calculate predicted score
        float score = -search_root->w / search_root->n;
        float max_ = 5000;
        float min_ = -5000;
        score = std::min(std::max((score+1)*(max_-min_)/2 + min_, min_), max_); //reverse normalize
        out_score = static_cast<int>(score);

        //move search tree down to chosen node
        shift_tree_down(best_move.value_sans_type());
        return best_move;
    }

    bool MCTSearch::shift_tree_down(libchess::Move::value_type move) {
        auto&& next_node = search_root->children.find(move);
        if (next_node != search_root->children.end()) {
            search_root = std::move(next_node->second);
            return true;
        }
        return false;
    }
}