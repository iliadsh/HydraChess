#pragma once

#include "serialize.hpp"
#include <torch/torch.h>
#include <Position.h>

namespace hydra {
    /**
     * Custom dataset loader for chess positions and their evaluations with a third-party engine.
     * As the chess engine evaluations also contain lookahead search (commonly alpha-beta), 
     * hopefully the result is a learned evaluation that contains lookahead charactaristics. 
     */
    class PositionDataset : public torch::data::Dataset<PositionDataset>
    {
        using DataType = std::vector<std::tuple<std::string, float>>;
        private:
            /**
             * Stored CSV parsed data.
             */
            DataType csv_;

            /**
             * Reads CSV file containing move + evaluation data. 
             * @param {const std::string&} location - file path.
             * @returns {DataType} parsed data.
             */ 
            DataType ReadCSV(const std::string& location) {
                DataType csv;

                std::fstream in(location, std::ios::in);
                std::string line;
                std::string pos;
                std::string score;

                while (std::getline(in, line)) {
                    std::stringstream postream{line};
                    std::getline(postream, pos, ',');
                    std::getline(postream, score, ',');
                    csv.push_back(std::make_tuple(pos, std::stof(score)));
                }

                return csv;
            }

        public:
            explicit PositionDataset(const std::string& file_name_csv) : csv_(ReadCSV(file_name_csv)) {}

            /**
             * Get training example.
             * @param {size_t} index - example index.
             * @returns {torch::data::Example<>} training example.
             */ 
            torch::data::Example<> get(size_t index) override {
                libchess::Position pos{std::get<0>(csv_[index])};
                float score = std::get<1>(csv_[index]);
                //if (pos.side_to_move() == libchess::constants::BLACK) score *= -1; //flip score
                //float certainty = std::min(5, pos.fullmoves()) / 5.0; //reduce certainty in early game
                //score *= certainty;

                torch::Tensor board_tensor = serialize(pos);
                torch::Tensor score_tensor = torch::full({1}, score);
                return {board_tensor, score_tensor};
            }

            /**
             * Size of training set.
             * @returns {torch::optional<size_t>} size of training set.
             */ 
            torch::optional<size_t> size() const override {
                return csv_.size();
            }
    };   
}