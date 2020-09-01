#include "train.hpp"

namespace hydra{
    void train(Eval net, const std::string& path) {
        std::cout << "Training Evaluator..." << std::endl;
        //load from checkpoint
        if(config::LOAD_CHECKPOINT) {
            torch::load(net, config::WEIGHTS_PATH);
        }
        //choose training device (GPU if supported, otherwise CPU)
        torch::DeviceType device_type;
        if(torch::cuda::is_available()) {
            std::cout << "Training on CUDA.\n";
            device_type = torch::kCUDA;
        }
        else {
            std::cout << "Training on CPU.\n";
            device_type = torch::kCPU;
        }
        torch::Device device(device_type);
        //move model to device
        net->to(device);
        
        //load dataset
        std::cout << "Loading train dataset...\n";
        auto data_set = PositionDataset(path).map(torch::data::transforms::Stack<>());
        int dataset_size = data_set.size().value();

        //setup dataloader
        auto data_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
            std::move(data_set),
            torch::data::DataLoaderOptions()
                .batch_size(config::BATCH_SIZE)
                .workers(4));
        std::cout << "Train dataset ready.\n";

        //create gradient optimizer
        torch::optim::Adam optimizer(net->parameters(), torch::optim::AdamOptions(1e-3));

        //best loss during training
        float best_mse = std::numeric_limits<float>::max();
        
        std::printf("Training for %ld epochs with a dataset size of %ld and batch size of %ld...\n", config::NUM_EPOCH, dataset_size, config::BATCH_SIZE);
        //train epochs
        for (int epoch = 1; epoch <= config::NUM_EPOCH; epoch++) {
            net->train();

            size_t batch_idx = 0;
            float mse = 0;
            int count = 0;
            int total_batches = 0;

            //minibatching
            for (auto& batch : *data_loader) {
                auto pos = batch.data.to(device), score = batch.target.to(device);
                
                //calculate loss
                optimizer.zero_grad();
                auto output = net->forward(pos);
                auto loss = torch::mse_loss(output, score);
                
                //do gradient step
                loss.backward();
                optimizer.step();

                mse += loss.template item<float>();

                batch_idx++;
                total_batches += batch.data.size(0);
                if (batch_idx % config::LOG_INTERVAL == 0) {
                    std::printf(
                        "\rTrain Epoch: %d/%ld [%5d/%5d] Loss: %.4f",
                        epoch,
                        config::NUM_EPOCH,
                        total_batches,
                        dataset_size,
                        loss.template item<float>()
                    );
                }

                count++;
            }

            mse /= (float)count;
            printf(" Mean Loss: %f\n", mse);

            //save best model
            if (mse < best_mse) {
                //ensure model is on CPU before saving
                net->to(torch::kCPU);
                torch::save(net, std::string(config::WEIGHTS_PATH) + "evaluator.pt");
                //return back to original device
                net->to(device);
                best_mse = mse;
            }
        }

        std::cout << "Training completed.\n";
    }
}