#pragma once

#include <torch/torch.h>

namespace hydra {
    /**
     * A convolutional block. Does a conv, batchnorm, and ReLU.
     */ 
    struct ConvBlockImpl : public torch::nn::Module {
        ConvBlockImpl(int64_t in_channels, int64_t out_channels, int64_t kernel_size)
            : conv(torch::nn::Conv2dOptions(in_channels, out_channels, kernel_size).padding(kernel_size / 2).bias(false)),
              batch_norm(out_channels)
        {
            register_module("conv", conv);
            register_module("batch_norm", batch_norm);
        }

        torch::Tensor forward(torch::Tensor x) {
            x = torch::relu(batch_norm(conv(x)));
            return x;
        }

        torch::nn::Conv2d conv;
        torch::nn::BatchNorm2d batch_norm;
    };
    TORCH_MODULE(ConvBlock);

    /**
     * A residual block. Essentially two conv blocks with a "residual" add before the last ReLU activation.
     */ 
    struct ResBlockImpl : public torch::nn::Module {
        ResBlockImpl(int64_t in_channels, int64_t out_channels, int64_t kernel_size)
            : conv1(torch::nn::Conv2dOptions(in_channels, out_channels, kernel_size).padding(kernel_size / 2).bias(false)),
              batch_norm1(out_channels),
              conv2(torch::nn::Conv2dOptions(out_channels, out_channels, kernel_size).padding(kernel_size / 2).bias(false)),
              batch_norm2(out_channels)
        {
            register_module("conv1", conv1);
            register_module("batch_norm1", batch_norm1);
            register_module("conv2", conv2);
            register_module("batch_norm2", batch_norm2);
        }

        torch::Tensor forward(torch::Tensor x) {
            auto input = x.clone();
            x = torch::relu(batch_norm1(conv1(x)));
            x = batch_norm2(conv2(x));
            x = torch::relu(x + input);
            return x;
        }

        torch::nn::Conv2d conv1, conv2;
        torch::nn::BatchNorm2d batch_norm1, batch_norm2;
    };
    TORCH_MODULE(ResBlock);

    /**
     * The actual value network. Takes in a 18x8x8 board state as input
     * and returns a single scalar from [-1, 1] as the predicted score.
     */ 
    struct ValueNetworkImpl : public torch::nn::Module {
        /**
         * Initialize network.
         */
        ValueNetworkImpl() 
            : conv(18, 256, 5),
              res1(256, 256, 3),
              res2(256, 256, 3),
              res3(256, 256, 3),
              res4(256, 256, 3),
              res5(256, 256, 3),
              res6(256, 256, 3),
              res7(256, 256, 3),
              last_conv(256, 4, 1),
              n(get_conv_output()),
              fc1(n, 256),
              fc2(256, 1),
              drop1(torch::nn::Dropout2dOptions().p(0.3)),
              drop2(torch::nn::Dropout2dOptions().p(0.3))
        {
            register_module("conv", conv);
            register_module("res1", res1);
            register_module("res2", res2);
            register_module("res3", res3);
            register_module("res4", res4);
            register_module("res5", res5);
            register_module("res6", res6);
            register_module("res7", res7);
            register_module("last_conv", last_conv);
            register_module("fc1", fc1);
            register_module("fc2", fc2);
            register_module("drop1", drop1);
            register_module("drop2", drop2);
        }

        /**
         * Forward implementation of network.
         * @param {torch::Tensor} x - input tensor (18x8x8).
         * @returns {torch::Tensor} output tensor [-1, 1].
         */ 
        torch::Tensor forward(torch::Tensor x) {
            //perform convolutions/residuals
            x = residual_tower(x);
            //flatten
            x = drop1(x.view({-1, n}));
            //fully connected layers
            x = drop2(torch::relu(fc1(x))); //?   -> 256 (Dense)
            x = torch::tanh(fc2(x));        //256 -> 1   (Dense)
            return x;
        }

        /**
         * Large residual tower.
         */ 
        torch::Tensor residual_tower(torch::Tensor x) {
            x = conv(x);      // 18 -> 256 (5x5)
            x = res1(x);      //256 -> 256 (3x3)
            x = res2(x);      //256 -> 256 (3x3)
            x = res3(x);      //256 -> 256 (3x3)
            x = res4(x);      //256 -> 256 (3x3)
            x = res5(x);      //256 -> 256 (3x3)
            x = res6(x);      //256 -> 256 (3x3)
            x = res7(x);      //256 -> 256 (3x3)
            x = last_conv(x); //256 -> 4   (1x1)
            return x;
        }

        /**
         * Get final convolution size for dense layers.
         * @returns {int64_t} element length.
         */ 
        int64_t get_conv_output() {
            torch::Tensor x = torch::zeros({1, 18, 8, 8});
            x = residual_tower(x);
            return x.numel();
        }

        ConvBlock conv, last_conv;
        ResBlock res1, res2, res3, res4, res5, res6, res7;
        int64_t n;
        torch::nn::Linear fc1, fc2;
        torch::nn::Dropout drop1, drop2;
    };
    TORCH_MODULE(ValueNetwork);
}