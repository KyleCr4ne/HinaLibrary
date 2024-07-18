#pragma once

#include <vector>
#include <random>
#include <algorithm>
#include "../LinearAlgebra/linearalgebra.hpp"
#include "../LossGradients/lossgradients.hpp"

namespace hina {
    namespace linear_sgd_optimizer {
        template <typename T>
        class StochasticGradientDescentParams {
            public:
            std::vector<std::vector<T>> X;
            std::vector<T> Y;
            std::function<T(const std::vector<T>&, const std::vector<T>&)> loss;
            std::function<std::pair<std::vector<T>, T>(const std::vector<T> &, const T,const std::vector<T> &, const T)> gradient;
            size_t batch_size = 32;
            size_t epoches = 100;
            T learning_rate = 0.05;
            std::string regularization = "None";
            T regularization_coef = 0.05;
            T end_loss_value = 1e-6;
        };

        template <typename T>
        class StochasticGradientDescent {
            public:
                StochasticGradientDescent() {};
                std::pair<std::vector<T>, T> optimize(StochasticGradientDescentParams<T>* params);
                ~StochasticGradientDescent() {};

        };
    }
}