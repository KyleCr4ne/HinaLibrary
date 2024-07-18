#include "./linear_stochastic_gradient_descent_optimizer.hpp"


namespace hina {
    namespace linear_sgd_optimizer {
        template <typename T>
        std::pair<std::vector<T>, T> StochasticGradientDescent<T>::optimize(StochasticGradientDescentParams<T>* params) {
            std::vector<T> coefs(params->X[0].size(), 0.0);
            T bias = 0.0;

            //T initial_loss = params->loss(linearalgebra::multiplyMatrixByWeights(params->X, coefs), params->Y);

            //T lambda = 2.0 / (params->epoches + 1);

            std::random_device rd;
            std::mt19937 gen(rd());
            
            std::vector<std::pair<std::vector<T>, T>> DATA;

            for (size_t i = 0; i < params->X.size(); ++i) {
                DATA.push_back({params->X[i], params->Y[i]});
            }
            for (size_t epoch = 0; epoch < params->epoches; ++epoch) {
                std::shuffle(DATA.begin(), DATA.end(), gen);
                for (size_t batch_end_idx = params->batch_size; batch_end_idx < DATA.size(); batch_end_idx += params->batch_size) {
                    std::vector<T> gradient_weights_by_batch(params->X[0].size(), 0.0);
                    T gradient_bias_by_batch = 0.0;
                    for (size_t i = batch_end_idx - params->batch_size; i < batch_end_idx; ++i) {
                        std::vector<T> gradient_weights_by_sample(params->X[0].size(), 0.0);
                        T gradient_bias_by_sample = 0.0;
                        auto optim_step = params->gradient(DATA[i].first, DATA[i].second, coefs, bias);
                        gradient_weights_by_sample = optim_step.first;
                        gradient_bias_by_sample = optim_step.second;
                        for (size_t j = 0; j < params->X[0].size(); ++j) {
                            gradient_weights_by_batch[j] += gradient_weights_by_sample[j] / params->batch_size;
                        }
                        gradient_bias_by_batch += gradient_bias_by_sample / params->batch_size;
                    }
                    for (size_t j = 0; j < params->X[0].size(); ++j) {
                        coefs[j] -= params->learning_rate * gradient_weights_by_batch[j];
                        if (params->regularization == "L2") {
                            coefs[j] -= params->learning_rate * params->regularization_coef * coefs[j];
                        } else if (params->regularization == "L1") {
                            coefs[j] -= params->learning_rate * params->regularization_coef * lossgradients::sign(coefs[j]);
                        }
                    }
                    bias -= params->learning_rate * gradient_bias_by_batch;

                    
                }
            }
            return std::pair(coefs, bias);
        }
    }
}