#include "./lossgradients.hpp"


namespace hina {
    namespace lossgradients {
        template <typename T>
        T sign(const T x) {
            if (x > 0) return 1.0;
            if (x < 0) return -1.0;
            return 0.0;
        }

        template <typename T>
        std::pair<std::vector<T>, T> mean_squared_loss_gradient(const std::vector<T> &sample, const T y,
        const std::vector<T> &weights, const T bias) {
            std::vector<T> gradient_weights(sample.size());
            T gradient_bias;

            T weights_by_features_product = linearalgebra::vector_scalar_product(sample, weights);

            for (size_t i = 0; i < sample.size(); ++i) {
                gradient_weights[i] = 2.0 * (weights_by_features_product + bias - y) * sample[i];
            }

            gradient_bias = 2.0 * (weights_by_features_product + bias - y);
            
            return std::pair(gradient_weights, gradient_bias);
        }

        template <typename T>
        std::pair<std::vector<T>, T> mean_absolute_loss_gradient(const std::vector<T> &sample, const T y,
        const std::vector<T> &weights, const T bias) {
            std::vector<T> gradient_weights(sample.size());
            T gradient_bias;

            T weights_by_features_product = linearalgebra::vector_scalar_product(sample, weights);

            for (size_t i = 0; i < sample.size(); ++i) {
                gradient_weights[i] = sign(weights_by_features_product + bias - y) * sample[i];
            }

            gradient_bias = sign(weights_by_features_product + bias - y);

            return std::pair(gradient_weights, gradient_bias);
        }

        template <typename T>
        std::pair<std::vector<T>, T> BCE_loss_gradient(const std::vector<T> &sample, const T y,
        const std::vector<T> &weights, const T bias) {
            std::vector<T> gradient_weights(sample.size());
            T gradient_bias;

            T weights_by_features_product = linearalgebra::vector_scalar_product(sample, weights);

            for (size_t i = 0; i < sample.size(); ++i) {
                gradient_weights[i] = (metrics::sigmoid(weights_by_features_product + bias) - y) * sample[i];
            }

            gradient_bias = metrics::sigmoid(weights_by_features_product + bias) - y;
            
            return std::pair(gradient_weights, gradient_bias);
        }
    }
}