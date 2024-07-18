#include "./linear_models.hpp"

namespace hina {
    namespace linear_models {
        template <typename T>
        void SGDRegressor<T>::fit(const std::vector<std::vector<T>> &x_train, const std::vector<T> &y_train) {
            hina::linear_sgd_optimizer::StochasticGradientDescentParams<T> params;
            params.batch_size = this->batch_size_;
            params.X = x_train;
            params.Y = y_train;
            params.loss = hina::metrics::mean_squared_error<T>;
            params.gradient = hina::lossgradients::mean_squared_loss_gradient<T>;

            hina::linear_sgd_optimizer::StochasticGradientDescent<T> SGD;

            std::pair<std::vector<T>, T> result = SGD.optimize(&params);

            this->coef_ = result.first;
            this->bias_ = result.second;
        }

        template <typename T>
        std::vector<T> SGDRegressor<T>::predict(const std::vector<std::vector<T>> &x_test) {
            std::vector<T> output = hina::linearalgebra::multiplyMatrixByWeights(x_test, this->coef_);
            for (size_t i = 0; i < x_test.size(); ++i) {
                output[i] += this->bias_;
            }
            return output;
        }

        template <typename T>
        void BinaryLogisticRegression<T>::fit(const std::vector<std::vector<T>> &x_train, const std::vector<T> &y_train) {
            hina::linear_sgd_optimizer::StochasticGradientDescentParams<T> params;
            params.batch_size = this->batch_size_;
            params.X = x_train;
            params.Y = y_train;
            params.loss = hina::metrics::BCE_loss<T>;
            params.gradient = hina::lossgradients::BCE_loss_gradient<T>;

            hina::linear_sgd_optimizer::StochasticGradientDescent<T> SGD;

            std::pair<std::vector<T>, T> result = SGD.optimize(&params);

            this->coef_ = result.first;
            this->bias_ = result.second;
        }

        template <typename T>
        std::vector<T> BinaryLogisticRegression<T>::predict_proba(const std::vector<std::vector<T>> &x_test) {
            std::vector<T> output = hina::linearalgebra::multiplyMatrixByWeights(x_test, this->coef_);
            for (size_t i = 0; i < x_test.size(); ++i) {
                output[i] += this->bias_;
                output[i] = hina::metrics::sigmoid(output[i]);
            }
            return output;
        }

        template <typename T>
        std::vector<T> BinaryLogisticRegression<T>::predict(const std::vector<std::vector<T>> &x_test) {
            std::vector<T> output = hina::linearalgebra::multiplyMatrixByWeights(x_test, this->coef_);
            for (size_t i = 0; i < x_test.size(); ++i) {
                output[i] += this->bias_;
                output[i] = hina::metrics::sigmoid(output[i]);
                output[i] = output[i] >= 0.5 ? 1.0 : 0.0;
            }
            return output;
        }
    }
}

