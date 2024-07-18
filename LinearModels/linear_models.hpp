#pragma once

#include <vector>
#include <string>

#include "../LossGradients/lossgradients.hpp"
#include "../Metrics/metrics.hpp"
#include "../Optimizers/linear_stochastic_gradient_descent_optimizer.hpp"
#include "../LinearAlgebra/linearalgebra.hpp"
namespace hina {
    namespace linear_models {
        template <typename T>
        class BaseLinearModel {
            public:
            std::vector<T> coef_;
            T bias_;
            T learning_rate_ = 0.05;
            size_t epoches_ = 100;
            size_t batch_size_ = 10;
            std::string regularization_ = "None";
            T regularization_coef_ = 0.05;
            virtual void fit(const std::vector<std::vector<T>> &x_train, const std::vector<T> &y_train) = 0;
            virtual std::vector<T> predict(const std::vector<std::vector<T>> &x_test) = 0;
            virtual ~BaseLinearModel() {}
        };

        template <typename T>
        class SGDRegressor : public BaseLinearModel<T> {
            public:
            SGDRegressor() {};
            void fit(const std::vector<std::vector<T>> &x_train, const std::vector<T> &y_train) override;
            std::vector<T> predict(const std::vector<std::vector<T>> &x_test) override;
        };

        template <typename T>
        class BinaryLogisticRegression : public BaseLinearModel<T> {
            public:
            BinaryLogisticRegression() {};
            void fit(const std::vector<std::vector<T>> &x_train, const std::vector<T> &y_train) override;
            std::vector<T> predict_proba(const std::vector<std::vector<T>> &x_test);
            std::vector<T> predict(const std::vector<std::vector<T>> &x_test) override;
        };
    }
}

