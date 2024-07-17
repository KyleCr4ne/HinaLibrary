#pragma once

#include <vector>
#include <cmath>

#include "../LinearAlgebra/linearalgebra.hpp"
#include "../Metrics/metrics.hpp"


namespace hina {
    namespace lossgradients {
        template <typename T>
        T sign(const T x);

        template <typename T>
        std::pair<std::vector<T>, T> mean_squared_loss_gradient(const std::vector<T> &sample, const T y,
        const std::vector<T> &weights, const T bias);

        template <typename T>
        std::pair<std::vector<T>, T> mean_absolute_loss_gradient(const std::vector<T> &sample, const T y,
        const std::vector<T> &weights, const T bias);

        template <typename T>
        std::pair<std::vector<T>, T> BCE_loss_gradient(const std::vector<T> &sample, const T y,
        const std::vector<T> &weights, const T bias);
    }
}