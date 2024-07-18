#pragma once

#include <vector>
#include <cmath>


namespace hina {
    namespace metrics {
        template <typename T>
        T mean_squared_error(const std::vector<T> &target_values, const std::vector<T> &received_values);

        template <typename T>
        T mean_absolute_error(const std::vector<T> &target_values, const std::vector<T> &received_values);

        template <typename T>
        T sigmoid (const T x);

        template <typename T>
        T BCE_loss(const std::vector<T> &target_values, const std::vector<T> &received_values);

        template <typename T>
        T accuracy_score(const std::vector<T> &target_values, const std::vector<T> &received_values);
    }
}