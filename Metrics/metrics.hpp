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

        template <typename T>
        std::vector<std::vector<T>> softmax(const std::vector<std::vector<T>> &X);

        template <typename T>
        T cross_entopy_loss(const std::vector<std::vector<T>> &target_probability_distribution, const std::vector<std::vector<T>> & received_probability_distribution);
    }
}