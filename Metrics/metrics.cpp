#include "./metrics.hpp"


namespace hina {
    namespace metrics {
        template <typename T>
        T mean_squared_error(const std::vector<T> &target_values, const std::vector<T> &received_values) {
            T sum_squared_diff = 0.0;
            for (size_t i = 0; i < target_values.size(); ++i) {
                sum_squared_diff += std::pow(received_values[i] - target_values[i], 2);
            }
            return sum_squared_diff / target_values.size();
        }

        template <typename T>
        T mean_absolute_error(const std::vector<T> &target_values, const std::vector<T> &received_values) {
            T sum_absolute_diff = 0.0;
            for (size_t i = 0; i < target_values.size(); ++i) {
                sum_absolute_diff += std::abs(received_values[i] - target_values[i]);
            }
            return sum_absolute_diff / target_values.size();
        }

        template <typename T>
        T sigmoid (const T x) {
            return 1.0 / (1 + std::exp(-x));
        }

        template <typename T>
        T BCE_loss(const std::vector<T> &target_values, const std::vector<T> &received_values) {
            const T EPS = 0.0001;
            T sum_logg_diff = 0.0;
            for (size_t i = 0; i < target_values.size(); ++i) {
                sum_logg_diff += target_values[i] * std::log(received_values[i] + EPS) + (1 - target_values[i]) * std::log(1 - received_values[i] + EPS);
            }
            return - sum_logg_diff / target_values.size();
        }

        template <typename T>
        T accuracy_score(const std::vector<T> &target_values, const std::vector<T> &received_values) {
            T counter = 0.0;
            for (size_t i = 0; i < target_values.size(); ++i) {
                counter += (target_values[i] == received_values[i] ? 1.0 : 0.0);
            }

            return counter / target_values.size();
        }

        template <typename T>
        std::vector<std::vector<T>> softmax(const std::vector<std::vector<T>> &X) {
            std::vector<std::vector<T>> softmax_result(X.size());
            for (size_t i = 0; i < X.size(); ++i) {
                T sm_exp = 0.0;
                for (size_t j = 0; j < X[i].size(); ++j) {
                    sm_exp += std::exp(X[i][j]);
                }
                for (size_t j = 0; j < X[i].size(); ++j) {
                    softmax_result[i].push_back(std::exp(X[i][j]) / sm_exp);
                }
            }
            return softmax_result;
        }

        template <typename T>
        T cross_entopy_loss(const std::vector<std::vector<T>> &target_probability_distribution, const std::vector<std::vector<T>> & received_probability_distribution) {
            const T EPS = 0.0001;
            T cross_entopy = 0.0;
            for (size_t i = 0; i < target_probability_distribution.size(); ++i) {
                T cross_entopy_by_sample = 0.0;
                for (size_t j = 0; j < target_probability_distribution[i].size(); ++j) {
                    cross_entopy_by_sample -= (target_probability_distribution[i][j] == 0.0 ? 0.0 : std::log(received_probability_distribution[i][j] + EPS));
                }
                cross_entopy += cross_entopy_by_sample / target_probability_distribution.size();
            }
            return cross_entopy;
        }
    }
}