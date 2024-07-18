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
    }
}