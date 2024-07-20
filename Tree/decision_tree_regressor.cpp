#include "./decision_tree.hpp"

namespace hina {
    namespace decision_tree {
        template <typename T>
        T DecisionTreeRegressor<T>::get_average(const std::vector<std::pair<std::vector<T>, T>> &cur_split) {
            T average_value = 0.0;
            for (size_t i = 0; i < cur_split.size(); ++i) {
                average_value += cur_split[i].second / cur_split.size();
            }

            return average_value;
        }

        template <typename T>
        T DecisionTreeRegressor<T>::dispersion(const std::vector<std::pair<std::vector<T>, T>> &cur_split) {
            T average_value = this->get_average(cur_split);
            T dispersion_value = 0.0;

            for (size_t i = 0; i < cur_split.size(); ++i) {
                dispersion_value += std::pow((cur_split[i].second - average_value), 2) / cur_split.size();
            }

            return dispersion_value;
        }

        template <typename T>
        void DecisionTreeRegressor<T>::fit(const std::vector<std::vector<T>> &x_train, const std::vector<T> &y_train) {
            this->num_of_features = x_train[0].size();

            for (size_t i = 0; i < x_train.size(); ++i) {
                this->data.push_back(std::make_pair(x_train[i], y_train[i]));
            }

            this->tree_root = this->tree_builder(1, this->data);
            std::vector<std::pair<std::vector<T>, T>>().swap(this->data);
        }

        template <typename T>
        Node<T, T>* DecisionTreeRegressor<T>::tree_builder(size_t cur_depth, const std::vector<std::pair<std::vector<T>, T>> &cur_data) {
            Node<T, T>* cur_node = new Node<T, T>();
            T base_information = this->dispersion(cur_data);
            if (cur_depth == this->max_tree_depth || base_information < this->min_dispersion_to_split) {
                cur_node->is_leaf = true;
                cur_node->label = this->get_average(cur_data);
                return cur_node;
            }
            T best_information_gain = 0.0;
            std::vector<std::pair<std::vector<T>, T>> best_left_split;
            std::vector<std::pair<std::vector<T>, T>> best_right_split;

            size_t best_feature_idx;
            T best_treshold;
            
            bool is_splitted_flag = false;
            for (size_t feature_idx = 0; feature_idx < this->num_of_features; ++feature_idx) {
                std::vector<std::pair<std::vector<T>, T>> sorted_data(cur_data);
                std::sort(sorted_data.begin(), sorted_data.end(), [&](const std::pair<std::vector<T>, T> &sample_1, std::pair<std::vector<T>, T> &sample_2) {
                    return sample_1.first[feature_idx] < sample_2.first[feature_idx];
                });
                std::vector<std::pair<std::vector<T>, T>> cur_left_data(sorted_data);
                std::vector<std::pair<std::vector<T>, T>> cur_right_data;

                while ((cur_left_data.size())) {
                    cur_right_data.push_back(cur_left_data.back());
                    cur_left_data.pop_back();

                    T left_dispersion = this->dispersion(cur_left_data);
                    T right_dispersion = this->dispersion(cur_right_data);

                    T treshold = (cur_left_data.back().first[feature_idx] + cur_right_data.back().first[feature_idx]) / 2.0;
                    
                    if (base_information - cur_left_data.size() * left_dispersion / cur_data.size() - cur_right_data.size() * right_dispersion / cur_right_data.size() > best_information_gain) {
                        is_splitted_flag = true;

                        best_information_gain = base_information - cur_left_data.size() * left_dispersion / cur_data.size() - cur_right_data.size() * right_dispersion / cur_right_data.size();
                        best_left_split = cur_left_data;
                        best_right_split = cur_right_data;
                        best_feature_idx = feature_idx;
                        best_treshold = treshold;
                    }
                }
            }
            if (!is_splitted_flag) {
                cur_node->is_leaf = true;
                cur_node->label = this->get_average(cur_data);
                return cur_node;
            }
            cur_node->is_leaf = false;
            cur_node->split_feature_idx = best_feature_idx;
            cur_node->treshold = best_treshold;
            cur_node->left_child = this->tree_builder(cur_depth + 1, best_left_split);
            cur_node->right_child = this->tree_builder(cur_depth + 1, best_right_split);

            return cur_node;
        }

        template <typename T> 
        std::vector<T> DecisionTreeRegressor<T>::predict(const std::vector<std::vector<T>> &x_test) {
            std::vector<T> answer(x_test.size());

            for (size_t i = 0; i < x_test.size(); ++i) {
                Node<T, T>* copy = this->tree_root;
                while (!copy->is_leaf) {
                    if (x_test[i][copy->split_feature_idx] > copy->treshold) copy = copy->right_child;
                    else copy = copy->left_child;
                }
                answer[i] = copy->label;
            }
            
            return answer;
        }
    }
}