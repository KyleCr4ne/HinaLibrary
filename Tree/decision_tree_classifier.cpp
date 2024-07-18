#include "./decision_tree.hpp"

namespace hina {
    namespace decision_tree {
        template <typename T, typename P>
        T DecisionTreeClassifier<T, P>::gini(const std::vector<std::pair<std::vector<T>, P>> &cur_split) {
            T gini_score = 1.0;
            std::unordered_map<P, T> class_probas;

            for (size_t i = 0; i < cur_split.size(); ++i) {
                class_probas[cur_split[i].second] += 1.0;
            }

            for (const auto & class_object : class_probas) {
                T proba = class_object.second / cur_split.size();
                gini_score -= std::pow(proba, 2);
            }

            return gini_score;
        }

        template <typename T, typename P>
        void DecisionTreeClassifier<T, P>::fit(const std::vector<std::vector<T>> &x_train, const std::vector<P> &y_train) {
            this->num_of_features = x_train[0].size();

            for (size_t i = 0; i < x_train.size(); ++i) {
                this->data.push_back(std::make_pair(x_train[i], y_train[i]));
            }

            this->tree_root = this->tree_builder(1, this->data);
        }

        template <typename T, typename P>
        Node<T, P>* DecisionTreeClassifier<T, P>::tree_builder(size_t cur_depth, const std::vector<std::pair<std::vector<T>, P>> &cur_data) {
            Node<T, P>* cur_node = new Node<T, P>();
            std::unordered_map<P, size_t> class_counter;

            for (size_t i = 0; i < cur_data.size(); ++i) {
                class_counter[cur_data[i].second]++;
            } 

            P answer; 
            size_t mx_count = 0;

            for (const auto & class_object : class_counter) {
                P class_label = class_object.first;
                size_t count = class_object.second;
                if (count > mx_count) {
                    mx_count = count;
                    answer = class_label; 
                }
            }

            if (cur_depth == this->max_tree_depth || class_counter.size() == 1) {
                cur_node->is_leaf = true;
                cur_node->label = answer;
                cur_node->label_proba = ((T) class_counter[answer]) / ((T) cur_data.size());
                return cur_node;
            }

            T base_information = this->gini(cur_data);
            T best_information_gain = 0.0;
            std::vector<std::pair<std::vector<T>, P>> best_left_split;
            std::vector<std::pair<std::vector<T>, P>> best_right_split;

            size_t best_feature_idx;
            T best_treshold;

            for (size_t feature_idx = 0; feature_idx < this->num_of_features; ++feature_idx) {
                std::vector<std::pair<std::vector<T>, P>> sorted_data(cur_data);
                std::sort(sorted_data.begin(), sorted_data.end(), [&](const std::pair<std::vector<T>, P> &sample_1, std::pair<std::vector<T>, P> &sample_2) {
                    return sample_1.first[feature_idx] < sample_2.first[feature_idx];
                });

                std::vector<std::pair<std::vector<T>, P>> cur_left_data(sorted_data);
                std::vector<std::pair<std::vector<T>, P>> cur_right_data;

                while ((cur_left_data.size())) {
                    cur_right_data.push_back(cur_left_data.back());
                    cur_left_data.pop_back();

                    T left_gini = this->gini(cur_left_data);
                    T right_gini = this->gini(cur_right_data);

                    T treshold = (cur_left_data.back().first[feature_idx] + cur_right_data.back().first[feature_idx]) / 2.0;

                    if (base_information - cur_left_data.size() * left_gini / cur_data.size() - cur_right_data.size() * right_gini / cur_right_data.size() > best_information_gain) {
                        best_information_gain = base_information - cur_left_data.size() * left_gini / cur_data.size() - cur_right_data.size() * right_gini / cur_right_data.size();
                        best_left_split = cur_left_data;
                        best_right_split = cur_right_data;
                        best_feature_idx = feature_idx;
                        best_treshold = treshold;
                    }
                }
            }
            cur_node->is_leaf = false;
            cur_node->split_feature_idx = best_feature_idx;
            cur_node->treshold = best_treshold;
            cur_node->left_child = this->tree_builder(cur_depth + 1, best_left_split);
            cur_node->right_child = this->tree_builder(cur_depth + 1, best_right_split);

            return cur_node;
        }

        template <typename T, typename P>
        std::vector<P> DecisionTreeClassifier<T, P>::predict(const std::vector<std::vector<T>> &x_test) {
            std::vector<P> answer(x_test.size());

            for (size_t i = 0; i < x_test.size(); ++i) {
                Node<T, P>* copy = this->tree_root;
                while (!copy->is_leaf) {
                    if (x_test[i][copy->split_feature_idx] > copy->treshold) copy = copy->right_child;
                    else copy = copy->left_child;
                }
                answer[i] = copy->label;
            }

            return answer;
        }

        template <typename T, typename P>
        std::vector<T> DecisionTreeClassifier<T, P>::predict_proba(const std::vector<std::vector<T>> &x_test) {
            std::vector<T> answer(x_test.size());

            for (size_t i = 0; i < x_test.size(); ++i) {
                Node<T, P>* copy = this->tree_root;
                while (!copy->is_leaf) {
                    if (x_test[i][copy->split_feature_idx] > copy->treshold) copy = copy->right_child;
                    else copy = copy->left_child;
                }
                answer[i] = copy->label_proba;
            }

            return answer;
        }
    }
}