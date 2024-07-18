#pragma once

#include <vector>
#include <unordered_map>
#include <algorithm>


namespace hina {
    namespace decision_tree {
        template <typename T, typename P>
        class Node {
            public:
            size_t split_feature_idx = 0;
            T treshold = 0.0;
            bool is_leaf = false;
            P label; // Double if its regression tree, int else.
            T label_proba;
            Node* left_child = nullptr;
            Node* right_child = nullptr;
        };

        template <typename T, typename P>
        class BaseDecisionTree {
            public:
            size_t max_tree_depth = 3;
            Node<T, P>* tree_root;
            std::vector<std::pair<std::vector<T>, P>> data;
            size_t num_of_features = 0;

            virtual void fit(const std::vector<std::vector<T>> &x_train, const std::vector<P> &y_train) = 0;
            virtual std::vector<P> predict(const std::vector<std::vector<T>> &x_test) = 0;
            virtual Node<T, P>* tree_builder(size_t cur_depth, const std::vector<std::pair<std::vector<T>, P>> &cur_data) = 0;
            virtual ~BaseDecisionTree() {};
        };

        template <typename T, typename P = int>
        class DecisionTreeClassifier : public BaseDecisionTree<T, P> {
            public:
            DecisionTreeClassifier() {};
            DecisionTreeClassifier(size_t max_depth) {this->max_tree_depth = max_depth;};
            void fit(const std::vector<std::vector<T>> &x_train, const std::vector<P> &y_train) override;
            std::vector<P> predict(const std::vector<std::vector<T>> &x_test) override;
            Node<T, P>* tree_builder(size_t cur_depth, const std::vector<std::pair<std::vector<T>, P>> &cur_data) override;
            std::vector<T> predict_proba(const std::vector<std::vector<T>> &x_test);
            T gini(const std::vector<std::pair<std::vector<T>, P>> &cur_split);
        };
    }
}