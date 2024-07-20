#pragma once

#include <vector>
#include <unordered_map>
#include <thread>
#include <future>
#include <random>
#include <algorithm>


#include "../Tree/decision_tree.hpp"

namespace hina {
    namespace tree_ensemble {
        template <typename T = double, typename P = double, typename DecisionTreeType = hina::decision_tree::DecisionTreeRegressor<T>> 
        class BaseRandomForest {
            public:
            size_t max_tree_depth = 10;
            size_t n_estimators = 500;
            T percentage_part_features;

            std::vector<DecisionTreeType> models;
            std::vector<std::vector<size_t>> models_feature_indexes;
            std::pair<std::vector<std::vector<T>>, std::vector<P>> bootstrap(const std::vector<std::vector<T>> &x_train, const std::vector<P> &y_train);
            std::vector<size_t> get_feature_subprojection(size_t n_features, T percentage_part);
            void fit_tree(DecisionTreeType& tree, const std::vector<std::vector<T>>& x_train, const std::vector<P>& y_train, size_t tree_idx);
            void fit(const std::vector<std::vector<T>>& x_train, const std::vector<P>& y_train);
            virtual std::vector<T> predict(const std::vector<std::vector<T>>& x_test) = 0;
            virtual ~BaseRandomForest() {};
        };

        template <typename T = double, typename P = double>
        class RandomForestRegressor : public BaseRandomForest<T, P, hina::decision_tree::DecisionTreeRegressor<T>> {
            public:
            RandomForestRegressor() {this->percentage_part_features = 0.34;};
            RandomForestRegressor(size_t n_estimators_) {this->n_estimators = n_estimators_; this->percentage_part_features = 0.34;};
            RandomForestRegressor(size_t n_estimators_, size_t max_tree_depth_) {this->n_estimators = n_estimators_; this->max_tree_depth = max_tree_depth_; this->percentage_part_features = 0.34;};

            std::vector<T> predict(const std::vector<std::vector<T>>& x_test) override;
        };

        template <typename T = double, typename P = double>
        class RandomForestClassifier : public BaseRandomForest<T, P, hina::decision_tree::DecisionTreeClassifier<T, P>> {
            public:
            RandomForestClassifier() {this->percentage_part_features = 0.34;};
            RandomForestClassifier(size_t n_estimators_) {this->n_estimators = n_estimators_; this->percentage_part_features = 0.5;};
            RandomForestClassifier(size_t n_estimators_, size_t max_tree_depth_) {this->n_estimators = n_estimators_; this->max_tree_depth = max_tree_depth_; this->percentage_part_features = 0.34;};

            std::vector<T> predict(const std::vector<std::vector<T>>& x_test) override;
        };
    }
}