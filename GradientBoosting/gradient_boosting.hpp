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
        class BaseGBDT {
            public:
            size_t max_tree_depth = 3;
            size_t n_estimators = 100;
            T learning_rate = 0.1;
            bool show_logs = false;

            virtual void fit(const std::vector<std::vector<T>>& x_train, const std::vector<P>& y_train) = 0;
            virtual std::vector<T> predict(const std::vector<std::vector<T>>& x_test) = 0;
            virtual ~BaseGBDT() {};
        };

        template <typename T = double, typename P = double>
        class GBDTRegressor : public BaseGBDT<T, P, hina::decision_tree::DecisionTreeRegressor<T>> {
            public:
            std::vector<hina::decision_tree::DecisionTreeRegressor<T>> models;
            GBDTRegressor() {};
            GBDTRegressor(size_t n_estimators_, size_t max_tree_depth_, bool show_logs_) {this->n_estimators = n_estimators_; this->max_tree_depth = max_tree_depth_; this->show_logs = show_logs_;};

            void fit(const std::vector<std::vector<T>>& x_train, const std::vector<P>& y_train) override;    
            std::vector<T> predict(const std::vector<std::vector<T>>& x_test) override;
        };

        template <typename T = double, typename P = double>
        class GBDTClassifier : public BaseGBDT<T, P, hina::decision_tree::DecisionTreeRegressor<T>> {
            public:
            std::unordered_map<size_t, std::vector<hina::decision_tree::DecisionTreeRegressor<T>>> models; // {class_0 : models, class_1 : models, ..}
            size_t n_classes = 0;
            GBDTClassifier() {};
            GBDTClassifier(size_t n_estimators_, size_t max_tree_depth_, bool show_logs_) {this->n_estimators = n_estimators_; this->max_tree_depth = max_tree_depth_; this->show_logs = show_logs_;};

            void fit(const std::vector<std::vector<T>>& x_train, const std::vector<P>& y_train) override;    
            std::vector<T> predict(const std::vector<std::vector<T>>& x_test) override;
        };
    }
}