#include "./random_forest.hpp"

namespace hina {
    namespace tree_ensemble {
        template<typename T, typename P, typename DecisionTreeType>
        std::pair<std::vector<std::vector<T>>, std::vector<P>> BaseRandomForest<T, P, DecisionTreeType>::bootstrap(const std::vector<std::vector<T>> &x_train, const std::vector<P> &y_train) {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<> dist(0, x_train.size() - 1);

            std::vector<std::vector<T>> features;
            std::vector<P> targets;

            for (size_t i = 0; i < x_train.size(); ++i) {
                int random_idx = dist(gen);
                features.push_back(x_train[random_idx]);
                targets.push_back(y_train[random_idx]);
            }
            return std::make_pair(features, targets);
        }

        template<typename T, typename P, typename DecisionTreeType>
        std::vector<size_t> BaseRandomForest<T, P, DecisionTreeType>::get_feature_subprojection(size_t n_features, T percentage_part) {
            size_t subprojection_n_features = size_t(n_features * percentage_part);
            std::vector<size_t> features_idx(n_features);
            std::iota(features_idx.begin(), features_idx.end(), 0);
            std::random_device rd;
            std::mt19937 gen(rd());
            std::shuffle(features_idx.begin(), features_idx.end(), gen);
            std::vector<size_t> projection_features;
            for (size_t i = 0; i < subprojection_n_features; ++i) {
                projection_features.push_back(features_idx[i]);
            }
            std::sort(projection_features.begin(), projection_features.end());

            return projection_features;
        }

        template<typename T, typename P, typename DecisionTreeType>
        void BaseRandomForest<T, P, DecisionTreeType>::fit_tree(DecisionTreeType& tree, const std::vector<std::vector<T>>& x_train, const std::vector<P>& y_train, size_t tree_idx) {
            std::vector<std::vector<T>> sub_x_train(x_train.size());
            for (size_t i = 0; i < x_train.size(); ++i) {
                for (size_t j = 0; j < this->models_feature_indexes[tree_idx].size(); ++j) {
                    sub_x_train[i].push_back(x_train[i][this->models_feature_indexes[tree_idx][j]]);
                }
            }
            std::pair<std::vector<std::vector<T>>, std::vector<T>> bs = this->bootstrap(x_train, y_train);
            tree.fit(bs.first, bs.second);
        }    

        template<typename T, typename P, typename DecisionTreeType>
        void BaseRandomForest<T, P, DecisionTreeType>::fit(const std::vector<std::vector<T>>& x_train, const std::vector<P>& y_train) {
            
            for (size_t i = 0; i < this->n_estimators; ++i) {
                DecisionTreeType clf(this->max_tree_depth);
                std::vector<size_t> feature_subprojection_indexes = this->get_feature_subprojection(x_train[0].size(), this->percentage_part_features);
                this->models.push_back(clf);
                this->models_feature_indexes.push_back(feature_subprojection_indexes);
            }

           
            std::vector<std::future<void>> futures;
            for (size_t i = 0; i < this->n_estimators; ++i) {
                size_t index = i;
                futures.push_back(std::async(std::launch::async, std::bind(&BaseRandomForest::fit_tree, this, std::ref(models[i]), std::ref(x_train), std::ref(y_train), std::ref(index))));
            }

            for (auto &thread : futures) {
                thread.get();
            }
        }
    }
}