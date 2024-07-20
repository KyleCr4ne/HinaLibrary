#include "./random_forest.hpp"

namespace hina {
    namespace tree_ensemble {
        template<typename T, typename P>
        std::vector<T> RandomForestClassifier<T, P>::predict(const std::vector<std::vector<T>>& x_test) {
            std::vector<T> y_pred_new(x_test.size());
            std::vector<std::unordered_map<T, size_t>> mp(x_test.size());

            for (size_t i = 0; i < this->n_estimators; ++i) {
                std::vector<std::vector<T>> sub_x_test(x_test.size());

                for (size_t j = 0; j < sub_x_test.size(); ++j) {
                    for (size_t p = 0; p < this->models_feature_indexes[i].size(); ++p) {
                        sub_x_test[j].push_back(x_test[j][this->models_feature_indexes[i][p]]);
                    }
                }

                std::vector<T> res = this->models[i].predict(sub_x_test);
                for (size_t j = 0; j < x_test.size(); ++j) {
                    mp[j][res[j]]++;
                }
            }
            for (size_t i = 0; i < x_test.size(); ++i) {
                size_t mx = 0;
                T ans;
                for (const auto &x : mp[i]) {
                    if (x.second > mx) {
                        mx = x.second;
                        ans = x.first;
                    }
                }
                y_pred_new[i] = ans;
            }
            return y_pred_new;
        }
    }
}