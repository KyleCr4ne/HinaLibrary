#include "./gradient_boosting.hpp"
#include "../Metrics/metrics.hpp"

namespace hina {
    namespace tree_ensemble {
        template <typename T, typename P>
        void GBDTRegressor<T, P>::fit(const std::vector<std::vector<T>>& x_train, const std::vector<P>& y_train) {
            std::vector<T> current_predictions(x_train.size(), 0.0);

            for (size_t i = 0; i < this->n_estimators; ++i) {
                hina::decision_tree::DecisionTreeRegressor<T> model(this->max_tree_depth);
                
                if (this->show_logs) std::cout << "[Tree " << i + 1 << "] MAE: " << hina::metrics::mean_absolute_error(y_train, current_predictions) << std::endl;

                std::vector<T> residuals(x_train.size(), 0.0);
                for (size_t j = 0; j < x_train.size(); ++j) {
                    residuals[j] = y_train[j] - current_predictions[j];
                }
                

                model.fit(x_train, residuals);
                this->models.push_back(model); 

                std::vector<T> model_prediction = model.predict(x_train);

                for (size_t j = 0; j < x_train.size(); ++j) {
                    current_predictions[j] += this->learning_rate * model_prediction[j];
                }
            }
        }
        template <typename T, typename P>
        std::vector<T> GBDTRegressor<T, P>::predict(const std::vector<std::vector<T>>& x_test) {
            std::vector<T> y_pred(x_test.size(), 0.0);
            for (size_t i = 0; i < this->n_estimators; ++i) {
                std::vector<T> model_prediction = this->models[i].predict(x_test);
                for (size_t j = 0; j < x_test.size(); ++j) y_pred[j] += this->learning_rate * model_prediction[j];
            }
            return y_pred;
        }

    }
}