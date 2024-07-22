#include "./gradient_boosting.hpp"
#include "../Metrics/metrics.hpp"

namespace hina {
    namespace tree_ensemble {
        template <typename T, typename P>
        void GBDTClassifier<T, P>::fit(const std::vector<std::vector<T>>& x_train, const std::vector<P>& y_train) {
            for (size_t i = 0; i < y_train.size(); ++i) this->n_classes = y_train[i] + 1 > this->n_classes ? y_train[i] + 1 : this->n_classes;
            std::vector<std::vector<T>> current_predictions(x_train.size(), std::vector<T>(this->n_classes, 0.0));
            std::vector<std::vector<T>> y_train_OHE(y_train.size());

            for (size_t i = 0; i < y_train.size(); ++i) {
                for (size_t class_label = 0; class_label < this->n_classes; ++class_label) {
                    if (y_train[i] == class_label) y_train_OHE[i].push_back(1.0);
                    else y_train_OHE[i].push_back(0.0);
                }
            }
            for (size_t i = 0; i < this->n_estimators; ++i) {
                std::vector<std::vector<T>> probabilities = hina::metrics::softmax(current_predictions);
                
                if (this->show_logs) std::cout << "[Tree " << i + 1 << "] Cross entropy loss: " << hina::metrics::cross_entopy_loss(y_train_OHE, probabilities) << std::endl;

                for (size_t class_label = 0; class_label < this->n_classes; ++class_label) {
                    hina::decision_tree::DecisionTreeRegressor<T> model(this->max_tree_depth);

                    std::vector<T> residuals(y_train.size());
                    for (size_t j = 0; j < residuals.size(); ++j) {
                        residuals[j] = y_train_OHE[j][class_label] - probabilities[j][class_label];
                    }

                    model.fit(x_train, residuals);
                    this->models[class_label].push_back(model);

                    std::vector<T> model_prediction = model.predict(x_train);
                    for (size_t j = 0; j < x_train.size(); ++j) {
                        current_predictions[j][class_label] += this->learning_rate * model_prediction[j];
                    }
                }
            }
        }

        template <typename T, typename P>
        std::vector<T> GBDTClassifier<T, P>::predict(const std::vector<std::vector<T>>& x_test) {
            std::vector<std::vector<T>> current_predictions(x_test.size(), std::vector<T>(this->n_classes, 0.0));
            for (size_t i = 0; i < this->n_estimators; ++i) {
                for (size_t class_label = 0; class_label < this->n_classes; ++class_label) {
                    std::vector<T> model_prediction = this->models[class_label][i].predict(x_test);

                    for (size_t j = 0; j < x_test.size(); ++j) {
                        current_predictions[j][class_label] += this->learning_rate * model_prediction[j];
                    }
                }
            }
            std::vector<T> y_pred(x_test.size());
            for (size_t i = 0; i < x_test.size(); ++i) {
                T mx_prob = 0.0;
                P answer = 0;
                for (size_t class_label = 0; class_label < this->n_classes; ++class_label) {
                    if (current_predictions[i][class_label] > mx_prob) {
                        mx_prob = current_predictions[i][class_label];
                        answer = class_label;
                    }
                }
                y_pred[i] = answer;
            }
            return y_pred;
        }
    }
}