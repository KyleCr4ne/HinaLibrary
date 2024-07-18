#include "../Hina.hpp"

int main() {
    hina::dataloader::FileData<double, double> dloader;
    dloader.read_csv("./data.txt", false);
    dloader.train_test_split(0.8);

    hina::linear_models::BinaryLogisticRegression<double> clf;

    clf.fit(dloader.x_train, dloader.y_train);

    std::vector<double> y_pred = clf.predict_proba(dloader.x_test);

    std::cout << "BCE Loss: " << hina::metrics::BCE_loss(dloader.y_test, y_pred) << std::endl;

    for (size_t i = 0; i < y_pred.size(); ++i) {
        std::cout << dloader.y_test[i] << " " << y_pred[i] << std::endl;
    }
    return 0;
}