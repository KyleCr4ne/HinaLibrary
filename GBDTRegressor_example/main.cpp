#include "../Hina.hpp"

#include <vector>

int main() {
    hina::dataloader::FileData<double, double> dloader;
    
    dloader.read_csv("./Battery_RUL.csv", false);
    
    dloader.train_test_split(0.8);
    
    hina::tree_ensemble::GBDTRegressor model(50, 3, true); // show_logs = true
    
    model.fit(dloader.x_train, dloader.y_train);

    std::vector<double> ans = model.predict(dloader.x_test);

    std::cout << "Mean absolute score on test part: " << hina::metrics::mean_absolute_error(dloader.y_test, ans) << std::endl;

    return 0;
}