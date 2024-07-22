#include "../Hina.hpp"

#include <vector>

int main() {
    hina::dataloader::FileData<long double, long double> dloader;
    
    dloader.read_csv("./train.csv", false);
    
    dloader.train_test_split(0.8);
    
    hina::tree_ensemble::GBDTClassifier<long double, long double> model(50, 4, true); // show_logs = true
    model.fit(dloader.x_train, dloader.y_train);

    return 0;
}