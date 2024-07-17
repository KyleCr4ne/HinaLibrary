#include "../Hina.hpp"

int main() {
    hina::dataloader::FileData<double, int> dloader;
    dloader.read_csv("./data.txt", false);
    dloader.train_test_split(0.8);
    return 0;
}