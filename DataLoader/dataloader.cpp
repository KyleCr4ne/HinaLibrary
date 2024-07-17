#include "./dataloader.hpp"

namespace hina {
    namespace dataloader {

        template <typename T, typename P>
        void FileData<T, P>::read_csv(const std::string &file_name, bool has_header) {
            std::ifstream file(file_name);
            if (!file.is_open()) {
                std::cerr << "Read CSV file error: unable to open file " << file_name << std::endl;
                return; 
            }

            std::string line;

            while (std::getline(file, line)) {
                std::stringstream ss(line);
                std::string cell;
                std::vector<T> row;
                if (has_header) {
                    while (std::getline(ss, cell, ',')) {
                        header.push_back(cell);
                    }
                    has_header = false;
                } else {
                    while (std::getline(ss, cell, ',')) {
                        row.push_back(static_cast<T>(std::stod(cell)));
                    }
                    objects_targets.push_back(static_cast<P>(row.back()));
                    row.pop_back();
                    objects_data.push_back(row);
                }

            }
            file.close();
        }

        template <typename T, typename P>
        void FileData<T, P>::train_test_split(const double train_size) {
            if (objects_data.empty()) {
                std::cerr << "First read data" << std::endl;
                return;
            }
            size_t num_of_train = (size_t) std::round(objects_data.size() * train_size);

            std::vector<std::pair<std::vector<T>, P>> DATA;
            for (size_t i = 0; i < objects_data.size(); ++i) {
                DATA.push_back({objects_data[i], objects_targets[i]});
            }

            std::random_device rd;
            std::mt19937 g(rd());
            std::shuffle(DATA.begin(), DATA.end(), g);

            for (size_t i = 0; i < num_of_train; ++i) {
                x_train.push_back(DATA[i].first);
                y_train.push_back(DATA[i].second);
            }
            for (size_t i = num_of_train; i < DATA.size(); ++i) {
                x_test.push_back(DATA[i].first);
                y_test.push_back(DATA[i].second);
            }
        }
    }
}
