#pragma once

#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <iostream>
#include <cmath>
#include <random>
#include <algorithm>

namespace hina {
    namespace dataloader {
        template <typename T, typename P>
        class FileData {
            public:
            std::vector<std::vector<T>> objects_data;
            std::vector<P> objects_targets;
            std::vector<std::string> header;

            std::vector<std::vector<T>> x_train;
            std::vector<std::vector<T>> x_test;
            std::vector<P> y_train;
            std::vector<P> y_test;

            FileData() {};

            void read_csv(const std::string &file_name, bool has_header);
            void train_test_split(const double train_size);
            ~FileData() {};
        };
    }
}
