#pragma once
#include <vector>


namespace hina {
    namespace linearalgebra {
        template <typename T>
        T vector_scalar_product(const std::vector<T> &vec_1, const std::vector<T> &vec_2);

        template <typename T>
        std::vector<T> multiplyMatrixByWeights(const std::vector<std::vector<T>> &objects_matrix, const std::vector<T> &weights);
    }
}

