#include "./linearalgebra.hpp"

namespace hina {
    namespace linearalgebra {
        template <typename T>
        T vector_scalar_product(const std::vector<T> &vec_1, const std::vector<T> &vec_2) {
            if (vec_1.size() != vec_2.size()) {
                std::cerr << "Vectors in scalar product must be the same size" << std::endl;
                return;
            }

            T product = 0.0;
            for (size_t i = 0; i < vec_1.size(); ++i) {
                product += vec_1[i] * vec_2[i];
            }

            return product;
        }

        template <typename T>
        std::vector<T> multiplyMatrixByWeights(const std::vector<std::vector<T>> &objects_matrix, const std::vector<T> &weights) {
            std::vector<T> product(objects_matrix.size());

            for (size_t i = 0; i < objects_matrix.size(); ++i) {
                product[i] = vector_scalar_product(objects_matrix[i], weights[i]);
            }

            return product;
        }
    }
}
