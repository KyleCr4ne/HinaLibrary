# Hina. The C++ Machine Learning library.

**Hina** is a library written with the C++ programming language, which implements the basic methods of classical machine learning for working with tabular data. 

Writing this library allowed me to delve deeper into the knowledge of this interesting area of data science, understand the engineering part and realize some details of the implementation of basic algorithms.

*I would like to express my deep gratitude to Professor A.G. Dyakonov, Professor K.V. Vorontsov and the author of the YouTube channel "selfedu" S.M. Balakirev for making lectures and text materials on machine and deep learning available to the public, which became the basis for this project.*
***
## Content
* [The main header file](#the-main-header-file)
* [The main namespace](#the-main-namespace)
* [Project file tree](#project-file-tree)
* [FileData](#filedata)
* [LinearAlgebra](#linearalgebra)
* [Metrics](#metrics)
* [LossGradients](#lossgradients)
* [SGD optimizer for linear models](#linear-sgd-optimizer)
* [Linear SGD Regression](#linear-sgd-regression)
* [Binary Logistic Regression](#binary-logistic-regression)
* [DecisionTree](#decisiontree)
* [RandomForest](#randomforest)
* [Gradient boosted decision trees](#gradient-boosted-decision-trees)

***
#### The main header file[](###the-main-header-file)

```cpp
#include "Hina.hpp"
```
#### The main namespace[](#the-main-namespace)
```cpp
namespace hina
```
#### Project file tree[](#project-file-tree)
```
├── DataLoader
│   ├── dataloader.cpp
│   └── dataloader.hpp
├── GradientBoosting
│   ├── gbdt_classifier.cpp
│   ├── gbdt_regressor.cpp
│   └── gradient_boosting.hpp
├── LinearAlgebra
│   ├── linearalgebra.cpp
│   └── linearalgebra.hpp
├── LinearModels
│   ├── linear_models.cpp
│   └── linear_models.hpp
├── LossGradients
│   ├── lossgradients.cpp
│   └── lossgradients.hpp
├── Metrics
│   ├── metrics.cpp
│   └── metrics.hpp
├── Optimizers
│   ├── linear_stochastic_gradient_descent_optimizer.cpp
│   └── linear_stochastic_gradient_descent_optimizer.hpp
├── RandomForest
│   ├── random_forest.cpp
│   ├── random_forest.hpp
│   ├── random_forest_classifier.cpp
│   └── random_forest_regressor.cpp
└── Tree
    ├── decision_tree.hpp
    ├── decision_tree_classifier.cpp
    └── decision_tree_regressor.cpp
```
****
#### FileData[](#filedata)
A simple class that allows you to read data from a text file.
```cpp
hina::dataloader::FileData<T, P>
```
**read_csv** - reading data from a csv file. 
```cpp
void hina::dataloader::FileData<T, P>::read_csv(const  std::string  &file_name, bool  has_header)
```
**train_test_split** -  divides the uploaded data into training and test subsamples. 
```cpp
void hina::dataloader::FileData<T, P>::train_test_split(const  double  train_size);
```
Example:
```cpp
#include  "../Hina.hpp"
int  main() {
hina::dataloader::FileData<double, int>  dloader;
dloader.read_csv("./data.txt", false);
dloader.train_test_split(0.8);
// Allows you to use dloader.x_train; dloader.y_train; dloader.x_test, dloader.y_test;
return  0;
}
```
****
#### LinearAlgebra[](#linearalgebra)
**The scalar product of vectors**
```cpp
template <typename  T>
T  hina::linearalgebra::vector_scalar_product(const  std::vector<T> &vec_1, const  std::vector<T> &vec_2);
```
  **The product of the matrix of objects and the vector of weights**
```cpp
template <typename  T>
std::vector<T> hina::linearalgebra::multiplyMatrixByWeights(const  std::vector<std::vector<T>> &objects_matrix, const  std::vector<T> &weights);
```
****
#### Metrics[](#metrics)

A module that includes the implementation of some metrics and auxiliary functions
**Mean squared error**
```cpp
template <typename  T>
T  hina::metrics::mean_squared_error(const  std::vector<T> &target_values, const  std::vector<T> &received_values);
```
**Mean absolute error**
```cpp
template <typename  T>
T  hina::metrics::mean_absolute_error(const  std::vector<T> &target_values, const  std::vector<T> &received_values);
```
**Sigmoid activation function**
 ```cpp
template <typename  T>
T  hina::metrics::sigmoid (const  T  x);
```
**Binary cross entropy loss function**
```cpp
template <typename  T>
T  hina::metrics::BCE_loss(const  std::vector<T> &target_values, const  std::vector<T> &received_values);
```
**Accuracy score**
```cpp
template <typename  T>
T  hina::metrics::accuracy_score(const  std::vector<T> &target_values, const  std::vector<T> &received_values);
```
**Softmax activation function**
```cpp
template <typename  T>
std::vector<std::vector<T>> hina::metrics::softmax(const  std::vector<std::vector<T>> &X);
```

$$softmax([x_{11}, ..., x_{1m}], ..., [x_{l1}, ..., x_{lm}])_{c\in m} = \frac{1}{l} \Sigma_{i}^{l}\frac{exp(x_{lc})}{\Sigma_{k}^{m}exp(x_{lk})}$$

**Cross entropy loss function**
```cpp
template <typename  T>
T  hina::metrics::cross_entopy_loss(const  std::vector<std::vector<T>> &target_probability_distribution, const  std::vector<std::vector<T>> &  received_probability_distribution);
```
$Loss(x, y) = -\frac{1}{l}\Sigma_{i}^{l} \Sigma_{k}^{m}y_i log(x_k)$
****

#### LossGradients[](#lossgradients)
Auxiliary functions that allow you to calculate the gradient value of the selected loss function at a specified point. 
```cpp
template <typename  T>
std::pair<std::vector<T>, T> hina::lossgradients::mean_squared_loss_gradient(const  std::vector<T> &sample, const  T  y, const  std::vector<T> &weights, const  T  bias);

template <typename  T>
std::pair<std::vector<T>, T> hina::lossgradients::mean_absolute_loss_gradient(const  std::vector<T> &sample, const  T  y, const  std::vector<T> &weights, const  T  bias);

template <typename  T>
std::pair<std::vector<T>, T> hina::lossgradients::BCE_loss_gradient(const  std::vector<T> &sample, const  T  y, const  std::vector<T> &weights, const  T  bias);
```
#### Linear SGD optimizer[](#linear-sgd-optimizer)
 **Stochastic gradient descent** for linear models. Includes a class *hina::linear_sgd_optimizer::StochasticGradientDescent* and and an auxiliary class of optimized parameters *hina::linear_sgd_optimizer::StochasticGradientDescentParams*. 
 ```cpp
 template <typename  T>
class  StochasticGradientDescentParams {
public:
std::vector<std::vector<T>>  X; // objects-features matrix
std::vector<T>  Y; // objects targets
std::function<T(const  std::vector<T>&, const  std::vector<T>&)>  loss; // Optimized functionality
std::function<std::pair<std::vector<T>, T>(const  std::vector<T>  &, const  T,const  std::vector<T>  &, const  T)>  gradient;
size_t  batch_size  =  32;
size_t  epoches  =  100;
T  learning_rate  =  0.05;
std::string  regularization  =  "None"; // L1 or L2 regularization
T  regularization_coef  =  0.05;
T  end_loss_value  =  1e-6;
};
```
This SGD optimizer class return a pair of best weights and bias
```cpp
template <typename  T>
std::pair<std::vector<T>, T> StochasticGradientDescent<T>::optimize(StochasticGradientDescentParams<T>*  params);
```
***
#### Linear SGD Regression[](#linear-sgd-regression)
The classical implementation of linear regression. The mean squared error is used as a loss function. And stochastic gradient descent is used for optimization. 
$y_{i} = w_{1} * x_{i1} + ... + w_{m} * x_{im} + w_{0}, y_i \in R$
```cpp
template <typename T>
hina::linear_models::SGDRegressor<T> model;

// We can change some params:
model.epoches_ = 100;
model.batch_size_ = 32;
model.regularization_ = "L1"; // or "L2" or "None";
model.regularization_coef_  =  0.05;
```
To fit model use class method **fit**:
```cpp
template <typename T>
void hina::linear_models::SGDRegressor<T>::fit(const  std::vector<std::vector<T>> &x_train, const  std::vector<T> &y_train);
```
To get predictions use class method **predict**:
```cpp
template <typename T>
std::vector<T> hina::linear_models::SGDRegressor<T>::predict(const  std::vector<std::vector<T>> &x_test);
```
****
#### Binary Logistic Regression[](#binary-logistic-regression)
A linear model for binary classification. Binary cross entropy is used for optimization.
$y_{i} = \sigma(w_{1} * x_{i1} + ... + w_{m} * x_{im} + w_{0}), y_i \in [0, 1]$
```cpp
template <typename T>
hina::linear_models::BinaryLogisticRegression<T> model;

// We can also change some params:
model.epoches_ = 100;
model.batch_size_ = 32;
model.regularization_ = "L1"; // or "L2" or "None";
model.regularization_coef_  =  0.05;
```
To fit model use class method **fit**:
```cpp
template <typename T>
void hina::linear_models::BinaryLogisticRegression<T>::fit(const  std::vector<std::vector<T>> &x_train, const  std::vector<T> &y_train);
```
To get predictions use class method **predict**:
```cpp
template <typename T>
std::vector<T> hina::linear_models::BinaryLogisticRegression<T>::predict(const  std::vector<std::vector<T>> &x_test);
```
And you can also use **predict_proba** to get an answer in the form of probability:
```cpp
template <typename T>
std::vector<T> hina::linear_models::BinaryLogisticRegression<T>::predict_proba(const  std::vector<std::vector<T>> &x_test);
```
****

#### DecisionTree[](#decisiontree)
A greedy strategy is used to build decision trees. At each step, branching tries to get an information gain. For regression trees, the criterion is the variance of targets in the subtree, and for classification trees - the gini criterion.

The node of the tree is represented as a class:
```cpp
template <typename  T, typename  P>
class  Node {
public:
size_t  split_feature_idx  =  0;
T  treshold  =  0.0;
bool  is_leaf  =  false;
P  label; // Double if its regression tree, int else.
T  label_proba;
Node*  left_child  =  nullptr;
Node*  right_child  =  nullptr;
};
```
We can define the maximum depth of tree. 

The main function of tree is **tree_builder**, which recursively builds a tree and returns its root.
```cpp
template <typename  T, typename  P  = int>
hina::decision_tree::DecisionTreeClassifier<T, P>
```
and
```cpp
template <typename  T>
hina::decision_tree::DecisionTreeRegressor<T>
```
contain methods:
```cpp
template <typename  T, typename  P>
void hina::decision_tree::BaseDecisionTree<T, P>::fit(const  std::vector<std::vector<T>> &x_train, const  std::vector<P> &y_train);
template  <typename T,  typename P>
std::vector<P> hina::decision_tree::BaseDecisionTree<T, P>predict(const  std::vector<std::vector<T>> &x_test);
```
****
#### RandomForest[](#randomforest)
The random forest method is an ensemble on decision trees using bootstrap and allocation of a random feature subspace for each tree. 

One of the features of this method is the possibility of parallel calculations. In my implementation, parallel calculations on threads are also used to build each tree.

We can choose the maximum depth of trees (**max_tree_depth**) and number of estimators (**n_estimators**). 
```cpp
template <typename  T  = double, typename  P  = double>
hina::tree_ensemble::RandomForestRegressor<T, P> rfr;
hina::tree_ensemble::RandomForestClassifier<T, P> rfc;
```
And it also has methods **fit** and **predict**. 
```cpp
template <typename  T  = double, typename  P  = double, typename  DecisionTreeType  =  hina::decision_tree::DecisionTreeRegressor<T>>
void hina::decision_tree::BaseRandomForest<T, P>>::fit(const  std::vector<std::vector<T>>&  x_train, const  std::vector<P>&  y_train);
std::vector<T> hina::decision_tree::BaseRandomForest<T, P>>::predict(const  std::vector<std::vector<T>>&  x_test);
```
****

#### Gradient boosted decision trees[](#gradient-boosted-decision-trees)
The base params of GBDT are maximum tree depth, number of estimators (base models - trees) and learning rate (**set 0.1**). 

Gradient boosting over decision trees consistently builds a new tree based on the gradient of the optimized loss function. For the regression problem, the mean squared loss function is used, in fact, allowing each new tree T to be trained on the difference between the true target values and those predicted at T-1 step. For the classification task, M trees are built at each step, each of which is responsible for its own class. And the cross entropy is optimized. Softmax is used to get answers for classification tasks. 

```cpp
template <typename  T  = double, typename  P  = double>
hina::tree_ensemble::GBDTRegressor<T, P> reg(size_t  n_estimators_, size_t  max_tree_depth_, bool  show_logs_);
hina::tree_ensemble::GBDTClassifier<T, P> clf(size_t  n_estimators_, size_t  max_tree_depth_, bool  show_logs_);
```
```cpp
template <typename  T  = double, typename  P  = double, typename  DecisionTreeType  =  hina::decision_tree::DecisionTreeRegressor<T>>
void hina::decision_tree::BaseGBDT<T, P, DecisionTreeType>>::fit(const  std::vector<std::vector<T>>&  x_train, const  std::vector<P>&  y_train);
std::vector<T> hina::decision_tree::BaseGBDT<T, P, DecisionTreeType>>::predict(const  std::vector<std::vector<T>>&  x_test);
```
Param **show_logs** is used to show loss value on each step of fitting.

You can see examples for classification:
  ```cpp
#include  "../Hina.hpp"
#include  <vector>

int  main() {
	hina::dataloader::FileData<long  double, long  double>  dloader;
	dloader.read_csv("./train.csv", false);
	dloader.train_test_split(0.8);
	hina::tree_ensemble::GBDTClassifier<long  double, long  double>  model(50, 4, true); // show_logs = true
	model.fit(dloader.x_train, dloader.y_train);
	return  0;
}
```

And regression:
  ```cpp
#include  "../Hina.hpp"
#include  <vector>

int  main() {
	hina::dataloader::FileData<double, double>  dloader;
	dloader.read_csv("./Battery_RUL.csv", false);
	dloader.train_test_split(0.8);
	hina::tree_ensemble::GBDTRegressor  model(50, 3, true); // show_logs = true
	model.fit(dloader.x_train, dloader.y_train);
	std::vector<double>  ans  =  model.predict(dloader.x_test);
	std::cout  <<  "Mean absolute score on test part: "  <<  hina::metrics::mean_absolute_error(dloader.y_test, ans) <<  std::endl;
	return  0;
}
```
****