#include <valarray>
#include <vector>
#include <random>
#include <cmath>
#include <cassert>
#include <algorithm>
#include <iostream>

using namespace std;

#define int long long
#define uint unsigned long long
#define float64 double

constexpr int inf = 0x3f3f3f3f;

class Data {
public:
    float64 x, y;

    Data(float64 x, float64 y)
        : x(x), y(y) {
    }

    friend float64 euclid(const Data &lhs, const Data &rhs) {
        return sqrt((lhs.x - rhs.x) * (lhs.x - rhs.x) + (lhs.y - rhs.y) * (lhs.y - rhs.y));
    }

    Data() = default;

};

class Dataset {
    valarray<Data> X;
    valarray<int> y;

public:
    Dataset() = default;

    Dataset(const valarray<Data> &x, const valarray<int> &y)
        : X(x), y(y) {
        assert(x.size() == y.size());
    }

    [[nodiscard]] uint size() const {
        return X.size();
    }

    pair<Data, int> get(int idx) {
        assert(idx < X.size());
        return {X[idx], y[idx]};
    }
};

class Classifier {
public:
    virtual ~Classifier() = default;

    virtual void fit(Dataset dataset) = 0;

    virtual int predict(Data data) = 0;

    virtual float64 score(Dataset dataset) = 0;

    virtual float64 accuracy(Dataset dataset) = 0;
};

class NeuralNetworkClassifier : public Classifier {
    vector<valarray<float64>> W1;  // Weights for input -> hidden layer
    valarray<float64> b1;          // Biases for hidden layer
    vector<valarray<float64>> W2;  // Weights for hidden -> output layer
    valarray<float64> b2;          // Biases for output layer
    int n_labels;
    int n_hidden;                  // Number of hidden units
    float64 lr;                    // Learning rate
    int epochs;                    // Number of training epochs

public:
    ~NeuralNetworkClassifier() override = default;

    explicit NeuralNetworkClassifier(int n_hidden = 10, float64 lr = 0.001, int epochs = 1000)
        : n_hidden(n_hidden), lr(lr), epochs(epochs), n_labels(0) {}

    void fit(Dataset dataset) override {
        // Determine the number of unique labels
        n_labels = find_max_label(dataset) + 1;
        int n_features = 2; // Assume 2D data points

        // Initialize weights and biases
        initialize_weights(n_features);

        // Training loop
        for (int epoch = 0; epoch < epochs; ++epoch) {
            for (int i = 0; i < dataset.size(); ++i) {
                auto [data, label] = dataset.get(i);
                valarray<float64> x = {data.x, data.y};

                // Forward pass
                auto [h, o] = forward(x);

                // Compute gradient via backpropagation
                backward(x, h, o, label);
            }
        }
    }

    int predict(Data data) override {
        valarray<float64> x = {data.x, data.y};

        // Forward pass
        auto [h, o] = forward(x);

        // Predict the label with the highest output probability (softmax)
        return distance(begin(o), max_element(begin(o), end(o)));
    }

    float64 score(Dataset dataset) override {
        return 0.0; // No direct score function for neural networks
    }

    float64 accuracy(Dataset dataset) override {
        uint siz = dataset.size();
        int positive = 0;
        for (int i = 0; i < siz; ++i) {
            auto [k, v] = dataset.get(i);
            positive += predict(k) == v;
        }
        return 1.0 * positive / siz;
    }

private:
    void initialize_weights(int n_features) {
        random_device rd;
        mt19937 gen(rd());
        normal_distribution<float64> dist(0, 1);

        // Initialize weights and biases for input -> hidden layer
        W1 = vector<valarray<float64>>(n_hidden, valarray<float64>(n_features));
        for (auto &w : W1)
            generate(begin(w), end(w), [&]() { return dist(gen); });
        b1 = valarray<float64>(0.0, n_hidden);

        // Initialize weights and biases for hidden -> output layer
        W2 = vector<valarray<float64>>(n_labels, valarray<float64>(n_hidden));
        for (auto &w : W2)
            generate(begin(w), end(w), [&]() { return dist(gen); });
        b2 = valarray<float64>(0.0, n_labels);
    }

    pair<valarray<float64>, valarray<float64>> forward(const valarray<float64> &x) {
        // Forward pass: Input -> Hidden layer
        valarray<float64> h(n_hidden);
        for (int i = 0; i < n_hidden; ++i) {
            h[i] = (W1[i] * x).sum() + b1[i];
            h[i] = tanh(h[i]); // Activation (tanh)
        }

        // Forward pass: Hidden -> Output layer
        valarray<float64> o(n_labels);
        for (int i = 0; i < n_labels; ++i) {
            o[i] = (W2[i] * h).sum() + b2[i];
        }

        // Apply softmax to the output layer
        o = softmax(o);

        return {h, o};
    }

    void backward(const valarray<float64> &x, const valarray<float64> &h, const valarray<float64> &o, int label) {
        // Convert label to one-hot encoding
        valarray<float64> y(0.0, n_labels);
        y[label] = 1.0;

        // Calculate gradients for output layer (softmax derivative)
        valarray<float64> dL_do = o - y;

        // Update weights and biases for hidden -> output layer
        for (int i = 0; i < n_labels; ++i) {
            W2[i] -= lr * dL_do[i] * h;
            b2[i] -= lr * dL_do[i];
        }

        // Calculate gradients for hidden layer
        valarray<float64> dL_dh(n_hidden);
        dL_dh = 0.0;  // Initialize gradient for hidden layer to zero

        // Accumulate gradient contributions from all output units
        for (int j = 0; j < n_labels; ++j) {
            for (int i = 0; i < n_hidden; ++i) {
                dL_dh[i] += dL_do[j] * W2[j][i];
            }
        }

        // Apply tanh derivative to the hidden layer gradient
        for (int i = 0; i < n_hidden; ++i) {
            dL_dh[i] *= (1 - h[i] * h[i]);  // tanh derivative
        }

        // Update weights and biases for input -> hidden layer
        for (int i = 0; i < n_hidden; ++i) {
            W1[i] -= lr * dL_dh[i] * x;
            b1[i] -= lr * dL_dh[i];
        }
    }


    valarray<float64> softmax(const valarray<float64> &v) {
        valarray<float64> exp_v = exp(v - v.max()); // Avoid overflow
        return exp_v / exp_v.sum();
    }

    int find_max_label( Dataset &dataset) {
        int max_label = -1;
        for (int i = 0; i < dataset.size(); ++i) {
            auto [_, label] = dataset.get(i);
            if (label > max_label) {
                max_label = label;
            }
        }
        return max_label;
    }
};


signed main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);
    cout.tie(nullptr);

    int siz = 5000;
    valarray<Data> X(siz);
    valarray<int> y(siz);

    random_device dev;
    mt19937 rng(dev());
    uniform_real_distribution real_dist(0.0, 1.0);

    float64 radius = 1.5;
    for (int i = 0; i < siz / 3; ++i) {
        X[i] = {-1.0 + real_dist(rng) * radius, 0.0 + real_dist(rng) * radius};
        y[i] = 0;
    }
    for (int i = siz / 3; i < 2 * siz / 3; ++i) {
        X[i] = {1.0 + real_dist(rng) * radius, 0.0 + real_dist(rng) * radius};
        y[i] = 1;
    }
    for (int i = 2 * siz / 3; i < siz; ++i) {
        X[i] = {0.0 + real_dist(rng) * radius, 1.0 + real_dist(rng) * radius};
        y[i] = 2;
    }

    uniform_int_distribution int_dist(0ll, inf);
    for (int i = 0; i < siz; ++i) {
        int idx = int_dist(rng) % (siz - i) + i;
        swap(X[i], X[idx]);
        swap(y[i], y[idx]);
    }

    float64 split = 0.8;
    size_t split_idx = siz * split; // NOLINT(*-narrowing-conversions)
    Dataset train_set(X[slice(0, split_idx, 1)], y[slice(0, split_idx, 1)]);
    Dataset test_set(X[slice(split_idx, siz - split_idx, 1)], y[slice(split_idx, siz - split_idx, 1)]);

    // Neural Network Classifier for Multi-label Classification
    NeuralNetworkClassifier nn_clf(16, 0.001, 1000);
    nn_clf.fit(train_set);
    cout << "Neural Network Model (Multi-label): \n";
    float64 acc = 100 * nn_clf.accuracy(test_set);
    cout << "Acc: " << acc << '%' << '\n';

    return 0;
}
