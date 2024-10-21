#include <cassert>
#include <random>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <cmath>
#include <map>
#include <vector>
#include <queue>
#include <valarray>

using namespace std;

#define int long long
#define uint unsigned long long
#define float64 double

constexpr int     inf = 0x3f3f3f3f;
constexpr float64 PI  = 3.1415926;

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
    valarray<int>  y;

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

class NearestNeighborClassifier : public Classifier {
    int     k;
    Dataset dataset;
    bool    verbose;

public:
    ~NearestNeighborClassifier() override = default;

    explicit NearestNeighborClassifier(int k, bool verbose = false)
        : k(k),
          verbose(verbose) {
    }

    void fit(Dataset dataset) override {
        this->dataset = dataset;
    }

    int predict(const Data data) override {
        priority_queue<pair<float64, int> > candidates;

        const uint siz = dataset.size();
        for (int i = 0; i < siz; ++i) {
            auto [feature, label] = dataset.get(i);

            float64 dis = euclid(data, feature);
            candidates.emplace(dis, label);
            if (candidates.size() > k)
                candidates.pop();
        }

        map<int, int> vote;
        while (!candidates.empty()) {
            auto [dis, label] = candidates.top();
            candidates.pop();
            vote[label] += 1;
        }

        pair<int, int> nax = {0, -inf};
        for (auto &[label, cnt]: vote)
            if (cnt > nax.second)
                nax = {label, cnt};

        return nax.first;
    }

    float64 score(Dataset dataset) override {
        // knn does not have a loss function to minimize during training
        // so scoring a knn is unnecessary
        float64 score = 0.0;
        return score;
    }

    float64 accuracy(Dataset dataset) override {
        uint siz      = dataset.size();
        int  positive = 0;
        for (int i = 0; i < siz; ++i) {
            auto [k, v] = dataset.get(i);
            positive += predict(k) == v;
        }

        if (verbose)
            cout << positive << " out of " << siz << " samples correct" << '\n';

        return 1.0 * positive / siz; // NOLINT(*-narrowing-conversions)
    }
};

class LinearSVMClassifier : public Classifier {
    valarray<float64> weights; // Weight vector
    float64           bias;    // Bias term
    float64           lr;      // Learning rate
    int               epochs;  // Number of training epochs

    bool verbose;

public:
    ~LinearSVMClassifier() override = default;

    explicit LinearSVMClassifier(float64 lr = 0.001, int epochs = 1000, bool verbose = false)
        : lr(lr), epochs(epochs), bias(0.0), verbose(verbose) {
    }

    void fit(Dataset dataset) override {
        uint n_features = 2; // Assume 2D data points

        weights = valarray<float64>(0.0, n_features);

        for (int epoch = 0; epoch < epochs; ++epoch) {
            for (int i = 0; i < dataset.size(); ++i) {
                auto [data, label] = dataset.get(i);

                valarray<float64> x = {data.x, data.y};
                // Convert label to {-1, 1}
                int y = (label == 1) ? 1 : -1;

                // Check for mis-classification (y * (w * x + b) < 1)
                if (y * (dot(weights, x) + bias) < 1) {
                    // Update weights and bias
                    weights += lr * (y * x);
                    bias += lr * y;
                }
            }
        }
    }

    int predict(Data data) override {
        valarray<float64> x = {data.x, data.y};
        return (dot(weights, x) + bias >= 0) ? 1 : 0;
    }

    float64 score(Dataset dataset) override {
        // SVM scoring doesn't apply in the usual sense as there is no loss function to optimize
        return 0.0;
    }

    float64 accuracy(Dataset dataset) override {
        uint siz      = dataset.size();
        int  positive = 0;
        for (int i = 0; i < siz; ++i) {
            auto [k, v] = dataset.get(i);
            positive += predict(k) == v;
        }

        if (verbose)
            cout << positive << " out of " << siz << " samples correct" << '\n';

        return 1.0 * positive / siz; // NOLINT(*-narrowing-conversions)
    }

private:
    float64 dot(const valarray<float64> &a, const valarray<float64> &b) {
        return (a * b).sum();
    }
};

class LinearSVMClassifierMultiLabel : public Classifier {
    vector<valarray<float64> > weights;  // Weight vectors for each label
    vector<float64>            biases;   // Bias terms for each label
    float64                    lr;       // Learning rate
    int                        epochs;   // Number of training epochs
    int                        n_labels; // Number of unique labels
    bool                       verbose;

public:
    ~LinearSVMClassifierMultiLabel() override = default;

    explicit LinearSVMClassifierMultiLabel(float64 lr = 0.001, int epochs = 1000, bool verbose = false)
        : lr(lr), epochs(epochs), n_labels(0), verbose(verbose) {
    }

    void fit(Dataset dataset) override {
        // Determine the number of unique labels
        n_labels = find_max_label(dataset) + 1;
        // Initialize weight vectors and biases for each label
        uint n_features = 2; // Assume 2D data points

        weights = vector<valarray<float64> >(n_labels, valarray<float64>(0.0, n_features));
        biases  = vector<float64>(n_labels, 0.0);

        for (int epoch = 0; epoch < epochs; ++epoch) {
            for (int i = 0; i < dataset.size(); ++i) {
                auto [data, label] = dataset.get(i);

                valarray<float64> x = {data.x, data.y};
                for (int lbl = 0; lbl < n_labels; ++lbl) {
                    // Convert label to {-1, 1} (one-vs-rest)
                    int y = (label == lbl) ? 1 : -1;

                    // Check for mis-classification (y * (w * x + b) < 1)
                    if (y * (dot(weights[lbl], x) + biases[lbl]) < 1) {
                        // Update weights and bias
                        weights[lbl] += lr * (y * x);
                        biases[lbl] += lr * y;
                    }
                }
            }
        }
    }

    int predict(Data data) override {
        valarray<float64> x = {data.x, data.y};

        float64 best_score = -inf;
        int     best_label = -1;
        // Calculate decision function for each label and pick the label with the highest score
        for (int lbl = 0; lbl < n_labels; ++lbl) {
            float64 score = dot(weights[lbl], x) + biases[lbl];
            if (score > best_score) {
                best_score = score;
                best_label = lbl;
            }
        }

        return best_label;
    }

    float64 score(Dataset dataset) override {
        return 0.0; // SVM scoring doesn't apply
    }

    float64 accuracy(Dataset dataset) override {
        uint siz      = dataset.size();
        int  positive = 0;
        for (int i = 0; i < siz; ++i) {
            auto [k, v] = dataset.get(i);
            positive += predict(k) == v;
        }

        if (verbose)
            cout << positive << " out of " << siz << " samples correct" << '\n';

        return 1.0 * positive / siz; // NOLINT(*-narrowing-conversions)
    }

private:
    float64 dot(const valarray<float64> &a, const valarray<float64> &b) {
        return (a * b).sum();
    }

    int find_max_label(Dataset &dataset) {
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


class NeuralNetworkClassifier : public Classifier {
    vector<valarray<float64> > W1; // Weights for input -> hidden layer
    valarray<float64>          b1; // Biases for hidden layer
    vector<valarray<float64> > W2; // Weights for hidden -> output layer
    valarray<float64>          b2; // Biases for output layer
    int                        n_labels;
    int                        n_hidden; // Number of hidden units
    float64                    lr;       // Learning rate
    int                        epochs;   // Number of training epochs

    bool verbose;

public:
    ~NeuralNetworkClassifier() override = default;

    explicit NeuralNetworkClassifier(int n_hidden = 10, float64 lr = 0.001, int epochs = 1000, bool verbose = false)
        : n_hidden(n_hidden), lr(lr), epochs(epochs), n_labels(0), verbose(verbose) {
    }

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
        uint siz      = dataset.size();
        int  positive = 0;
        for (int i = 0; i < siz; ++i) {
            auto [k, v] = dataset.get(i);
            positive += predict(k) == v;
        }
        if (verbose)
            cout << positive << " out of " << siz << " samples correct" << '\n';


        return 1.0 * positive / siz;
    }

private:
    void initialize_weights(int n_features) {
        random_device         rd;
        mt19937               gen(rd());
        normal_distribution<> dist(0, 1);

        // Initialize weights and biases for input -> hidden layer
        W1 = vector(n_hidden, valarray<float64>(n_features));
        for (auto &w: W1)
            generate(begin(w), end(w), [&]() { return dist(gen); });
        b1 = valarray(0.0, n_hidden);

        // Initialize weights and biases for hidden -> output layer
        W2 = vector(n_labels, valarray<float64>(n_hidden));
        for (auto &w: W2)
            generate(begin(w), end(w), [&]() { return dist(gen); });
        b2 = valarray(0.0, n_labels);
    }

    pair<valarray<float64>, valarray<float64> > forward(const valarray<float64> &x) {
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
        dL_dh = 0.0; // Initialize gradient for hidden layer to zero

        // Accumulate gradient contributions from all output units
        for (int j = 0; j < n_labels; ++j) {
            for (int i = 0; i < n_hidden; ++i) {
                dL_dh[i] += dL_do[j] * W2[j][i];
            }
        }

        // Apply tanh derivative to the hidden layer gradient
        for (int i = 0; i < n_hidden; ++i) {
            dL_dh[i] *= (1 - h[i] * h[i]); // tanh derivative
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

    int find_max_label(Dataset &dataset) {
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

    int            siz = 5000;
    valarray<Data> X(siz);
    valarray<int>  y(siz);

    random_device             dev;
    mt19937                   rng(dev());
    uniform_real_distribution real_dist(0.0, 1.0);

    float64 max_radius = 1.5;
    for (int i = 0; i < siz / 3; ++i) {
        float64 radius = real_dist(rng) * max_radius;
        float64 angle  = real_dist(rng) * 2 * PI;

        X[i] = {-1.0 + cos(angle) * radius, 0.0 + sin(angle) * radius};
        y[i] = 0;
    }
    for (int i = siz / 3; i < 2 * siz / 3; ++i) {
        float64 radius = real_dist(rng) * max_radius;
        float64 angle  = real_dist(rng) * 2 * PI;

        X[i] = {1.0 + cos(angle) * radius, 0.0 + sin(angle) * radius};
        y[i] = 1;
    }

    for (int i = 2 * siz / 3; i < siz; ++i) {
        float64 radius = real_dist(rng) * max_radius;
        float64 angle  = real_dist(rng) * 2 * PI;

        X[i] = {0.0 + cos(angle) * radius, 2.0 + sin(angle) * radius};
        y[i] = 2;
    }

    /*cout << "xx,yy,label" << '\n';
    for (int i = 0; i < siz; ++i) {
        cout << X[i].x << "," << X[i].y << "," << y[i] << '\n';
    }*/

    uniform_int_distribution int_dist(0ll, inf);
    for (int i = 0; i < siz; ++i) {
        int idx = int_dist(rng) % (siz - i) + i;
        swap(X[i], X[idx]);
        swap(y[i], y[idx]);
    }

    float64 split     = 0.8;
    size_t  split_idx = siz * split; // NOLINT(*-narrowing-conversions)
    Dataset train_set(X[slice(0, split_idx, 1)],
                      y[slice(0, split_idx, 1)]);

    Dataset test_set(X[slice(split_idx, siz - split_idx, 1)],
                     y[slice(split_idx, siz - split_idx, 1)]);

    // NeuralNetworkClassifier clf(10, 0.001, 1000, true);
    NearestNeighborClassifier clf(5, true);
    clf.fit(train_set);

    cout << "neural network model: \n";
    float64 acc = 100 * clf.accuracy(test_set);
    cout << "Acc: " << acc << '%' << '\n';

    return 0;
}
