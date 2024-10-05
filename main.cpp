#include <cassert>
#include <random>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <cmath>
#include <map>
#include <queue>
#include <valarray>

using namespace std;

#define int long long
#define uint unsigned long long
#define float64 double


constexpr int inf = 0x3f3f3f3f;

class Data {
public:
    Data(float64 x, float64 y)
        : x(x), y(y) {
    }

    friend float64 euclid(const Data &lhs, const Data &rhs) {
        return sqrt((lhs.x - rhs.x) * (lhs.x - rhs.x) + (lhs.y - rhs.y) * (lhs.y - rhs.y));
    }

    Data() = default;

private:
    float64 x, y;
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

class NearestNeighborClassifier : public Classifier {
    int k;
    Dataset dataset;
    bool verbose;

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
        uint siz = dataset.size();
        int positive = 0;
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
public:
    ~LinearSVMClassifier() override = default;

    explicit LinearSVMClassifier() = default;

    void fit(Dataset dataset) override {
    }

    int predict(Data data) override;

    float64 score(Dataset dataset) override;

    float64 accuracy(Dataset dataset) override;
};


signed main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);
    cout.tie(nullptr);

    int siz = 2500;
    valarray<Data> X(siz);
    valarray<int> y(siz);

    random_device dev;
    mt19937 rng(dev());
    uniform_real_distribution real_dist(0.0, 1.0);

    float64 radius = 2.5;
    for (int i = 0; i < siz / 3; ++i) {
        X[i] = {-1.0 + real_dist(rng) * radius, 0.0 + real_dist(rng) * radius};
        y[i] = 0;
    }
    for (int i = siz / 3; i < siz; ++i) {
        X[i] = {1.0 + real_dist(rng) * radius, 0.0 + real_dist(rng) * radius};
        y[i] = 1;
    }

    uniform_int_distribution int_dist(0ll, inf);
    for (int i = 0; i < siz; ++i) {
        int idx = int_dist(rng) % (siz - i) + i;
        swap(X[i], X[idx]);
        swap(y[i], y[idx]);
    }

    float64 split = 0.8;
    size_t split_idx = siz * split; // NOLINT(*-narrowing-conversions)
    Dataset train_set(X[slice(0, split_idx, 1)],
                      y[slice(0, split_idx, 1)]);

    Dataset test_set(X[slice(split_idx, siz - split_idx, 1)],
                     y[slice(split_idx, siz - split_idx, 1)]);

    NearestNeighborClassifier clf(5, true);
    clf.fit(train_set);

    float64 acc = 100 * clf.accuracy(test_set);
    cout << "Acc: " << acc << '%' << '\n';

    return 0;
}
