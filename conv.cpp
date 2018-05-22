#include <fenv.h>
#include <algorithm>
#include <boost/python/numpy.hpp>
#include <iomanip>
#include <iostream>
#include <list>
#include <memory>
#include <random>
#include <stdexcept>
#include <vector>

float gen_gaussian() {
    static std::random_device r;
    static std::default_random_engine generator(r());
    static std::normal_distribution<float> distribution;
    return distribution(generator);
}

struct Dims {
    int w;
    int h;
    int c;

    int sz() const { return w * h * c; }
};

struct Settings {
    int epochs;
    float lr;
    float l2;
    float grad_max;
    int batch_size;
    float lr_decay;
    Optimizer optimizer;

    Settings()
        : lr(0.1),
          batch_size(8),
          epochs(10),
          l2(0.0001),
          grad_max(5),
          lr_decay(1),
          optimizer(Optimizer::Sgdm) {}
};

template <class T>
class Pool {
   public:
    class PoolAlloc {
        std::unique_ptr<T> x_;
        int sz_;
        int owners_;

       public:
        PoolAlloc(std::unique_ptr<T>&& x, int sz)
            : x_(std::move(x)), sz_(sz), owners_(0) {}

        bool is_free() const { return owners_ == 0; }

        void add_owner() { ++owners_; }
        void remove_owner() {
#if _GLIBCXX_DEBUG
            if (owners_ == 0)
                throw std::runtime_error(
                    "trying to deown something that was said to have zero "
                    "wner");
#endif
            --owners_;
        }

        auto* get() { return x_.get(); }
        const auto* get() const { return x_.get(); }

        int sz() const { return sz_; }
    };

    class Recyclable {
       public:
        Recyclable(PoolAlloc* alloc) : alloc_(alloc) { alloc_->add_owner(); }

        Recyclable(Recyclable&& o) {
            alloc_ = o.alloc_;
            o.alloc_ = nullptr;
        }

        Recyclable() : alloc_(nullptr) {}

        Recyclable(const Recyclable& o) {
            alloc_ = o.alloc_;
            if (alloc_) {
                alloc_->add_owner();
            }
        }

        ~Recyclable() {
            if (alloc_) {
                alloc_->remove_owner();
            }
        }

        Recyclable& operator=(Recyclable&& o) {
            if (alloc_) {
                alloc_->remove_owner();
            }
            alloc_ = o.alloc_;
            o.alloc_ = nullptr;
            return *this;
        }

        Recyclable& operator=(const Recyclable& o) {
            if (alloc_) {
                alloc_->remove_owner();
            }
            alloc_ = o.alloc_;
            if (alloc_) {
                alloc_->add_owner();
            }
            return *this;
        }

        auto& operator*() { return *alloc_->get(); }
        const auto& operator*() const { return *alloc_->get(); }

        auto& operator-> () { return alloc_->get().operator->(); }
        const auto& operator-> () const { return alloc_->get().operator->(); }

        auto& operator[](int i) { return alloc_->get()[i]; }
        const auto& operator[](int i) const { return alloc_->get()[i]; }

        int sz() const { return alloc_->sz_; }

       private:
        PoolAlloc* alloc_;
    };

    Recyclable alloc(int sz) {
        auto found = std::find_if(pool_.begin(), pool_.end(), [=](auto& r) {
            return r.sz() == sz && r.is_free();
        });

        if (found != pool_.end()) {
            PoolAlloc* p = &*found;
            return Recyclable(p);
        }

        pool_.emplace_back(std::make_unique<T>(sz), sz);
        return Recyclable(&pool_.back());
    }

   private:
    std::list<PoolAlloc> pool_;
};

static Pool<float[]> float_pool_;

class Volume {
   public:
    Volume(int w, int h, int c)
        : w_(w), h_(h), c_(c), sz_(h * w * c), res_(float_pool_.alloc(sz_)) {}
    Volume(const Dims& sz) : Volume(sz.w, sz.h, sz.c) {}

    Volume() = default;
    Volume(Volume&& o) = default;

    Volume from_shape() const { return Volume(w_, h_, c_); }
    void zero() {
        for (int i = 0; i < sz_; ++i) {
            res_[i] = 0;
        }
    }

    Volume& operator=(Volume&& o) = default;

    float& operator[](int i) {
#if _GLIBCXX_DEBUG
        if (i >= sz_) {
            throw std::runtime_error("volume bound checks fail");
        }
#endif
        return res_[i];
    }
    float operator[](int i) const {
#if _GLIBCXX_DEBUG
        if (i >= sz_) {
            throw std::runtime_error("volume bound checks fail");
        }
#endif
        return res_[i];
    }

    int w() const { return w_; }
    int h() const { return h_; }
    int c() const { return c_; }
    int sz() const { return sz_; }

    int row_idx(int i) const { return i * w_; };
    int cha_idx(int i) const { return i * w_ * h_; }

    void show() const {
        std::cout << "[";
        for (int c = 0; c < c_; ++c) {
            std::cout << "[\n";
            for (int h = 0; h < h_; ++h) {
                std::cout << "  [";
                std::cout << res_[h_ * c * w_ + h * w_];
                for (int w = 1; w < w_; ++w) {
                    std::cout << ", " << res_[w + w_ * h + w_ * h_ * c];
                }
                std::cout << "],\n";
            }
            std::cout << "]";
        }
        std::cout << "]\n";
    }

    void share_with(Volume& v) const {
        v.res_ = res_;
        v.w_ = w_;
        v.h_ = h_;
        v.c_ = c_;
        v.sz_ = sz_;
    }

    void gaussian() {
        for (int i = 0; i < sz_; ++i) {
            res_[i] = gen_gaussian();
        }
    }

    std::pair<float, float> mean_std() const {
        float m = 0;
        for (int i = 0; i < sz_; ++i) {
            m += res_[i];
        }
        m = m / sz_;

        float s = 0;
        for (int i = 0; i < sz_; ++i) {
            s += (res_[i] - m) * (res_[i] - m);
        }
        s = std::sqrt(s / (sz_ - 1));
        return std::make_pair(m, s);
    }

    float min() const {
        float m = res_[0];
        for (int i = 1; i < sz_; ++i) {
            m = m < res_[i] ? m : res_[i];
        }
        return m;
    }

    float max() const {
        float m = res_[0];
        for (int i = 1; i < sz_; ++i) {
            m = m > res_[i] ? m : res_[i];
        }
        return m;
    }

    int nonzero() const {
        int count = 0;
        for (int i = 1; i < sz_; ++i) {
            count += res_[i] == 0 ? 0 : 1;
        }
        return count;
    }

    float norm() const {
        float len = 0;
        for (int i = 0; i < sz_; ++i) {
            len += res_[i] * res_[i];
        }
        len = std::sqrt(len + 1e-6);
        return len;
    }

   private:
    int w_;
    int h_;
    int c_;
    int sz_;
    Pool<float[]>::Recyclable res_;
};

class Param {
   public:
    Param() = default;
    Param(Param&&) = default;
    Param(Dims d) : Param(d.w, d.h, d.c) {}
    Param(int w, int h, int c) : val_(w, h, c), grad_(w, h, c), mem_(w, h, c) {
        val_.gaussian();
        grad_.zero();
        mem_.zero();
    }

    float& operator[](int i) { return val_[i]; }
    float operator[](int i) const { return val_[i]; }

    void reset_grad() { grad_.zero(); }
    Volume& grad() { return grad_; }
    const Volume& grad() const { return grad_; }

    int sz() const { return val_.sz(); }
    Volume& vol() { return val_; }
    const Volume& vol() const { return val_; }

    Param& operator=(Param&& o) = default;

    void reinit(Volume& v) {
        v.share_with(val_);
        grad_ = v.from_shape();
        grad_.zero();
        mem_ = v.from_shape();
        mem_.zero();
    }

    int w() const { return val_.w(); }
    int h() const { return val_.h(); }
    int c() const { return val_.c(); }
    int cha_idx(int c) const { return val_.cha_idx(c); }

    void descend(const Settings& s) {
        switch (s.optimizer) {
            case Optimizer::Sgdm:
                sgdm(s);
                break;
            case Optimizer::Sgd:
                sgd(s);
                break;
            case Optimizer::PowerSign:
                power_sign(s);
                break;
        }
    }

   private:
    void sgd(const Settings& s) {
        float len = grad_.norm();
        for (int i = 0; i < val_.sz(); ++i) {
            val_[i] = val_[i] - s.lr * grad_[i] - s.l2 * 2 * val_[i];
        }
        grad_.zero();
    }

    void sgdm(const Settings& s) {
        float len = grad_.norm();
        for (int i = 0; i < val_.sz(); ++i) {
            grad_[i] =
                len > s.grad_max ? grad_[i] / len * s.grad_max : grad_[i];
            mem_[i] = 0.9 * mem_[i] + 0.1 * grad_[i];
            val_[i] = val_[i] - s.lr * mem_[i] - s.l2 * 2 * val_[i];
        }
        grad_.zero();
    }

    void power_sign(const Settings& s) {
        float len = grad_.norm();
        for (int i = 0; i < val_.sz(); ++i) {
            grad_[i] =
                len > s.grad_max ? grad_[i] / len * s.grad_max : grad_[i];
            mem_[i] = 0.9 * mem_[i] + 0.1 * grad_[i];
            val_[i] -= s.lr * exp(sgn(grad_[i]) * sgn(mem_[i])) * grad_[i];
        }
        grad_.zero();
    }

    Volume val_;
    Volume grad_;
    Volume mem_;
};

class Layer {
   public:
    virtual Volume& forward(const Volume&) = 0;
    virtual Volume backward(const Volume& grad) = 0;
    virtual void update(const Settings&) {}
    virtual Dims out_shape() const = 0;
    virtual void set_train_mode(bool train) {}
};

class Verbose : public Layer {
   public:
    Verbose(Dims d, std::string name)
        : out_shape_(d), name_(name), train_(false) {}

    virtual Volume& forward(const Volume& x) override {
        if (train_) {
            fwd_.emplace_back();
            x.share_with(fwd_.back());
            return fwd_.back();
        } else {
            x.share_with(test_thru_);
            return test_thru_;
        }
    }

    virtual Volume backward(const Volume& x) override {
        bwd_.emplace_back();
        x.share_with(bwd_.back());
        Volume v;
        bwd_.back().share_with(v);
        return v;
    }

    virtual void update(const Settings&) override {
        float statsfwd[] = {0, 0, 0, 0, 0};
        for (auto& v : fwd_) {
            auto local = v.mean_std();
            statsfwd[0] += local.first;
            statsfwd[1] += local.second;
            statsfwd[2] += v.max();
            statsfwd[3] += v.min();
            statsfwd[4] += float(v.nonzero()) / v.sz();
        }
        float statsbwd[] = {0, 0, 0, 0, 0};
        for (auto& v : bwd_) {
            auto local = v.mean_std();
            statsbwd[0] += local.first;
            statsbwd[1] += local.second;
            statsbwd[2] += v.max();
            statsbwd[3] += v.min();
            statsbwd[4] += float(v.nonzero()) / v.sz();
        }

        std::cout << "               Statistics for " << name_ << ":\n";
        std::cout << "  Batch size: " << fwd_.size() << "\n";
        std::cout << "  Forward pass:                       Backward Pass:\n";
        std::cout << "    Shape: (" << std::setw(3) << fwd_[0].w() << ", "
                  << std::setw(3) << fwd_[0].h() << ", " << std::setw(3)
                  << fwd_[0].c() << ")            "
                  << "  Shape: (" << std::setw(3) << bwd_[0].w() << ", "
                  << std::setw(3) << bwd_[0].h() << ", " << std::setw(3)
                  << bwd_[0].c() << ")\n";

        std::cout << "    Mean:     " << std::setw(8)
                  << statsfwd[0] / fwd_.size()
                  << "                  Mean:     " << statsbwd[0] / bwd_.size()
                  << "\n";
        std::cout << "    Std:      " << std::setw(8)
                  << statsfwd[1] / fwd_.size()
                  << "                  Std:      " << statsbwd[1] / bwd_.size()
                  << "\n";
        std::cout << "    Max:      " << std::setw(8)
                  << statsfwd[2] / fwd_.size()
                  << "                  Max:      " << statsbwd[2] / bwd_.size()
                  << "\n";
        std::cout << "    Min:      " << std::setw(8)
                  << statsfwd[3] / fwd_.size()
                  << "                  Min:      " << statsbwd[3] / bwd_.size()
                  << "\n";
        std::cout << "    Non-Zero: " << std::setw(8)
                  << statsfwd[4] / fwd_.size()
                  << "                  Non-Zero: " << statsbwd[4] / fwd_.size()
                  << "\n";

        fwd_.clear();
        bwd_.clear();
    }

    virtual Dims out_shape() const override { return out_shape_; }
    virtual void set_train_mode(bool b) override { train_ = b; }

   private:
    std::vector<Volume> fwd_;
    std::vector<Volume> bwd_;
    Dims out_shape_;
    std::string name_;
    Volume test_thru_;
    bool train_;
};

class Input : public Layer {
   public:
    Input(Dims d) : out_shape_(d) {}
    virtual Volume& forward(const Volume& x) override {
        x.share_with(out_);
        return out_;
    }
    virtual Volume backward(const Volume& grad) override {
        Volume out;
        grad.share_with(out);
        return out;
    }
    virtual void update(const Settings&) {}
    virtual Dims out_shape() const { return out_shape_; }
    virtual void set_train_mode(bool train) override {}

   private:
    Dims out_shape_;
    Volume out_;
};

class Relu : public Layer {
   public:
    Relu() = default;
    Relu(Dims d) : out_shape_(d){};
    virtual Volume& forward(const Volume& input) override {
        auto res = input.from_shape();
        for (int i = 0; i < input.sz(); ++i) {
            float x = input[i];
            res[i] = x < 0 ? 0 : x;
        }
        res_ = std::move(res);
        return res_;
    }

    virtual Volume backward(const Volume& pgrad) override {
        Volume grad = pgrad.from_shape();
        for (int i = 0; i < grad.sz(); ++i) {
            grad[i] = res_[i] > 0 ? pgrad[i] : 0;
        }
        return std::move(grad);
    }

    virtual void update(const Settings&) override {}
    virtual void set_train_mode(bool train) override {}
    virtual Dims out_shape() const { return out_shape_; }

   private:
    Volume res_;
    Dims out_shape_;
};

class FullyConn : public Layer {
   public:
    FullyConn() = default;
    FullyConn(Dims in_sz, int c) : w_(in_sz.sz(), c, 1), b_(1, c, 1) {
        b_.vol().zero();
    }

    virtual Volume& forward(const Volume& input) override {
        input.share_with(x_);
        res_ = Volume(b_.sz(), 1, 1);

        for (int c = 0, wptr = 0; c < b_.sz(); ++c, wptr += input.sz()) {
            float sum = b_[c];
            for (int i = 0; i < input.sz(); ++i) {
                sum += input[i] * w_[wptr + i];
            }
            res_[c] = sum;
        }
        return res_;
    }

    virtual Volume backward(const Volume& pgrad) override {
        Volume x_grad = x_.from_shape();
        x_grad.zero();

        for (int c = 0, wptr = 0; c < b_.sz(); ++c, wptr += x_.sz()) {
            float d = pgrad[c];
            auto& dw = w_.grad();

            b_.grad()[c] += d;
            for (int i = 0; i < x_.sz(); ++i) {
                dw[wptr + i] += d * x_[i];
            }

            for (int i = 0; i < x_.sz(); ++i) {
                x_grad[i] += d * w_[wptr + i];
            }
        }
        return x_grad;
    }

    virtual void update(const Settings& s) override {
        w_.descend(s);
        b_.descend(s);
    }

    void set_weights(std::vector<Volume>&& w, Volume b) {
        Volume v(w[0].sz(), w.size(), 1);
        for (int i = 0; i < w.size(); ++i) {
            for (int j = 0; j < w[i].sz(); ++j) {
                v[v.w() * i + j] = w[i][j];
            }
        }
        w_.reinit(v);
        b_.reinit(b);
    }

    std::vector<Volume> grads() const {
        std::vector<Volume> gs;
        for (int i = 0; i < b_.sz(); ++i) {
            gs.emplace_back(x_.w(), x_.h(), x_.c());
            for (int j = 0; j < w_.w(); ++j) {
                gs.back()[j] = w_.grad()[w_.w() * i + j];
            }
        }
        return gs;
    }
    float bias() const { return b_.vol()[0]; }
    std::vector<Volume> weights() const {
        std::vector<Volume> gs;
        for (int i = 0; i < b_.sz(); ++i) {
            gs.emplace_back(x_.w(), x_.h(), x_.c());
            for (int j = 0; j < w_.w(); ++j) {
                gs.back()[j] = w_[w_.w() * i + j];
            }
        }
        return gs;
    }

    virtual void set_train_mode(bool train) override {}
    virtual Dims out_shape() const override { return Dims{b_.h(), 1, 1}; }

   private:
    Param w_;
    Param b_;
    Volume res_;
    Volume x_;
};

class MaxPool : public Layer {
   public:
    MaxPool(Dims d) : out_shape_(Dims{d.w / 2, d.h / 2, d.c}) {}

    virtual Volume& forward(const Volume& input) override {
        input.share_with(x_);
        res_ = Volume(input.w() / 2, input.h() / 2, input.c());
        if (!cache_) {
            cache_ = std::make_unique<int[]>(res_.sz());
            std::fill(cache_.get(), cache_.get() + res_.sz(), 0);
        }

        for (int c = 0; c < input.c(); ++c) {
            int src_row_idx = input.cha_idx(c);
            int dst_row_idx = res_.cha_idx(c);
            for (int h = 0; h < res_.h(); ++h) {
                for (int w = 0; w < res_.w(); ++w) {
                    int dst_idx = dst_row_idx + w;
                    int src_idx = src_row_idx + 2 * w;
                    int max_pos = std::max(
                        {src_idx,
                         src_idx + 1,
                         src_idx + input.w(),
                         src_idx + input.w() + 1},
                        [&](int a, int b) { return input[a] < input[b]; });
                    cache_[dst_idx] = max_pos;
                    res_[dst_idx] = input[max_pos];
                }
                src_row_idx += 2 * input.w();
                dst_row_idx += res_.w();
            }
        }

        return res_;
    }

    virtual Volume backward(const Volume& pgrad) override {
        Volume grad = x_.from_shape();
        grad.zero();
        for (int i = 0; i < res_.sz(); ++i) {
            grad[cache_[i]] = pgrad[i];
        }
        return grad;
    }

    virtual void set_train_mode(bool train) override {}
    virtual void update(const Settings&) override {}

    virtual Dims out_shape() const override { return out_shape_; }

   private:
    Volume x_;
    std::unique_ptr<int[]> cache_;
    Volume res_;
    Dims out_shape_;
};

class MSE : public Layer {
   public:
    void set_target(const Volume& target) { target.share_with(target_); }

    virtual Volume& forward(const Volume& input) override {
        input.share_with(x_);
        res_ = Volume(1, 1, 1);

        float total = 0;

        for (int i = 0; i < input.sz(); ++i) {
            float err = input[i] - target_[i];
            total += err * err;
        }

        res_[0] = total / (2 * input.sz());

        return res_;
    }

    virtual Volume backward(const Volume& pgrad) override {
        Volume grad = x_.from_shape();
        for (int i = 0; i < grad.sz(); ++i) {
            grad[i] = pgrad[0] * (x_[i] - target_[i]);
        }
        return grad;
    }

    virtual void set_train_mode(bool train) override {}
    virtual void update(const Settings&) override {}
    virtual Dims out_shape() const override { return Dims{1, 1, 1}; }

   private:
    Volume target_;
    Volume x_;
    Volume res_;
};

class Conv : public Layer {
   public:
    Conv() = default;
    Conv(Dims in, Dims kerns) : in_sz_(in), kern_sz_(kerns) {
        for (int i = 0; i < kern_sz_.c; ++i) {
            filters_.emplace_back(kern_sz_.w, kern_sz_.h, in.c);
            biases_.emplace_back(1, 1, 1);
            biases_.back().vol().zero();
        }
    }

    virtual Volume& forward(const Volume& input) override {
        input.share_with(x_);
        res_ = Volume(input.w(), input.h(), filters_.size());

        for (int filter = 0; filter < int(filters_.size()); ++filter) {
            for (int i = res_.cha_idx(filter), end = res_.cha_idx(filter + 1);
                 i < end;
                 ++i) {
                res_[i] = biases_[filter][0];
            }
            for (int wptr = 0; wptr < filters_[0].sz(); ++wptr) {
                auto& f = filters_[filter].vol();

                float weight = f[wptr];

                int channel = wptr / (f.w() * f.h());
                int hoffset = (wptr - f.cha_idx(channel)) / f.w() - f.h() / 2;
                int woffset = (wptr - f.cha_idx(channel)) % f.w() - f.w() / 2;

                int dst_row_ptr = res_.cha_idx(filter) +
                                  std::max(0, hoffset) * res_.w() +
                                  std::max(0, woffset);
                int src_row_ptr = input.cha_idx(channel) +
                                  std::max(0, -hoffset) * input.w() +
                                  std::max(0, -woffset);
                for (int h = 0; h < input.h() - std::abs(hoffset); ++h) {
                    for (int w = 0; w < input.w() - std::abs(woffset); ++w) {
                        int dst_ptr = dst_row_ptr + w;
                        int src_ptr = src_row_ptr + w;

                        res_[dst_ptr] += weight * input[src_ptr];
                    }
                    dst_row_ptr += res_.w();
                    src_row_ptr += input.w();
                }
            }
        }

        return res_;
    }

    virtual Volume backward(const Volume& pgrad) override {
        Volume dx = x_.from_shape();
        dx.zero();

        for (int filter = 0; filter < int(filters_.size()); ++filter) {
            for (int i = pgrad.cha_idx(filter), end = pgrad.cha_idx(filter + 1);
                 i < end;
                 ++i) {
                biases_[filter].grad()[0] += pgrad[i];
            }
            auto& f = filters_[filter].vol();
            auto& df = filters_[filter].grad();
            for (int wptr = 0; wptr < filters_[0].sz(); ++wptr) {
                float weight = f[wptr];

                int channel = wptr / (f.w() * f.h());
                int hoffset = (wptr - f.cha_idx(channel)) / f.w() - f.h() / 2;
                int woffset = (wptr - f.cha_idx(channel)) % f.w() - f.w() / 2;

                int dst_row_ptr = res_.cha_idx(filter) +
                                  std::max(0, -hoffset) * res_.w() +
                                  std::max(0, -woffset);
                int src_row_ptr = dx.cha_idx(channel) +
                                  std::max(0, hoffset) * dx.w() +
                                  std::max(0, woffset);
                for (int h = 0; h < dx.h() - std::abs(hoffset); ++h) {
                    for (int w = 0; w < dx.w() - std::abs(woffset); ++w) {
                        int dst_ptr = dst_row_ptr + w;
                        int src_ptr = src_row_ptr + w;

                        dx[src_ptr] += pgrad[dst_ptr] * weight;
                        df[wptr] += pgrad[dst_ptr] * x_[src_ptr];
                    }
                    dst_row_ptr += dx.w();
                    src_row_ptr += dx.w();
                }
            }
        }
        return dx;
    }

    void set_filters(std::vector<Volume>&& fs, std::vector<float>&& bs) {
        if (fs.size() != bs.size()) {
            throw std::runtime_error("filters and biases of a different size");
        }

        filters_.clear();
        for (auto& v : fs) {
            filters_.emplace_back();
            filters_.back().reinit(v);
        }

        biases_.clear();
        for (auto& v : bs) {
            biases_.emplace_back(1, 1, 1);
            biases_.back()[0] = v;
        }
    }

    std::vector<Volume> filters_grad() const {
        std::vector<Volume> vs;
        for (auto& f : filters_) {
            vs.emplace_back();
            f.grad().share_with(vs.back());
        }
        return vs;
    }

    virtual void update(const Settings& s) override {
        for (int i = 0; i < int(filters_.size()); ++i) {
            filters_[i].descend(s);
            biases_[i].descend(s);
        }
    }

    virtual Dims out_shape() const override {
        return Dims{in_sz_.w, in_sz_.h, kern_sz_.c};
    }
    virtual void set_train_mode(bool train) override {}

   private:
    int nb_f_;
    std::vector<Param> filters_;
    std::vector<Param> biases_;
    Volume res_;
    Volume x_;
    Dims in_sz_;
    Dims kern_sz_;
};

class Net {
   public:
    void set_lr(float x) { settings_.lr = x; }
    void set_batch_size(int x) { settings_.batch_size = x; }
    void set_epochs(int e) { settings_.epochs = e; }
    void set_l2(float x) { settings_.l2 = x; }
    void set_lr_decay(float x) { settings_.lr_decay = x; }
    void set_grad_max(float x) { settings_.grad_max = x; }
    void set_optimizer(Optimizer o) { settings_.optimizer = o; }
    void set_train_mode(bool train) {
        for (auto& l : layers_) {
            l->set_train_mode(train);
        }
    }

    void train(std::vector<Volume>&& xs, std::vector<Volume>&& ys) {
        xs_ = std::move(xs);
        ys_ = std::move(ys);

        set_train_mode(true);
        std::vector<float> errs(xs_.size() / settings_.batch_size + 1);
        for (int e = 0; e < settings_.epochs; ++e) {
            settings_.lr *= settings_.lr_decay;
            std::cout << "Epoch " << e << "\n";
            for (int i = 0; i < int(xs_.size()); i += settings_.batch_size) {
                errs.clear();
                for (int j = i;
                     j < std::min(int(xs_.size()), i + settings_.batch_size);
                     ++j) {
                    mse_.set_target(ys_[j]);
                    errs.push_back(forward(xs_[j])[0]);
                    backward();
                }
                update();
                float total = 0;
                for (float x : errs) {
                    total += x;
                }
                total /= errs.size();
                std::cout << "loss: " << total << "\n";
            }
        }
    }

    void conv(int kw, int kh, int nb) {
        if (layers_.empty()) {
            throw std::invalid_argument("input() must be the first layer");
        }
        auto c = std::make_unique<Conv>(layers_.back()->out_shape(),
                                        Dims{kw, kh, nb});
        layers_.emplace_back(std::move(c));
    }

    void input(int w, int h, int c) {
        if (!layers_.empty()) {
            throw std::invalid_argument("input() must be the first layer");
        }
        layers_.emplace_back(std::make_unique<Input>(Dims{w, h, c}));
    }

    void relu() {
        layers_.emplace_back(
            std::make_unique<Relu>(layers_.back()->out_shape()));
    }

    void maxpool() {
        layers_.emplace_back(
            std::make_unique<MaxPool>(layers_.back()->out_shape()));
    }

    void fc(int out) {
        layers_.emplace_back(
            std::make_unique<FullyConn>(layers_.back()->out_shape(), out));
    }

    void verbose(std::string name) {
        layers_.emplace_back(
            std::make_unique<Verbose>(layers_.back()->out_shape(), name));
    }

    const Volume predict(const Volume& x) {
        set_train_mode(false);
        Volume it;
        x.share_with(it);
        for (int i = 0; i < int(layers_.size()); ++i) {
            layers_[i]->forward(it).share_with(it);
        }
        return std::move(it);
    }
    const Volume& forward(const Volume& x) {
        Volume it;
        x.share_with(it);
        for (int i = 0; i < int(layers_.size()); ++i) {
            layers_[i]->forward(it).share_with(it);
        }
        return mse_.forward(it);
    }

    Volume backward() {
        Volume one(1, 1, 1);
        one[0] = 1;
        Volume it = mse_.backward(one);
        for (int i = layers_.size(); i-- > 0;) {
            layers_[i]->backward(it).share_with(it);
        }
        return it;
    }

    void update() {
        for (auto& l : layers_) {
            l->update(settings_);
        }
    }

   private:
    std::vector<std::unique_ptr<Layer>> layers_;
    std::vector<Volume> xs_;
    std::vector<Volume> ys_;
    MSE mse_;
    Settings settings_;
};

namespace p = boost::python;
namespace np = boost::python::numpy;

Volume from_array(np::ndarray arr) {
    if (arr.get_nd() != 3) {
        throw std::runtime_error("volume doesn't have 3 dimensions");
    }
    auto as_f = arr.astype(np::dtype::get_builtin<float>());
    Volume v(arr.shape(1), arr.shape(0), arr.shape(2));
    const int channel_size = v.cha_idx(1);
    const int img_size = v.w() * v.h();

    for (int dst = 0, src = 0; dst < img_size; ++dst, src += arr.shape(2)) {
        int channel_dst = dst;
        for (int j = 0; j < arr.shape(2); ++j) {
            float x = reinterpret_cast<float*>(as_f.get_data())[src + j];
            v[channel_dst] = x;
            channel_dst += channel_size;
        }
    }
    return v;
}

np::ndarray to_array(const Volume& v) {
    auto res = np::empty(p::make_tuple(v.h(), v.w(), v.c()),
                         np::dtype::get_builtin<float>());
    float* dat = reinterpret_cast<float*>(res.get_data());

    int img_sz = v.w() * v.h();
    for (int src = 0, dst = 0; src < img_sz; ++src, dst += v.c()) {
        for (int c = 0, src_c = 0; c < v.c(); ++c, src_c += v.cha_idx(1)) {
            dat[dst + c] = v[src + src_c];
        }
    }

    return res;
}

BOOST_PYTHON_MODULE(miniconv) {
    using namespace boost::python;
    class_<Conv, boost::noncopyable>("Conv")
        .def("forward",
             +[](Conv* conv, np::ndarray arr) {
                 return to_array(conv->forward(from_array(arr)));
             })
        .def("backward",
             +[](Conv* conv, np::ndarray arr) {
                 return to_array(conv->backward(from_array(arr)));
             })
        .def("set_filters",
             +[](Conv* conv, p::list kerns, p::list biases) {
                 std::vector<Volume> ks;
                 std::vector<float> bs;
                 for (int i = 0; i < p::len(kerns); ++i) {
                     ks.emplace_back(
                         from_array(p::extract<np::ndarray>(kerns[i])));
                     bs.push_back(p::extract<float>(biases[i]));
                 }
                 conv->set_filters(std::move(ks), std::move(bs));
             })
        .def("filters_grad",
             +[](Conv* conv) {
                 p::list res;
                 for (auto& vol : conv->filters_grad()) {
                     res.append(to_array(vol));
                 }
                 return res;
             })
        .def("update", &Conv::update);

    class_<MSE, boost::noncopyable>("MSE")
        .def("forward",
             +[](MSE* mse, np::ndarray arr) {
                 return to_array(mse->forward(from_array(arr)));
             })
        .def("backward",
             +[](MSE* mse, np::ndarray arr) {
                 return to_array(mse->backward(from_array(arr)));
             })
        .def("set_target",
             +[](MSE* mse, np::ndarray arr) {
                 return mse->set_target(from_array(arr));
             })
        .def("update", &MSE::update);

    class_<Relu, boost::noncopyable>("Relu")
        .def("forward",
             +[](Relu* relu, np::ndarray arr) {
                 return to_array(relu->forward(from_array(arr)));
             })
        .def("backward",
             +[](Relu* relu, np::ndarray arr) {
                 return to_array(relu->backward(from_array(arr)));
             })
        .def("update", &Relu::update);

    class_<FullyConn, boost::noncopyable>("FullyConn")
        .def("forward",
             +[](FullyConn* relu, np::ndarray arr) {
                 return to_array(relu->forward(from_array(arr)));
             })
        .def("backward",
             +[](FullyConn* fc, np::ndarray arr) {
                 return to_array(fc->backward(from_array(arr)));
             })
        .def("set_weights",
             +[](FullyConn* fc, p::list weights, np::ndarray b) {
                 std::vector<Volume> ws;
                 for (int i = 0; i < p::len(weights); ++i) {
                     ws.emplace_back(
                         from_array(p::extract<np::ndarray>(weights[i])));
                 }
                 fc->set_weights(std::move(ws), from_array(b));
             })
        .def("grads",
             +[](FullyConn* fc) {
                 p::list ws;
                 auto ws2 = fc->grads();
                 for (auto& w : ws2) {
                     ws.append(to_array(w));
                 }
                 return ws;
             })
        .def("bias", +[](FullyConn* fc) { return fc->bias(); })
        .def("weights",
             +[](FullyConn* fc) {
                 p::list ws;
                 auto ws2 = fc->weights();
                 for (auto& w : ws2) {
                     ws.append(to_array(w));
                 }
                 return ws;
             })
        .def("update", &FullyConn::update);
    class_<Net, boost::noncopyable>("Net")
        .def("conv", &Net::conv)
        .def("fc", &Net::fc)
        .def("backward", +[](Net* net) { return to_array(net->backward()); })
        .def("forward",
             +[](Net* net, np::ndarray arr) {
                 return to_array(net->forward(from_array(arr)));
             })
        .def("predict",
             +[](Net* net, np::ndarray arr) {
                 return to_array(net->predict(from_array(arr)));
             })
        .def("input", &Net::input)
        .def("maxpool", &Net::maxpool)
        .def("relu", &Net::relu)
        .def("verbose", &Net::verbose)
        .def("set_batch_size", &Net::set_batch_size)
        .def("set_l2", &Net::set_l2)
        .def("set_epochs", &Net::set_epochs)
        .def("set_lr", &Net::set_lr)
        .def("set_lr_decay", &Net::set_lr_decay)
        .def("set_optimizer", &Net::set_optimizer)
        .def("set_grad_max", &Net::set_grad_max)
        .def("set_train_mode", &Net::set_train_mode)
        .def("train",
             +[](Net* net, p::list xs, p::list ys) {
                 std::vector<Volume> vxs;
                 std::vector<Volume> vys;
                 if (p::len(xs) != p::len(ys)) {
                     throw std::invalid_argument(
                         "ys and xs must be of equal size");
                 }
                 for (int i = 0; i < p::len(xs); ++i) {
                     vxs.emplace_back(
                         from_array(p::extract<np::ndarray>(xs[i])));
                     vys.emplace_back(
                         from_array(p::extract<np::ndarray>(ys[i])));
                 }
                 net->train(std::move(vxs), std::move(vys));
             })
        .def("update", &Net::update);

    class_<Settings>("Settings")
        .def_readwrite("lr", &Settings::lr)
        .def_readwrite("lr_decay", &Settings::lr_decay)
        .def_readwrite("l2", &Settings::l2)
        .def_readwrite("epochs", &Settings::epochs)
        .def_readwrite("grad_max", &Settings::grad_max)
        .def_readwrite("batch_size", &Settings::batch_size);

    enum_<Optimizer>("Optimizer")
        .value("sgd", Optimizer::Sgd)
        .value("sgdm", Optimizer::Sgdm)
        .value("powersign", Optimizer::PowerSign);

    np::initialize();
    //    feenableexcept(FE_ALL_EXCEPT & ~FE_INEXACT);
}
