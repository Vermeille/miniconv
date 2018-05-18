#include <algorithm>
#include <iostream>
#include <memory>
#include <vector>

template <class T>
class Pool {
   public:
    class Recyclable {
       public:
        Recyclable(std::unique_ptr<T>&& ptr, Pool* pool, int sz)
            : ptr_(std::move(ptr)), from_(pool), sz_(sz) {}
        Recyclable(Recyclable&&) = default;
        Recyclable() = default;
        Recyclable(const Recyclable&) = delete;

        ~Recyclable() {
            if (from_ && ptr_) {
                from_->free(std::move(*this));
            }
        }

        Recyclable& operator=(Recyclable&&) = default;

        auto& operator*() { return *ptr_; }
        const auto& operator*() const { return *ptr_; }

        auto& operator-> () { return ptr_.operator->(); }
        const auto& operator-> () const { return ptr_.operator->(); }

        auto& operator[](int i) { return ptr_[i]; }
        const auto& operator[](int i) const { return ptr_[i]; }

        int sz() const { return sz_; }

        void release_from_pool() { from_ = nullptr; }
        void attach_to_pool(Pool* p) { from_ = p; }

       private:
        std::unique_ptr<T> ptr_;
        Pool* from_;
        int sz_;
    };

    void free(Recyclable&& ptr) {
        ptr.release_from_pool();
        pool_.emplace_back(std::move(ptr));
    }

    Recyclable alloc(int sz) {
        auto found = std::find_if(
            pool_.begin(), pool_.end(), [=](auto& r) { return r.sz() == sz; });

        if (found != pool_.end()) {
            auto handle = std::move(*found);
            pool_.erase(found);
            handle.attach_to_pool(this);
            return std::move(handle);
        }

        return Recyclable(std::make_unique<T>(sz), this, sz);
    }

   private:
    std::vector<Recyclable> pool_;
};

static Pool<float[]> float_pool_;

class Volume {
   public:
    Volume(int w, int h, int c)
        : w_(w), h_(h), c_(c), sz_(h * w * c), res_(float_pool_.alloc(sz_)) {}

    Volume() = default;
    Volume(Volume&& o) = default;

    Volume from_shape() const { return Volume(w_, h_, c_); }
    void zero() {
        for (int i = 0; i < sz_; ++i) {
            res_[i] = 0;
        }
    }

    Volume& operator=(Volume&& o) = default;

    float& operator[](int i) { return res_[i]; }
    float operator[](int i) const { return res_[i]; }

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

   private:
    int w_;
    int h_;
    int c_;
    int sz_;
    Pool<float[]>::Recyclable res_;
};

class Layer {
    virtual Volume& forward(const Volume&) = 0;
    virtual Volume backward(const Volume& grad) = 0;
};

class Relu : public Layer {
   public:
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
            grad[i] += res_[i] > 0 ? pgrad[i] : 0;
        }
        return std::move(grad);
    }

   private:
    Volume res_;
};

class FullyConn : public Layer {
   public:
    FullyConn(int w, int h, int c)
        : w_(w, h, c), w_grad_(w, h, c), b_grad_(1, 1, 1) {}

    virtual Volume& forward(const Volume& input) override {
        x_ = &input;
        float sum = b_[0];
        for (int i = 0; i < input.sz(); ++i) {
            sum += input[i] * w_[i];
        }
        res_ = Volume(1, 1, 1);
        res_[0] = sum;
        return res_;
    }

    virtual Volume backward(const Volume& pgrad) override {
        float d = pgrad[0];
        b_grad_[0] = d;
        for (int i = 0; i < w_grad_.sz(); ++i) {
            w_grad_[i] = d * (*x_)[i];
        }

        Volume x_grad = w_grad_.from_shape();

        for (int i = 0; i < w_grad_.sz(); ++i) {
            x_grad[i] = d * w_[i];
        }
        return x_grad;
    }

   private:
    Volume w_;
    Volume b_;
    Volume w_grad_;
    Volume b_grad_;
    Volume res_;
    const Volume* x_;
};

class MaxPool : public Layer {
   public:
    virtual Volume& forward(const Volume& input) override {
        x_ = &input;
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
        Volume grad = x_->from_shape();
        for (int i = 0; i < res_.sz(); ++i) {
            grad[cache_[i]] = pgrad[i];
        }
        return grad;
    }

   private:
    const Volume* x_;
    std::unique_ptr<int[]> cache_;
    Volume res_;
};

class MSE : public Layer {
   public:
    void set_target(const Volume& target) { target_ = &target; }

    virtual Volume& forward(const Volume& input) override {
        x_ = &input;
        res_ = Volume(1, 1, 1);

        res_[0] = ((*target_)[0] - input[0]) * ((*target_)[0] - input[0]) / 2;

        return res_;
    }

    virtual Volume backward(const Volume& pgrad) override {
        Volume grad = x_->from_shape();
        grad[0] = pgrad[0] * ((*target_)[0] - (*x_)[0]);
        return grad;
    }

   private:
    const Volume* target_;
    const Volume* x_;
    Volume res_;
};

class Conv : public Layer {
   public:
    Conv(int f) : nb_f_(f) {}

    virtual Volume& forward(const Volume& input) override {
        x_ = &input;
        if (filters_.empty()) {
            for (int i = 0; i < nb_f_; ++i) {
                filters_.emplace_back(3, 3, input.c());
                biases_.push_back(0);
            }
        }

        res_ = Volume(input.w(), input.h(), filters_.size());
        res_.zero();

        for (int filter = 0; filter < filters_.size(); ++filter) {
            for (int i = res_.cha_idx(filter), end = res_.cha_idx(filter + 1);
                 i < end;
                 ++i) {
                res_[i] = biases_[filter];
            }
            for (int wptr = 0; wptr < filters_[0].sz(); ++wptr) {
                auto& f = filters_[filter];

                float weight = f[wptr];

                int channel = wptr / (f.w() * f.h());
                int hoffset = (wptr - f.cha_idx(channel)) / f.h() - f.h() / 2;
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
        Volume dx = x_->from_shape();

        for (int i = 0; i < nb_f_; ++i) {
            if (dfilters_.empty()) {
                dfilters_.emplace_back(3, 3, x_->c());
                dbiases_.push_back(0);
            }
            dfilters_[i].zero();
            dbiases_[i] = 0;
        }

        for (int filter = 0; filter < filters_.size(); ++filter) {
            for (int i = pgrad.cha_idx(filter), end = pgrad.cha_idx(filter + 1);
                 i < end;
                 ++i) {
                dbiases_[filter] += pgrad[i];
            }
            for (int wptr = 0; wptr < filters_[0].sz(); ++wptr) {
                auto& f = filters_[filter];
                auto& df = dfilters_[filter];

                float weight = f[wptr];

                int channel = wptr / (f.w() * f.h());
                int hoffset = (wptr - f.cha_idx(channel)) / f.h() - f.h() / 2;
                int woffset = (wptr - f.cha_idx(channel)) % f.w() - f.w() / 2;

                int dst_row_ptr = res_.cha_idx(filter) +
                                  std::max(0, hoffset) * dx.w() +
                                  std::max(0, woffset);
                int src_row_ptr = res_.cha_idx(channel) +
                                  std::max(0, -hoffset) * dx.w() +
                                  std::max(0, -woffset);
                for (int h = 0; h < dx.h() - std::abs(hoffset); ++h) {
                    for (int w = 0; w < dx.w() - std::abs(woffset); ++w) {
                        int dst_ptr = dst_row_ptr + w;
                        int src_ptr = src_row_ptr + w;

                        dx[src_ptr] += pgrad[dst_ptr] * weight;
                        df[wptr] += pgrad[dst_ptr] * (*x_)[src_ptr];
                    }
                    dst_row_ptr += dx.w();
                    src_row_ptr += dx.w();
                }
            }
        }
        return dx;
    }

    void set_filters(std::vector<Volume>&& fs) {
        filters_ = std::move(fs);
        biases_.clear();
        for (int i = 0; i < filters_.size(); ++i) {
            biases_.push_back(0);
        }
    }

   private:
    int nb_f_;
    std::vector<Volume> filters_;
    std::vector<float> biases_;
    std::vector<Volume> dfilters_;
    std::vector<float> dbiases_;
    Volume res_;
    const Volume* x_;
};

int main() {
    Pool<float[]> p;

    Volume v(6, 6, 2);
    for (int i = 0; i < v.sz(); ++i) {
        v[i] = i;
    }
    v.show();

    std::vector<Volume> filters;
    Volume f(3, 3, 2);
    // clang-format off
    f[0] = 0; f[1] = 0; f[2] = 0;
    f[3] = 0; f[4] = -1; f[5] = 0;
    f[6] = 0; f[7] = 0; f[8] = 0;

    f[0 + 9] = 0; f[1 + 9] = 0; f[2 + 9] = 0;
    f[3 + 9] = 0; f[4 + 9] = 1; f[5 + 9] = 0;
    f[6 + 9] = 0; f[7 + 9] = 0; f[8 + 9] = 0;
    // clang-format on

    Conv c(0);
    filters.emplace_back(std::move(f));
    c.set_filters(std::move(filters));
    auto& v2 = c.forward(v);

    /*
    MaxPool mp;
    auto& v2 = mp.forward(v);
    auto grad = mp.backward(v2);
    grad.show();
    */
    v2.show();

    return 0;
}
