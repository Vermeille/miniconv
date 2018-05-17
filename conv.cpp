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
            if (from_) {
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
            std::cout << "[";
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
    virtual Volume& forward(const Volume& input) override {}
    virtual Volume backward(const Volume& pgrad) override {}
};

int main() {
    Pool<float[]> p;

    Volume v(6, 6, 2);
    for (int i = 0; i < v.sz(); ++i) {
        v[i] = (1 + i) % 5;
    }
    v.show();
    MaxPool mp;
    auto& v2 = mp.forward(v);
    auto grad = mp.backward(v2);
    v2.show();
    grad.show();

    return 0;
}
