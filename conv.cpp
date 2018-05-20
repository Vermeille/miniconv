#include <algorithm>
#include <boost/python/numpy.hpp>
#include <iostream>
#include <list>
#include <memory>
#include <stdexcept>
#include <vector>

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

    void update_with(float lr, const Volume& o) {
#if _GLIBCXX_DEBUG
        if (sz_ != o.sz_) {
            throw std::runtime_error(
                "trying to update a matrix with a different size");
        }
#endif
        for (int i = 0; i < sz_; ++i) {
            res_[i] -= lr * o.res_[i];
        }
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
    virtual void update(float lr) = 0;
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

    virtual void update(float lr) override {}

   private:
    Volume res_;
};

class FullyConn : public Layer {
   public:
    virtual Volume& forward(const Volume& input) override {
        input.share_with(x_);

        if (w_.sz() == 0) {
            w_ = input.from_shape();
            b_ = 0;
        }

        float sum = b_;
        for (int i = 0; i < input.sz(); ++i) {
            sum += input[i] * w_[i];
        }
        res_ = Volume(1, 1, 1);
        res_[0] = sum;
        return res_;
    }

    virtual Volume backward(const Volume& pgrad) override {
        float d = pgrad[0];
        b_grad_ += d;
        for (int i = 0; i < w_grad_.sz(); ++i) {
            w_grad_[i] += d * x_[i];
        }

        Volume x_grad = w_grad_.from_shape();

        for (int i = 0; i < w_grad_.sz(); ++i) {
            x_grad[i] = d * w_[i];
        }
        return x_grad;
    }

    virtual void update(float lr) override {
        w_.update_with(lr, w_grad_);
        b_ -= lr * b_grad_;

        w_grad_.zero();
        b_grad_ = 0;
    }

    void set_weights(Volume w, float b) {
        w_ = std::move(w);
        b_ = b;

        w_grad_ = w_.from_shape();
        w_grad_.zero();
        b_grad_ = 0;
    }

    const Volume& grads() const { return w_grad_; }
    float bias() const { return b_; }
    const Volume& weights() const { return w_; }

   private:
    Volume w_;
    float b_;
    Volume w_grad_;
    float b_grad_;
    Volume res_;
    Volume x_;
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

    virtual void update(float lr) override {}

   private:
    const Volume* x_;
    std::unique_ptr<int[]> cache_;
    Volume res_;
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

    virtual void update(float lr) override {}

   private:
    Volume target_;
    Volume x_;
    Volume res_;
};

class Conv : public Layer {
   public:
    Conv(int f) : nb_f_(f) {}

    virtual Volume& forward(const Volume& input) override {
        input.share_with(x_);
        if (filters_.empty()) {
            for (int i = 0; i < nb_f_; ++i) {
                filters_.emplace_back(3, 3, input.c());
                biases_.push_back(0);
            }
        }

        res_ = Volume(input.w(), input.h(), filters_.size());

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

        if (dfilters_.empty()) {
            for (int i = 0; i < filters_.size(); ++i) {
                dfilters_.emplace_back(
                    filters_[i].w(), filters_[i].h(), filters_[i].c());
                dbiases_.push_back(0);
            }
        }
        for (int filter = 0; filter < filters_.size(); ++filter) {
            for (int i = pgrad.cha_idx(filter), end = pgrad.cha_idx(filter + 1);
                 i < end;
                 ++i) {
                dbiases_[filter] += pgrad[i];
            }
            auto& f = filters_[filter];
            auto& df = dfilters_[filter];
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
        filters_ = std::move(fs);
        biases_ = std::move(bs);
    }

    const std::vector<Volume>& filters_grad() const { return dfilters_; }

    virtual void update(float lr) override {
        for (int i = 0; i < filters_.size(); ++i) {
            filters_[i].update_with(lr, dfilters_[i]);
        }
        for (int i = 0; i < biases_.size(); ++i) {
            biases_[i] -= lr * dbiases_[i];
        }

        for (int i = 0; i < filters_.size(); ++i) {
            dfilters_[i].zero();
            dbiases_[i] = 0;
        }
    }

   private:
    int nb_f_;
    std::vector<Volume> filters_;
    std::vector<float> biases_;
    std::vector<Volume> dfilters_;
    std::vector<float> dbiases_;
    Volume res_;
    Volume x_;
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
    class_<Conv, boost::noncopyable>("Conv", init<int>())
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
             +[](FullyConn* fc, np::ndarray weights, float b) {
                 fc->set_weights(from_array(weights), b);
             })
        .def("grads", +[](FullyConn* fc) { return to_array(fc->grads()); })
        .def("bias", +[](FullyConn* fc) { return fc->bias(); })
        .def("weights", +[](FullyConn* fc) { return to_array(fc->weights()); })
        .def("update", &FullyConn::update);

    np::initialize();
}
