#include <algorithm>
#include <cfloat>
#include <numeric>
#include <vector>

#include "caffe/layers/topk_softmax_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void TopkSoftmaxLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  top_k_ = this->layer_param_.accuracy_param().top_k();
}

template <typename Dtype>
void TopkSoftmaxLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  num_examples_ = bottom[0]->shape(0);
  num_classes_ = bottom[0]->shape(1);
  CHECK_GE(top_k_, 1) << "top_k must be at least 1.";
  CHECK_LT(top_k_, num_classes_) << "top_k must be less than num_classes.";
  CHECK_EQ(num_examples_, bottom[1]->count())
      << "Number of labels must match the number of examples in a minibatch.";
  scratch_.resize(num_classes_);
  idx_.resize(num_classes_);
}

template <typename Dtype>
void TopkSoftmaxLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* labels = bottom[1]->cpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  auto ifirst = idx_.begin() + 1; // ground truth at 0, start at 1
  auto ikth = idx_.begin() + top_k_; // first + k - 1
  auto ilast = idx_.end();
  auto kth = scratch_.begin() + top_k_;
  auto last = scratch_.end();
  double loss = 0;

  for (int i = 0; i < num_examples_; ++i) {
    const Dtype* scores = bottom_data + i * num_classes_;
    const int label = static_cast<int>(labels[i]);

    // Ground truth at 0
    std::iota(idx_.begin(), idx_.end(), 0);
    std::swap(idx_[0], idx_[label]);

    // Find k largest elements (re-order indexes)
    std::nth_element(ifirst, ikth, ilast,
      [scores](int i1, int i2) { return scores[i1] > scores[i2]; });

    // Compute the re-ordered scores, starting with the kth largest one
    scratch_[0] = scores[label];
    for (int ix = top_k_; ix < num_classes_; ++ix) {
      scratch_[ix] = scores[idx_[ix]];
    }

    // Compute exp(score_j - M), where M is the kth largest score
    Dtype M(*kth); // kth is the maximum in [kth, last)
    std::for_each(kth + 1, last, [=](Dtype &x){ x = std::exp(x - M); });

    // Compute the log(1 + sum exp) loss and the intermediate terms
    double s = std::accumulate(kth + 1, last, 0.0);
    double a = static_cast<double>(M) - static_cast<double>(scratch_[0]);
    double b = std::exp(-a);
    loss += a + std::log1p(b + s);

    // Coefficients for the gradient
    Dtype* diff = bottom_diff + i * num_classes_;
    std::for_each(ifirst, ikth, [&](int ix){ diff[ix] = 0; });
    double coeff = 1 / (1 + s + b) / num_examples_;
    diff[idx_[top_k_]] = static_cast<Dtype>(coeff); // exp(kth - gt) / (1 + Z)
    diff[label] = static_cast<Dtype>(- (1 + s) * coeff); // - Z / (1 + Z)
    for (int ix = top_k_ + 1; ix < num_classes_; ++ix) {
      diff[idx_[ix]] = scratch_[ix] * static_cast<Dtype>(coeff);
    }
  }

  top[0]->mutable_cpu_data()[0] = loss / num_examples_;
}

template <typename Dtype>
void TopkSoftmaxLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    const Dtype coeff = top[0]->cpu_diff()[0];
    if (coeff != 1) {
      Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
      caffe_scal(bottom[0]->count(), coeff, bottom_diff);
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(TopkSoftmaxLossLayer);
#endif

INSTANTIATE_CLASS(TopkSoftmaxLossLayer);
REGISTER_LAYER_CLASS(TopkSoftmaxLoss);

}  // namespace caffe
