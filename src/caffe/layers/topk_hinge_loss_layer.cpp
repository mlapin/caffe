#include <algorithm>
#include <cfloat>
#include <numeric>
#include <vector>

#include "caffe/layers/topk_hinge_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "sdca/prox.h"

namespace caffe {

template <typename Dtype>
void TopkHingeLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  top_k_ = this->layer_param_.accuracy_param().top_k();
  gamma_ = 1; // smoothing parameter

  // Parameters for the knapsack projection
  lo_ = 0;
  hi_ = gamma_ / top_k_;
  rhs_ = gamma_;
}

template <typename Dtype>
void TopkHingeLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  num_examples_ = bottom[0]->shape(0);
  num_classes_ = bottom[0]->shape(1);
  CHECK_GE(top_k_, 1) << "top_k must be at least 1.";
  CHECK_LT(top_k_, num_classes_) << "top_k must be less than num_classes.";
  CHECK_EQ(num_examples_, bottom[1]->count())
      << "Number of labels must match the number of examples in a minibatch.";
  scratch_.resize(num_classes_);
}

template <typename Dtype>
void TopkHingeLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* labels = bottom[1]->cpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  Dtype* first = &scratch_[0] + 1;
  Dtype* last = &scratch_[0] + num_classes_;
  double loss = 0;

  caffe_copy(bottom[0]->count(), bottom_data, bottom_diff);

  for (int i = 0; i < num_examples_; ++i) {
    //const Dtype* scores = bottom_data + i * num_classes_;
    const int label = static_cast<int>(labels[i]);
    Dtype* diff = bottom_diff + i * num_classes_;

    // diff <- (a + c) (the point to project)
    std::swap(diff[0], diff[label]);
    Dtype a = 1 - diff[0];
    std::for_each(diff + 1, diff + num_classes_, [=](Dtype &x){ x += a; });

    // Compute thresholds for the projection
    caffe_copy(num_classes_ - 1, diff + 1, first);
    auto t = sdca::thresholds_knapsack_le(first, last, lo_, hi_, rhs_);

    // Compute the dot products and the loss
    Dtype ph = sdca::dot_prox(t, first, last);
    Dtype pp = sdca::dot_prox_prox(t, first, last);
    loss += ph - static_cast<Dtype>(0.5) * pp;

    // Project the scores stored in diff
    sdca::prox(t, diff + 1, diff + num_classes_);
    diff[0] = - std::accumulate(
      diff + 1, diff + num_classes_, static_cast<Dtype>(0));

    std::swap(diff[0], diff[label]);
  }

  top[0]->mutable_cpu_data()[0] = loss / (gamma_ * num_examples_);
}

template <typename Dtype>
void TopkHingeLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    const Dtype coeff = top[0]->cpu_diff()[0] / (gamma_ * num_examples_);
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    caffe_scal(bottom[0]->count(), coeff, bottom_diff);
  }
}

#ifdef CPU_ONLY
STUB_GPU(TopkHingeLossLayer);
#endif

INSTANTIATE_CLASS(TopkHingeLossLayer);
REGISTER_LAYER_CLASS(TopkHingeLoss);

}  // namespace caffe
