#ifndef CAFFE_TOPK_HINGE_LOSS_LAYER_HPP_
#define CAFFE_TOPK_HINGE_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {

// smooth top-k-of-hinge loss (top-k hinge beta)
template <typename Dtype>
class TopkHingeLossLayer : public LossLayer<Dtype> {
 public:
   explicit TopkHingeLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "TopkHingeLoss"; }
  virtual inline int ExactNumTopBlobs() const { return -1; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline int MaxTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  int top_k_;
  int num_examples_;
  int num_classes_;
  Dtype gamma_;
  Dtype lo_;
  Dtype hi_;
  Dtype rhs_;
  std::vector<Dtype> scratch_;
};

}  // namespace caffe

#endif  // CAFFE_TOPK_HINGE_LOSS_LAYER_HPP_
