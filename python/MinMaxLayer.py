import caffe
import numpy as np

class MinMaxLayer(caffe.Layer):
  def setup(self, bottom, top):
    # make sure only one input and one output
    assert len(bottom)==1 and len(top)==1, "min_max_layer expects a single input and a single output"

  def reshape(self, bottom, top):
    # reshape output to be identical to input
    top[0].reshape(*bottom[0].data.shape)

  def forward(self, bottom, top):
    # YOUR IMPLEMENTATION HERE!!
    in_ = np.array(bottom[0].data)
    x_min = in_.min()
    x_max = in_.max()
    top[0].data[...] = (in_-x_min)/(x_max-x_min)

  def backward(self, top, propagate_down, bottom):
    # backward pass is not implemented!
    pass