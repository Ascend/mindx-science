
from mindspore.train.callback import Callback
from mindspore import ops


class TlossCallback(Callback):
    def __init__(self, net, feature, label):
        super(TlossCallback, self).__init__()
        self.net = net
        self.feature = feature
        self.label = label
        self.rs = ops.ReduceSum(keep_dims=True)

    def epoch_end(self, run_context):
        predict = self.net(self.feature)
        result = ops.L2Loss()(predict - self.label)
        predict_h = ops.Sqrt()(self.rs(ops.Pow()(predict, 2), 1))
        label_h = ops.Sqrt()(self.rs(ops.Pow()(self.label, 2), 1))
        print('test_loss:', result)
        print('u, v=', ops.Sqrt()(
            self.rs(ops.Pow()(predict - self.label, 2), 0) /
            self.rs(ops.Pow()(self.label, 2), 0))
              )
        print('h=', ops.Sqrt()(
            self.rs(ops.Pow()(predict_h - label_h, 2), 0) /
            self.rs(ops.Pow()(label_h, 2), 0))
              )
