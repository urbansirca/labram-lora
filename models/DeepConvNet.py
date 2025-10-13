import numpy as np
from torch import nn
from torch.nn import init
from torch.nn.functional import elu
import torch


# Lightweight local replacements for braindecode helpers
class Expression(torch.nn.Module):
    """
    Compute given expression on forward pass.

    Parameters
    ----------
    expression_fn: function
        Should accept variable number of objects of type
        `torch.autograd.Variable` to compute its output.
    """

    def __init__(self, expression_fn):
        super(Expression, self).__init__()
        self.expression_fn = expression_fn

    def forward(self, *x):
        return self.expression_fn(*x)

    def __repr__(self):
        if hasattr(self.expression_fn, "func") and hasattr(
            self.expression_fn, "kwargs"
        ):
            expression_str = "{:s} {:s}".format(
                self.expression_fn.func.__name__, str(self.expression_fn.kwargs)
            )
        elif hasattr(self.expression_fn, "__name__"):
            expression_str = self.expression_fn.__name__
        else:
            expression_str = repr(self.expression_fn)
        return self.__class__.__name__ + "(" + "expression=" + str(expression_str) + ")"


def identity(x):
    return x


class DeepConvNet(nn.Module):
    """
    Deep ConvNet model from [1]_.

    References
    ----------

    .. [1] Schirrmeister, R. T., Springenberg, J. T., Fiederer, L. D. J.,
       Glasstetter, M., Eggensperger, K., Tangermann, M., Hutter, F. & Ball, T. (2017).
       Deep learning with convolutional neural networks for EEG decoding and
       visualization.
       Human Brain Mapping , Aug. 2017. Online: http://dx.doi.org/10.1002/hbm.23730
    """

    def __init__(
        self,
        in_chans,
        n_classes,
        input_time_length,
        final_conv_length,
        n_filters_time=25,
        n_filters_spat=25,
        filter_time_length=10,
        pool_time_length=3,
        pool_time_stride=3,
        n_filters_2=50,
        filter_length_2=10,
        n_filters_3=100,
        filter_length_3=10,
        n_filters_4=200,
        filter_length_4=10,
        first_nonlin=elu,
        first_pool_mode="max",
        first_pool_nonlin=identity,
        later_nonlin=elu,
        later_pool_mode="max",
        later_pool_nonlin=identity,
        drop_prob=0.5,
        double_time_convs=False,
        split_first_layer=True,
        batch_norm=True,
        batch_norm_alpha=0.1,
        stride_before_pool=False,
    ):
        super().__init__()
        if final_conv_length == "auto":
            assert input_time_length is not None

        self.__dict__.update(locals())

        # Build underlying sequential network once
        self.model = self.create_network()

    def create_network(self):
        if self.stride_before_pool:
            conv_stride = self.pool_time_stride
            pool_stride = 1
        else:
            conv_stride = 1
            pool_stride = self.pool_time_stride
        pool_class_dict = dict(max=nn.MaxPool2d, mean=nn.AvgPool2d)
        first_pool_class = pool_class_dict[self.first_pool_mode]
        later_pool_class = pool_class_dict[self.later_pool_mode]
        model = nn.Sequential()
        if self.split_first_layer:
            model.add_module("dimshuffle", Expression(_transpose_time_to_spat))
            model.add_module(
                "conv_time",
                nn.Conv2d(
                    1,
                    self.n_filters_time,
                    (self.filter_time_length, 1),
                    stride=1,
                ),
            )
            model.add_module(
                "conv_spat",
                nn.Conv2d(
                    self.n_filters_time,
                    self.n_filters_spat,
                    (1, self.in_chans),
                    stride=(conv_stride, 1),
                    bias=not self.batch_norm,
                ),
            )
            n_filters_conv = self.n_filters_spat
        else:
            model.add_module(
                "conv_time",
                nn.Conv2d(
                    self.in_chans,
                    self.n_filters_time,
                    (self.filter_time_length, 1),
                    stride=(conv_stride, 1),
                    bias=not self.batch_norm,
                ),
            )
            n_filters_conv = self.n_filters_time
        if self.batch_norm:
            model.add_module(
                "bnorm",
                nn.BatchNorm2d(
                    n_filters_conv,
                    momentum=self.batch_norm_alpha,
                    affine=True,
                    eps=1e-5,
                ),
            )
        model.add_module("conv_nonlin", Expression(self.first_nonlin))
        model.add_module(
            "pool",
            first_pool_class(
                kernel_size=(self.pool_time_length, 1), stride=(pool_stride, 1)
            ),
        )
        model.add_module("pool_nonlin", Expression(self.first_pool_nonlin))

        def add_conv_pool_block(
            model, n_filters_before, n_filters, filter_length, block_nr
        ):
            suffix = "_{:d}".format(block_nr)
            model.add_module("drop" + suffix, nn.Dropout(p=self.drop_prob))
            model.add_module(
                "conv" + suffix,
                nn.Conv2d(
                    n_filters_before,
                    n_filters,
                    (filter_length, 1),
                    stride=(conv_stride, 1),
                    bias=not self.batch_norm,
                ),
            )
            if self.batch_norm:
                model.add_module(
                    "bnorm" + suffix,
                    nn.BatchNorm2d(
                        n_filters,
                        momentum=self.batch_norm_alpha,
                        affine=True,
                        eps=1e-5,
                    ),
                )
            model.add_module("nonlin" + suffix, Expression(self.later_nonlin))

            model.add_module(
                "pool" + suffix,
                later_pool_class(
                    kernel_size=(self.pool_time_length, 1),
                    stride=(pool_stride, 1),
                ),
            )
            model.add_module("pool_nonlin" + suffix, Expression(self.later_pool_nonlin))

        add_conv_pool_block(
            model, n_filters_conv, self.n_filters_2, self.filter_length_2, 2
        )
        add_conv_pool_block(
            model, self.n_filters_2, self.n_filters_3, self.filter_length_3, 3
        )
        add_conv_pool_block(
            model, self.n_filters_3, self.n_filters_4, self.filter_length_4, 4
        )

        # model.add_module('drop_classifier', nn.Dropout(p=self.drop_prob))
        model.eval()
        if self.final_conv_length == "auto":
            out = model(
                torch.tensor(
                    np.ones(
                        (1, self.in_chans, self.input_time_length, 1),
                        dtype=np.float32,
                    )
                )
            )
            n_out_time = out.cpu().data.numpy().shape[2]
            self.final_conv_length = n_out_time
        model.add_module(
            "conv_classifier",
            nn.Conv2d(
                self.n_filters_4,
                self.n_classes,
                (self.final_conv_length, 1),
                bias=True,
            ),
        )
        # model.add_module("softmax", nn.LogSoftmax(dim=1))
        model.add_module("squeeze", Expression(_squeeze_final_output))

        # Initialization, xavier is same as in our paper...
        # was default from lasagne
        init.xavier_uniform_(model.conv_time.weight, gain=1)
        # maybe no bias in case of no split layer and batch norm
        if self.split_first_layer or (not self.batch_norm):
            init.constant_(model.conv_time.bias, 0)
        if self.split_first_layer:
            init.xavier_uniform_(model.conv_spat.weight, gain=1)
            if not self.batch_norm:
                init.constant_(model.conv_spat.bias, 0)
        if self.batch_norm:
            init.constant_(model.bnorm.weight, 1)
            init.constant_(model.bnorm.bias, 0)
        param_dict = dict(list(model.named_parameters()))
        for block_nr in range(2, 5):
            conv_weight = param_dict["conv_{:d}.weight".format(block_nr)]
            init.xavier_uniform_(conv_weight, gain=1)
            if not self.batch_norm:
                conv_bias = param_dict["conv_{:d}.bias".format(block_nr)]
                init.constant_(conv_bias, 0)
            else:
                bnorm_weight = param_dict["bnorm_{:d}.weight".format(block_nr)]
                bnorm_bias = param_dict["bnorm_{:d}.bias".format(block_nr)]
                init.constant_(bnorm_weight, 1)
                init.constant_(bnorm_bias, 0)

        init.xavier_uniform_(model.conv_classifier.weight, gain=1)
        init.constant_(model.conv_classifier.bias, 0)

        # Start in eval mode
        model.eval()
        return model

    def forward(self, x, **kwargs):
        """
        Accepts (B, C, T) or (B, C, P, T). Returns network output.

        This mirrors the simple forward used in `EEGNet.forward` while
        preserving the expected input layout for this architecture.
        """
        # If input has patches dimension, flatten it into time
        if x.ndim == 4:
            # (B, C, P, T) -> (B, C, P*T)
            # print("dimension is ", x.shape)
            # print("DeepConvNet reshaping input from (B,C,P,T) to (B,C,P*T)")
            b, c, p, t = x.shape
            x = x.reshape(b, c, p * t)
            # print("dimension is ", x.shape)

        elif x.ndim != 3:
            raise ValueError(
                f"DeepConvNet expects (B,C,T) or (B,C,P,T), got shape {tuple(x.shape)}"
            )

        # Braindecode DeepConvNet expects input as (B, C, T, 1)
        x = x.unsqueeze(-1)
        out = self.model(x)
        # print("DeepConvNet output shape:", out.shape)  # Debug print
        return out


# remove empty dim at end and potentially remove empty time dim
# do not just use squeeze as we never want to remove first dim
def _squeeze_final_output(x):
    assert x.size()[3] == 1
    x = x[:, :, :, 0]
    if x.size()[2] == 1:
        x = x[:, :, 0]
    return x


def _transpose_time_to_spat(x):
    return x.permute(0, 3, 2, 1)



def freeze_all_but_head_deepconvnet(model):
    """
    Freezes all parameters in the model except for the classification head (conv_classifier).
    This function assumes the classification head is named 'conv_classifier' as in the provided DeepConvNet.
    """
    # print all parameters
    # print("model", model)
    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False

    # Check if 'conv_classifier' exists inside the 'model' (which is a Sequential container)
    if hasattr(model.model, 'conv_classifier'):
        # Unfreeze the classification head (conv_classifier)
        for param in model.model.conv_classifier.parameters():
            param.requires_grad = True
    else:
        raise AttributeError("Model does not have 'conv_classifier' as a head. Please check the model architecture.")

    print("Number of parameters after freezing:", sum(p.numel() for p in model.parameters() if p.requires_grad), " out of ", sum(p.numel() for p in model.parameters()))
    model.train()
    return model
