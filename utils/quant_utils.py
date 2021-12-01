import torch
import torch.nn as nn
from typing import  Tuple
import warnings
from torch.quantization import (
    FakeQuantize, 
    MovingAverageMinMaxObserver, 
    MinMaxObserver, 
    PerChannelMinMaxObserver
)
from torch.nn import BatchNorm2d
from torch.quantization.fake_quantize import _is_per_channel, _is_per_tensor
import torch.nn.functional as F
from torch.quantization.observer import (
    HistogramObserver, MinMaxObserver, _ObserverBase
)

BITS_QUANT_RANGE_MAPPING = {
    "signed":{
        8: [-128, 127], 
        7: [-64, 63], 
        6: [-32, 31],
        5: [-16, 15], 
        4: [-8, 7], 
        3: [-4, 3], 
        2: [-2, 1]
    },
    "unsigned": {
        8: [0, 255], 
        7: [0, 127], 
        6: [0, 63], 
        5: [0, 31], 
        4: [0, 15], 
        3: [0, 7], 
        2: [0, 3]}
}


def print_model(model: nn.Module):
    """Print modules in model."""
    for i, (n, m) in enumerate(model.named_modules()):
        print(f"{n}: {type(m)}")

        # break early since I don't need to check whole thing
        if i == 30:
            break


# There is an existing bug in PyTorch where the quantisation bounds
# are not passed to the simulated observer
# observer normally just assumes defaults then (8 bit signed/unsigned)
class EditedFakeQuantize(FakeQuantize):
    
    r""" Simulate the quantize and dequantize operations in training time.
    The output of this module is given by
    x_out = (clamp(round(x/scale + zero_point),
             quant_min, quant_max)-zero_point)*scale
    * :attr:`scale` defines the scale factor used for quantization.
    * :attr:`zero_point` specifies the quantized value to which 0 in floating point maps to
    * :attr:`quant_min` specifies the minimum allowable quantized value.
    * :attr:`quant_max` specifies the maximum allowable quantized value.
    * :attr:`fake_quant_enable` controls the application of fake quantization on tensors, note that
      statistics can still be updated.
    * :attr:`observer_enable` controls statistics collection on tensors
    * :attr:`dtype` specifies the quantized dtype that is being emulated with fake-quantization,
                    allowable values are torch.qint8 and torch.quint8. The values of quant_min and
                    quant_max should be chosen to be consistent with the dtype
    Args:
        observer (module): Module for observing statistics on input tensors and calculating scale
                           and zero-point.
        quant_min (int): The minimum allowable quantized value.
        quant_max (int): The maximum allowable quantized value.
        observer_kwargs (optional): Arguments for the observer module
    Attributes:
        observer (Module): User provided module that collects statistics on the input tensor and
                           provides a method to calculate scale and zero-point.
    """

    scale: torch.Tensor
    zero_point: torch.Tensor
    def __init__(
        self, 
        observer=MovingAverageMinMaxObserver, 
        quant_min=0, 
        quant_max=255, 
        **observer_kwargs
    ):
        super().__init__(
            observer=observer,
            quant_min=quant_min,
            quant_max=quant_max,
            **observer_kwargs
        )
        # overwrite error in sourcecode
        # quant_min and quant_max previously not passed 
        self.activation_post_process = observer(
            quant_min=quant_min, quant_max=quant_max, **observer_kwargs
        )
        assert torch.iinfo(
            self.activation_post_process.dtype
        ).min <= quant_min, 'quant_min out of bound'
        assert quant_max <= torch.iinfo(
            self.activation_post_process.dtype
        ).max, 'quant_max out of bound'

        self.dtype = self.activation_post_process.dtype
        self.qscheme = self.activation_post_process.qscheme
        self.ch_axis = self.activation_post_process.ch_axis \
        if hasattr(self.activation_post_process, 'ch_axis') else -1
        assert _is_per_channel(self.qscheme) or \
            _is_per_tensor(self.qscheme), \
            'Only per channel and per tensor quantization are supported in fake quantize' + \
            ' got qscheme: ' + str(self.qscheme)
        self.is_per_channel = _is_per_channel(self.qscheme)

    # note that the forward() method that is inherited automatically
    # clamps the output, however the actually quantized backend (not 
    # simulated, QNNpack or FBGEMM) does not clamp 
    # and is only passed zeropoint and scale

# -----------------------------------------------------------------------------
# copied and edited from 
# https://github.com/pytorch/pytorch/blob/master/torch/quantization/observer.py
# by default this method only does 8-bit activations for some reason
# longboi
class EditedHistogramObserver(_ObserverBase):
    r"""
    The module records the running histogram of tensor values along with
    min/max values. ``calculate_qparams`` will calculate scale and zero_point.
    Edited from original PyTorch version to support different bitwidths.

    Args:
        bins: Number of bins to use for the histogram
        upsample_rate: Factor by which the histograms are upsampled, this is
                       used to interpolate histograms with varying 
                       ranges across observations
        dtype: Quantized data type
        qscheme: Quantization scheme to be used
        reduce_range: Reduces the range of the quantized data type by 1 bit

    The scale and zero point are computed as follows:

    1. Create the histogram of the incoming inputs.
        The histogram is computed continuously, and the ranges per bin change
        with every new tensor observed.
    2. Search the distribution in the histogram for optimal min/max values.
        The search for the min/max values ensures the minimization of the
        quantization error with respect to the floating point model.
    3. Compute the scale and zero point the same way as in the
        :class:`~torch.quantization.MinMaxObserver`
    """
    histogram: torch.Tensor
    min_val: torch.Tensor
    max_val: torch.Tensor

    def __init__(
        self,
        bins: int = 2048,
        upsample_rate: int = 128,
        dtype: torch.dtype = torch.quint8,
        qscheme=torch.per_tensor_affine,
        quant_min=0, # additional kwargs that specify bitwidth
        quant_max=255, # actually need these defaults because 
        # EditedFakeQuantize ends up not passing these arguments when it calls
        # super()
        reduce_range=False,
        factory_kwargs=None,
    ) -> None:

        # bins: The number of bins used for histogram calculation.
        super().__init__(
            dtype=dtype,
            qscheme=qscheme,
            reduce_range=reduce_range,
            quant_min=quant_min,
            quant_max=quant_max,
            factory_kwargs=factory_kwargs,
        )

        # need this as tensors will clash otherwise
        self.dev = "cpu"

        factory_kwargs = torch.nn.factory_kwargs(factory_kwargs)
        self.bins = bins
        self.register_buffer('histogram', torch.zeros(
            self.bins, **factory_kwargs).to(self.dev))
        self.register_buffer('min_val', torch.tensor(
            float('inf'), **factory_kwargs).to(self.dev))
        self.register_buffer('max_val', torch.tensor(
            float('-inf'), **factory_kwargs).to(self.dev))

        # 2^nbits gives number of quant levels in original code
        # self.dst_nbins = 2 ** torch.iinfo(self.dtype).bits

        # replace with 
        self.dst_nbins = self.quant_max - self.quant_min

        self.upsample_rate = upsample_rate



    def _get_norm(
        self,
        delta_begin: torch.Tensor,
        delta_end: torch.Tensor,
        density: torch.Tensor
    ) -> torch.Tensor:
        r"""
        Compute the norm of the values uniformaly distributed between
        delta_begin and delta_end.
        Currently only L2 norm is supported.

        norm = density * (integral_{begin, end} x^2)
             = density * (end^3 - begin^3) / 3
        """
        norm = (
            delta_end * delta_end * delta_end
            - delta_begin * delta_begin * delta_begin
        ) / 3
        return density.to(self.dev) * norm.to(self.dev)

    def _compute_quantization_error(
        self, next_start_bin: int, next_end_bin: int
    ):
        r"""
        Compute the quantization error if we use start_bin to end_bin as the
        min and max to do the quantization.
        """
        bin_width = (self.max_val.item() - self.min_val.item()) / self.bins

        dst_bin_width = bin_width * \
            (next_end_bin - next_start_bin + 1) / self.dst_nbins
        if dst_bin_width == 0.0:
            return 0.0

        src_bin = torch.arange(self.bins).to(self.dev)
        # distances from the beginning of first dst_bin to the beginning and
        # end of src_bin
        src_bin_begin = (src_bin - next_start_bin) * bin_width
        src_bin_end = src_bin_begin + bin_width

        # which dst_bins the beginning and end of src_bin belong to?
        # placing histogram bins into quantization bins
        dst_bin_of_begin = torch.clamp(
            src_bin_begin // dst_bin_width, 0, self.dst_nbins - 1)
        dst_bin_of_begin_center = (dst_bin_of_begin + 0.5) * dst_bin_width

        dst_bin_of_end = torch.clamp(
            src_bin_end // dst_bin_width, 0, self.dst_nbins - 1)
        dst_bin_of_end_center = (dst_bin_of_end + 0.5) * dst_bin_width

        density = self.histogram / bin_width

        norm = torch.zeros(self.bins).to(self.dev)

        # I think this is considering the expected squared error
        # below, within and above the quantization range

        delta_begin = src_bin_begin - dst_bin_of_begin_center
        delta_end = dst_bin_width / 2
        norm += self._get_norm(
            delta_begin,
            torch.ones(self.bins).to(self.dev) * delta_end, 
            density
        )

        norm += (dst_bin_of_end - dst_bin_of_begin - 1) * self._get_norm(
            torch.tensor(-dst_bin_width / 2).to(self.dev), 
            torch.tensor(dst_bin_width / 2).to(self.dev), 
            density
        )

        dst_bin_of_end_center = (
            dst_bin_of_end * dst_bin_width + dst_bin_width / 2
        )

        delta_begin = -dst_bin_width / 2
        delta_end = src_bin_end - dst_bin_of_end_center
        norm += self._get_norm(
            torch.tensor(delta_begin).to(self.dev), delta_end, density
        )

        return norm.sum().item()

    def _non_linear_param_search(self) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Non-linear parameter search.

        An approximation for L2 error minimization for selecting min/max.
        By selecting new min/max, we filter out outliers in input distribution.
        This follows the implementation of 
        NormMinimization::NonlinearQuantizationParamsSearch in
        caffe2/quantization/server/norm_minimization.cc
        """
        assert self.histogram.size()[0] == self.bins, "bins mistmatch"
        bin_width = (self.max_val - self.min_val) / self.bins

        # cumulative sum
        total = torch.sum(self.histogram).item()
        cSum = torch.cumsum(self.histogram, dim=0)

        stepsize = 1e-5  # granularity
        alpha = 0.0  # lower bound
        beta = 1.0  # upper bound
        start_bin = 0
        end_bin = self.bins - 1
        norm_min = float("inf")

        while alpha < beta:
            # Find the next step
            next_alpha = alpha + stepsize
            next_beta = beta - stepsize

            # find the left and right bins between the quantile bounds
            l = start_bin
            r = end_bin
            while l < end_bin and cSum[l] < next_alpha * total:
                l = l + 1
            while r > start_bin and cSum[r] > next_beta * total:
                r = r - 1

            # decide the next move
            next_start_bin = start_bin
            next_end_bin = end_bin
            if (l - start_bin) > (end_bin - r):
                # move the start bin
                next_start_bin = l
                alpha = next_alpha
            else:
                # move the end bin
                next_end_bin = r
                beta = next_beta

            if next_start_bin == start_bin and next_end_bin == end_bin:
                continue

            # calculate the quantization error using next_start_bin and next_end_bin
            norm = self._compute_quantization_error(
                next_start_bin, next_end_bin)

            if norm > norm_min:
                break
            norm_min = norm
            start_bin = next_start_bin
            end_bin = next_end_bin

        new_min = self.min_val + bin_width * start_bin
        new_max = self.min_val + bin_width * (end_bin + 1)
        return new_min, new_max

    def _adjust_min_max(
        self,
        combined_min: torch.Tensor,
        combined_max: torch.Tensor,
        upsample_rate: int
    ) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
        # We ensure that:
        # (combined_max - combined_min)/(downsample_rate*Nbins) = (max - min)/(upsample_rate*Nbins)
        # This allows us to have a common grid of resolution s, where we can align
        # the input histogram
        # start_idx maps min_val to the histogram bin index.

        hist_bin_width = (self.max_val - self.min_val) / \
            (self.bins * upsample_rate)
        downsample_rate = int(torch.ceil(
            (combined_max - combined_min) / (self.bins * hist_bin_width)).item())
        e = downsample_rate * (self.bins * hist_bin_width) - \
            (combined_max - combined_min)
        # Relax only the max, not the min, so that for one sided distributions, 
        # min stays at zero
        combined_max = combined_max + e
        combined_min = combined_min
        start_idx = int(torch.round(
            (self.min_val - combined_min) / hist_bin_width).item())
        return combined_min, combined_max, downsample_rate, start_idx

    def _combine_histograms(self,
                            orig_hist: torch.Tensor,
                            new_hist: torch.Tensor,
                            upsample_rate: int,
                            downsample_rate: int,
                            start_idx: int,
                            Nbins: int) -> torch.Tensor:
        # First up-sample the histogram with new data by a factor of L
        # This creates an approximate probability density thats piecwise constant
        upsampled_histogram = new_hist.repeat_interleave(upsample_rate)
        # Now insert the upsampled histogram into the output
        # histogram, which is initialized with zeros.
        # The offset at which the histogram is introduced is determined
        # by the start index as the output histogram can cover a wider range
        histogram_with_output_range = torch.zeros(
            (Nbins * downsample_rate), 
            device=orig_hist.device
        ).to(self.dev)
        histogram_with_output_range[start_idx:Nbins *
                                    upsample_rate + start_idx] = upsampled_histogram
        
        # Compute integral histogram, double precision is needed to ensure
        # that there are no overflows
        integral_histogram = torch.cumsum(
            histogram_with_output_range, 0,
            dtype=torch.double
        )[downsample_rate - 1:: downsample_rate]
        # Finally perform interpolation
        shifted_integral_histogram = torch.zeros(
            (Nbins), device=orig_hist.device
        ).to(self.dev)
        shifted_integral_histogram[1:Nbins] = integral_histogram[0:-1]
        interpolated_histogram = (
            integral_histogram - shifted_integral_histogram) / upsample_rate
        orig_hist = orig_hist + interpolated_histogram.to(torch.float)
        return orig_hist

    def forward(self, x_orig: torch.Tensor) -> torch.Tensor:

        # set device depending on input
        self.dev = x_orig.device
        if x_orig.numel() == 0:
            return x_orig
        x = x_orig.detach()
        min_val = self.min_val
        max_val = self.max_val
        same_values = min_val.item() == max_val.item()
        is_uninitialized = min_val == float('inf') and max_val == float('-inf')
        if is_uninitialized or same_values:
            min_val, max_val = torch._aminmax(x)
            self.min_val.resize_(min_val.shape)
            self.min_val.copy_(min_val)
            self.max_val.resize_(max_val.shape)
            self.max_val.copy_(max_val)
            assert min_val.numel() == 1 and max_val.numel() == 1, (
                "histogram min/max values must be scalar."
            )
            torch.histc(x, self.bins, min=int(min_val),
                        max=int(max_val), out=self.histogram)
        else:
            new_min, new_max = torch._aminmax(x)
            combined_min = torch.min(new_min, min_val)
            combined_max = torch.max(new_max, max_val)
            # combine the existing histogram and new histogram into 1 histogram
            # We do this by first upsampling the histogram to a dense grid
            # and then downsampling the histogram efficiently
            combined_min, combined_max, downsample_rate, start_idx = \
                self._adjust_min_max(
                    combined_min, combined_max, self.upsample_rate)
            assert combined_min.numel() == 1 and combined_max.numel() == 1, (
                "histogram min/max values must be scalar."
            )
            combined_histogram = torch.histc(
                x, self.bins, min=int(combined_min), max=int(combined_max))
            if combined_min == min_val and combined_max == max_val:
                combined_histogram += self.histogram
            else:
                combined_histogram = self._combine_histograms(
                    combined_histogram,
                    self.histogram,
                    self.upsample_rate,
                    downsample_rate,
                    start_idx,
                    self.bins)

            self.histogram.resize_(combined_histogram.shape)
            self.histogram.copy_(combined_histogram)
            self.min_val.resize_(combined_min.shape)
            self.min_val.copy_(combined_min)
            self.max_val.resize_(combined_max.shape)
            self.max_val.copy_(combined_max)
        return x_orig

    @torch.jit.export
    def calculate_qparams(self):
        is_uninitialized = (self.min_val == float('inf') and
                            self.max_val == float('-inf'))
        if is_uninitialized:
            warnings.warn(
                "must run observer before calling calculate_qparams.\
                                    Returning default scale and zero point "
            )
            return torch.tensor([1.0]).to(self.dev), torch.tensor([0]).to(self.dev)
        assert self.bins == len(self.histogram), (
            "The number of bins in histogram should be equal to the number of bins "
            "supplied while making this observer"
        )

        new_min, new_max = self._non_linear_param_search()

        # from _ObserverBase
        return self._calculate_qparams(new_min, new_max)

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        super(HistogramObserver, self)._save_to_state_dict(
            destination, prefix, keep_vars)
        destination[prefix + 'min_val'] = self.min_val
        destination[prefix + 'max_val'] = self.max_val

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        version = local_metadata.get('version', None)

        if version is None or version < 3:
            # if min_val and max_val are not initialized, update their shape
            # to account for the differences between v2 and v3
            min_val_name, max_val_name = prefix + 'min_val', prefix + 'max_val'
            if min_val_name in state_dict:
                if state_dict[min_val_name].shape == torch.Size([0]):
                    state_dict[min_val_name] = torch.tensor(float('inf')).to(self.dev)
            if max_val_name in state_dict:
                if state_dict[max_val_name].shape == torch.Size([0]):
                    state_dict[max_val_name] = torch.tensor(float('-inf')).to(self.dev)

        local_state = ['min_val', 'max_val']
        for name in local_state:
            key = prefix + name
            if key in state_dict:
                val = state_dict[key]
                setattr(self, name, val)
            elif strict:
                missing_keys.append(key)
        super(HistogramObserver, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs
        )


# Per-channel version of the above
class PerChannelHistogramObserver(_ObserverBase):
    r"""
    The module records the running histogram of tensor values along with
    min/max values. ``calculate_qparams`` will calculate scale and zero_point.
    Edited from original PyTorch version to support different bitwidths.

    Args:
        bins: Number of bins to use for the histogram
        upsample_rate: Factor by which the histograms are upsampled, this is
                       used to interpolate histograms with varying 
                       ranges across observations
        dtype: Quantized data type
        qscheme: Quantization scheme to be used
        reduce_range: Reduces the range of the quantized data type by 1 bit

    The scale and zero point are computed as follows:

    1. Create the histogram of the incoming inputs.
        The histogram is computed continuously, and the ranges per bin change
        with every new tensor observed.
    2. Search the distribution in the histogram for optimal min/max values.
        The search for the min/max values ensures the minimization of the
        quantization error with respect to the floating point model.
    3. Compute the scale and zero point the same way as in the
        :class:`~torch.quantization.MinMaxObserver`
    """
    histogram: torch.Tensor
    min_val: torch.Tensor
    max_val: torch.Tensor

    def __init__(
        self,
        ch_axis=0,
        bins: int = 2048,
        upsample_rate: int = 128,
        dtype: torch.dtype = torch.quint8,
        qscheme=torch.per_tensor_affine,
        quant_min=0,  # additional kwargs that specify bitwidth
        quant_max=255,  # actuall need these defaults because
        # EditedFakeQuantize ends up not passing these arguments when it calls
        # super()
        reduce_range=False,
        factory_kwargs=None,
    ) -> None:

        # bins: The number of bins used for histogram calculation.
        super().__init__(
            dtype=dtype,
            qscheme=qscheme,
            reduce_range=reduce_range,
            quant_min=quant_min,
            quant_max=quant_max,
            factory_kwargs=factory_kwargs,
        )

        # need this as tensors will clash otherwise
        self.dev = "cpu"

        factory_kwargs = torch.nn.factory_kwargs(factory_kwargs)
        self.bins = bins
        self.register_buffer('histograms', torch.tensor([], **factory_kwargs).to(self.dev))
        self.ch_axis = ch_axis
        self.register_buffer('min_vals', torch.tensor([], **factory_kwargs))
        self.register_buffer('max_vals', torch.tensor([], **factory_kwargs))
        self.new_mins = torch.tensor([])
        self.new_maxs = torch.tensor([])
        if (
            self.qscheme == torch.per_channel_symmetric
            and self.reduce_range
            and self.dtype == torch.quint8
        ):
            raise NotImplementedError(
                "Cannot reduce range for symmetric quantization for quint8"
            )
        # 2^nbits gives number of quant levels in original code
        # self.dst_nbins = 2 ** torch.iinfo(self.dtype).bits

        # replace with
        self.dst_nbins = self.quant_max - self.quant_min

        self.upsample_rate = upsample_rate


    def _get_norm(
        self,
        delta_begin: torch.Tensor,
        delta_end: torch.Tensor,
        density: torch.Tensor
    ) -> torch.Tensor:
        r"""
        Compute the norm of the values uniformaly distributed between
        delta_begin and delta_end.
        Currently only L2 norm is supported.

        norm = density * (integral_{begin, end} x^2)
             = density * (end^3 - begin^3) / 3
        """
        norm = (
            delta_end * delta_end * delta_end
            - delta_begin * delta_begin * delta_begin
        ) / 3
        return density.to(self.dev) * norm.to(self.dev)

    def _compute_quantization_error(
        self, next_start_bin: int, next_end_bin: int, histogram,
        min_val, max_val,
    ):
        r"""
        Compute the quantization error if we use start_bin to end_bin as the
        min and max to do the quantization.
        """
        bin_width = (max_val.item() - min_val.item()) / self.bins

        dst_bin_width = bin_width * \
            (next_end_bin - next_start_bin + 1) / self.dst_nbins
        if dst_bin_width == 0.0:
            return 0.0

        src_bin = torch.arange(self.bins).to(self.dev)
        # distances from the beginning of first dst_bin to the beginning and
        # end of src_bin
        src_bin_begin = (src_bin - next_start_bin) * bin_width
        src_bin_end = src_bin_begin + bin_width

        # which dst_bins the beginning and end of src_bin belong to?
        # placing histogram bins into quantization bins
        dst_bin_of_begin = torch.clamp(
            src_bin_begin // dst_bin_width, 0, self.dst_nbins - 1)
        dst_bin_of_begin_center = (dst_bin_of_begin + 0.5) * dst_bin_width

        dst_bin_of_end = torch.clamp(
            src_bin_end // dst_bin_width, 0, self.dst_nbins - 1)
        dst_bin_of_end_center = (dst_bin_of_end + 0.5) * dst_bin_width

        density = histogram / bin_width

        norm = torch.zeros(self.bins).to(self.dev)

        # I think this is considering the expected squared error
        # below, within and above the quantization range

        delta_begin = src_bin_begin - dst_bin_of_begin_center
        delta_end = dst_bin_width / 2
        norm += self._get_norm(
            delta_begin,
            torch.ones(self.bins).to(self.dev) * delta_end,
            density
        )

        norm += (dst_bin_of_end - dst_bin_of_begin - 1) * self._get_norm(
            torch.tensor(-dst_bin_width / 2).to(self.dev),
            torch.tensor(dst_bin_width / 2).to(self.dev),
            density
        )

        dst_bin_of_end_center = (
            dst_bin_of_end * dst_bin_width + dst_bin_width / 2
        )

        delta_begin = -dst_bin_width / 2
        delta_end = src_bin_end - dst_bin_of_end_center
        norm += self._get_norm(
            torch.tensor(delta_begin).to(self.dev), delta_end, density
        )

        return norm.sum().item()

    def _non_linear_param_search(self) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Non-linear parameter search.

        An approximation for L2 error minimization for selecting min/max.
        By selecting new min/max, we filter out outliers in input distribution.
        This follows the implementation of 
        NormMinimization::NonlinearQuantizationParamsSearch in
        caffe2/quantization/server/norm_minimization.cc
        """

        new_mins = []
        new_maxs = []

        for i, histogram in enumerate(self.histograms):
            assert histogram.size()[0] == self.bins, "bins mistmatch"
            bin_width = (self.max_vals[i] - self.min_vals[i]) / self.bins

            # cumulative sum
            total = torch.sum(histogram).item()
            cSum = torch.cumsum(histogram, dim=0)

            stepsize = 1e-5  # granularity
            alpha = 0.0  # lower bound
            beta = 1.0  # upper bound
            start_bin = 0
            end_bin = self.bins - 1
            norm_min = float("inf")
            norm = self._compute_quantization_error(
                start_bin, end_bin, histogram, 
                self.min_vals[i], self.max_vals[i]
            )

            # break before any iterations
            # print(norm)

            if norm > norm_min:
                pass

            # set new min as the starting 
            else:
                norm_min = norm
                while alpha < beta:

                    # Find the next step
                    next_alpha = alpha + stepsize
                    next_beta = beta - stepsize

                    # find the left and right bins between the quantile bounds
                    l = start_bin
                    r = end_bin
                    while l < end_bin and cSum[l] < next_alpha * total:
                        l = l + 1
                    while r > start_bin and cSum[r] > next_beta * total:
                        r = r - 1

                    # decide the next move
                    next_start_bin = start_bin
                    next_end_bin = end_bin
                    if (l - start_bin) > (end_bin - r):
                        # move the start bin
                        next_start_bin = l
                        alpha = next_alpha
                    else:
                        # move the end bin
                        next_end_bin = r
                        beta = next_beta

                    if next_start_bin == start_bin and next_end_bin == end_bin:
                        continue

                    # calculate the quantization error using next_start_bin and next_end_bin
                    norm = self._compute_quantization_error(
                        next_start_bin, next_end_bin, histogram, 
                        self.min_vals[i], self.max_vals[i]
                    )

                    # print(norm)

                    if norm > norm_min:
                        # print("break")
                        break
                    norm_min = norm
                    start_bin = next_start_bin
                    end_bin = next_end_bin

            new_min = self.min_vals[i] + bin_width * start_bin
            new_max = self.min_vals[i] + bin_width * (end_bin + 1)
            new_mins.append(new_min)
            new_maxs.append(new_max)
        

        return torch.tensor(new_mins).to(self.dev), torch.tensor(new_maxs).to(self.dev)




    def forward(self, x_orig: torch.Tensor) -> torch.Tensor:

        # set device depending on input
        self.dev = x_orig.device
        if x_orig.numel() == 0:
            return x_orig
        x = x_orig.detach()
        min_vals = self.min_vals
        max_vals = self.max_vals
        x_dim = x.size()

        # no combination/memory as this is just going to be for weights

        new_axis_list = [i for i in range(len(x_dim))]  # noqa: C416
        new_axis_list[self.ch_axis] = 0
        new_axis_list[0] = self.ch_axis
        y = x.permute(new_axis_list)
        # Need to match dtype of min/max because the updates to buffers
        # are done in place and types need to match for comparisons
        y = y.to(self.min_vals.dtype)
        y = torch.flatten(y, start_dim=1)

        # not found yet
        if min_vals.numel() == 0 or max_vals.numel() == 0:
            min_vals, max_vals = torch._aminmax(y, 1)
       

            self.min_vals.resize_(min_vals.shape)
            self.max_vals.resize_(max_vals.shape)
            self.min_vals.copy_(min_vals)
            self.max_vals.copy_(max_vals)

            # reset histograms
            histograms = torch.zeros((len(self.min_vals), self.bins))
            self.histograms.resize_(histograms.shape)
            self.histograms.copy_(histograms)

            for i in range(len(self.histograms)):
                self.histograms[i] = torch.histc(y[i], self.bins, min=int(min_vals[i]),
                            max=int(max_vals[i])) 

        return x_orig

    @torch.jit.export
    def calculate_qparams(self):
        is_uninitialized = (
            (self.min_vals == float('inf')).any() 
            and
            (self.max_vals == float('-inf')).any()
        )
        if is_uninitialized:
            warnings.warn(
                "must run observer before calling calculate_qparams.\
                                    Returning default scale and zero point "
            )

            # function exits early
            return torch.tensor([1.0]).to(self.dev), torch.tensor([0]).to(self.dev)

        assert self.bins == len(self.histograms[0]), (
            "The number of bins in histogram should be equal to the number of bins "
            "supplied while making this observer"
        )

        if self.new_mins.numel() == 0 or self.new_maxs.numel() == 0:

            new_mins, new_maxs = self._non_linear_param_search()
            self.new_mins, self.new_maxs = new_mins, new_maxs
            print("min increase ", (self.new_mins - self.min_vals).mean().item())
            print("max decrease ", (self.new_maxs - self.max_vals).mean().item())
            print("range ", (self.max_vals - self.min_vals).mean().item())

            # from _ObserverBase
            return self._calculate_qparams(new_mins, new_maxs)
        else:
            return self._calculate_qparams(self.new_mins, self.new_maxs)

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        super(HistogramObserver, self)._save_to_state_dict(
            destination, prefix, keep_vars)
        destination[prefix + 'min_val'] = self.min_val
        destination[prefix + 'max_val'] = self.max_val

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        version = local_metadata.get('version', None)

        if version is None or version < 3:
            # if min_val and max_val are not initialized, update their shape
            # to account for the differences between v2 and v3
            min_val_name, max_val_name = prefix + 'min_val', prefix + 'max_val'
            if min_val_name in state_dict:
                if state_dict[min_val_name].shape == torch.Size([0]):
                    state_dict[min_val_name] = torch.tensor(
                        float('inf')).to(self.dev)
            if max_val_name in state_dict:
                if state_dict[max_val_name].shape == torch.Size([0]):
                    state_dict[max_val_name] = torch.tensor(
                        float('-inf')).to(self.dev)

        local_state = ['min_val', 'max_val']
        for name in local_state:
            key = prefix + name
            if key in state_dict:
                val = state_dict[key]
                setattr(self, name, val)
            elif strict:
                missing_keys.append(key)
        super(HistogramObserver, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs
        )


# -----------------------------------------------------------------------------
# QAT BatchNorm2d 
# not actually used since
class BatchNorm2d(torch.nn.BatchNorm2d):
    r"""This is a QAT version of :class:`~torch.nn.BatchNorm2d`.
    """

    def __init__(
        self, 
        num_features, 
        eps=1e-5, 
        momentum=0.1, 
        device=None, 
        dtype=None, 
        qconfig=None
    ) -> None:

        factory_kwargs = {'device': device, 'dtype': dtype}
        super(BatchNorm2d, self).__init__(num_features, **factory_kwargs)
        self.eps = eps
        self.scale = 1.0
        self.zero_point = 0
        assert qconfig, 'qconfig must be provided for QAT module'
        self.qconfig = qconfig
        self.weight_fake_quant = qconfig.activation(
            factory_kwargs=factory_kwargs
        )

    def forward(self, input):
        # just BN with the weight quantized
        qweight = self.weight_fake_quant(self.weight)
        return F.batch_norm(
            input, 
            running_mean = self.running_mean,
            running_var = self.running_var,
            weight=qweight, 
            bias=self.bias, 
            training=self.training
        )


    @classmethod
    def from_float(cls, mod):
        assert hasattr(
            mod, 'qconfig'
        ), 'Input float module must have qconfig defined'
        assert mod.qconfig, 'Input float module must have a valid qconfig'
        
        qconfig = mod.qconfig

        # activation_post_process = mod.activation_post_process
        # # if type(mod) == nni.BNReLU2d:
        # #     mod = mod[0]
        # scale, zero_point = activation_post_process.calculate_qparams()
        new_mod = cls(mod.num_features, mod.eps, qconfig=qconfig)
        new_mod.weight = mod.weight
        new_mod.bias = mod.bias
        new_mod.running_mean = mod.running_mean
        new_mod.running_var = mod.running_var
        return new_mod

def get_qconfig(
    activations=8, weights=8, observer="minmax"
):
    """Get a Pytorch Qconfig for certain bitwidths."""

    OBSERVER_MAPPING = {
        "histogram":EditedHistogramObserver,
        "minmax":MinMaxObserver,
        "moving_average_minmax":MovingAverageMinMaxObserver
    }

    # only change activation observer (for now)
    observer = OBSERVER_MAPPING[observer]

    # configuration from https://arxiv.org/abs/1806.08342
    # per layer/tensor affine for activations
    # per channel affine for weights

    # allow for FP32 for ablation
    if activations == "fp":
        activation_quantizer = torch.nn.Identity
    else:
        activation_quantizer = EditedFakeQuantize.with_args(
            observer=observer,
            reduce_range=False,
            quant_min=BITS_QUANT_RANGE_MAPPING["unsigned"][activations][0],
            quant_max=BITS_QUANT_RANGE_MAPPING["unsigned"][activations][1],
            dtype=torch.quint8,
            qscheme=torch.per_tensor_affine,
        )

    if weights == "fp":
        weight_quantizer = torch.nn.Identity

    # only one option for weight observer
    else:
        weight_quantizer = EditedFakeQuantize.with_args(
            observer=PerChannelMinMaxObserver,
            quant_min=BITS_QUANT_RANGE_MAPPING["signed"][weights][0],
            quant_max=BITS_QUANT_RANGE_MAPPING["signed"][weights][1],
            reduce_range=False,
            dtype=torch.qint8,
            qscheme=torch.per_channel_symmetric,
        )

    
    qconfig = torch.quantization.QConfig(
        activation=activation_quantizer,
        weight=weight_quantizer
    )

    return qconfig




