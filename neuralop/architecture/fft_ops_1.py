# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

from mindspore._checkparam import Validator as validator
from mindspore._checkparam import Rel
from mindspore.ops import functional as F
from mindspore.ops import prim_attr_register, PrimitiveWithInfer
from mindspore import dtype as mstype
from mindspore.ops import op_info_register, AiCPURegOp, DataType
import mindspore.ops as ops
import mindspore as ms

################ FFT1D op ################


class FFT1(PrimitiveWithInfer):
    @prim_attr_register
    def __init__(self):
        """Initialize FFT1"""
        self.init_prim_io_names(
            inputs=[
                'x_re', 'x_im'], outputs=[
                'y_re', 'y_im'])
        self.add_prim_attr("cust_aicpu", "FFT_new")
        pass

    def infer_shape(self, x_re_shape, x_im_shape):
        rank_re = len(x_re_shape)
        rank_im = len(x_im_shape)
        if rank_re != rank_im:
            raise ValueError(
                f"For '{self.name}', input 'x_re' and 'x_im' should have the same shape.")
        for i in range(rank_re):
            if x_re_shape[i] != x_im_shape[i]:
                raise ValueError(
                    f"For '{self.name}', input 'x_re' and 'x_im' should have the same shape.")
        return x_re_shape, x_re_shape

    def infer_dtype(self, x_re_dtype, x_im_dtype):
        if (x_re_dtype != x_im_dtype):
            raise TypeError(
                f"For '{self.name}', dtype of 'x_re' and 'x_im' should be same.")
        if (x_re_dtype != mstype.tensor_type(mstype.float32)
                and x_re_dtype != mstype.tensor_type(mstype.float64)):
            raise TypeError(
                f"For '{self.name}', dtype of 'x_re' and 'x_im' should be float32 or float64.")
        return x_re_dtype, x_re_dtype

    def get_bprop(self):
        ifft1 = IFFT1()
        shape_op = ops.Shape()

        def bprop(x_re, x_im, out, dout):
            real_type = x_re.dtype
            x_shape = shape_op(x_re)
            rank = len(x_shape)
            n = x_shape[rank - 1]
            dx_re, dx_im = ifft1(dout[0], dout[1])
            dx_re = F.cast(dx_re, ms.float32) * n
            dx_re = F.cast(dx_re, real_type)
            dx_im = F.cast(dx_im, ms.float32) * n
            dx_im = F.cast(dx_im, real_type)
            return (dx_re, dx_im)
        return bprop


fft1_op_info = AiCPURegOp("FFT1") .fusion_type("OPAQUE") .input(
    0,
    "x_re",
    "required") .input(
        1,
        "x_im",
        "required") .output(
            0,
            "y_re",
            "required") .output(
                1,
                "y_im",
                "required") .attr(
                    "cust_aicpu",
                    "str") .dtype_format(
                        DataType.F32_Default,
                        DataType.F32_Default,
                        DataType.F32_Default,
                        DataType.F32_Default) .dtype_format(
                            DataType.F64_Default,
                            DataType.F64_Default,
                            DataType.F64_Default,
    DataType.F64_Default) .get_op_info()


@op_info_register(fft1_op_info)
def _fft1_aicpu():
    """FFT1 AiCPU register"""
    return
################ IFFT1D op ################


class IFFT1(PrimitiveWithInfer):
    @prim_attr_register
    def __init__(self):
        """Initialize IFFT1"""
        self.init_prim_io_names(
            inputs=[
                'x_re', 'x_im'], outputs=[
                'y_re', 'y_im'])
        self.add_prim_attr("cust_aicpu", "FFT_new")
        pass

    def infer_shape(self, x_re_shape, x_im_shape):
        rank_re = len(x_re_shape)
        rank_im = len(x_im_shape)
        if rank_re != rank_im:
            raise ValueError(
                f"For '{self.name}', input 'x_re' and 'x_im' should have the same shape.")
        for i in range(rank_re):
            if x_re_shape[i] != x_im_shape[i]:
                raise ValueError(
                    f"For '{self.name}', input 'x_re' and 'x_im' should have the same shape.")
        return x_re_shape, x_re_shape

    def infer_dtype(self, x_re_dtype, x_im_dtype):
        if (x_re_dtype != x_im_dtype):
            raise TypeError(
                f"For '{self.name}', dtype of 'x_re' and 'x_im' should be same.")
        if (x_re_dtype != mstype.tensor_type(mstype.float32)
                and x_re_dtype != mstype.tensor_type(mstype.float64)):
            raise TypeError(
                f"For '{self.name}', dtype of 'x_re' and 'x_im' should be float32 or float64.")
        return x_re_dtype, x_re_dtype

    def get_bprop(self):
        fft1 = FFT1()
        shape_op = ops.Shape()

        def bprop(x_re, x_im, out, dout):
            real_type = x_re.dtype
            x_shape = shape_op(x_re)
            rank = len(x_shape)
            n = x_shape[rank - 1]
            dx_re, dx_im = fft1(dout[0], dout[1])
            dx_re = F.cast(dx_re, ms.float32) / n
            dx_re = F.cast(dx_re, real_type)
            dx_im = F.cast(dx_im, ms.float32) / n
            dx_im = F.cast(dx_im, real_type)
            return (dx_re, dx_im)
        return bprop


ifft1_op_info = AiCPURegOp("IFFT1") .fusion_type("OPAQUE") .input(
    0,
    "x_re",
    "required") .input(
        1,
        "x_im",
        "required") .output(
            0,
            "y_re",
            "required") .output(
                1,
                "y_im",
                "required") .attr(
                    "cust_aicpu",
                    "str") .dtype_format(
                        DataType.F32_Default,
                        DataType.F32_Default,
                        DataType.F32_Default,
                        DataType.F32_Default) .dtype_format(
                            DataType.F64_Default,
                            DataType.F64_Default,
                            DataType.F64_Default,
    DataType.F64_Default) .get_op_info()


@op_info_register(ifft1_op_info)
def _ifft1_aicpu():
    """IFFT1 AiCPU register"""
    return
################ RFFT1D op ################


class RFFT1(PrimitiveWithInfer):
    @prim_attr_register
    def __init__(self, s):
        """Initialize RFFT1"""
        fft_rank = 1
        self.init_prim_io_names(inputs=['x'], outputs=['y_re', 'y_im'])
        self.add_prim_attr("cust_aicpu", "FFT_new")
        self.s = s
        validator.check_value_type("s", s, [tuple], self.name)
        for item in self.s:
            validator.check_value_type("item of s", item, [int], self.name)
        validator.check("size of 's'", len(self.s), "fft rank",
                        fft_rank, Rel.EQ, self.name)
        pass

    def infer_shape(self, x_shape):
        fft_rank = 1
        validator.check(
            "rank of 'x",
            len(x_shape),
            "fft rank",
            fft_rank,
            Rel.GE,
            self.name)
        is_forward = True
        y_shape = x_shape
        for i in range(fft_rank):
            inner_most = (i == fft_rank - 1)
            min_input_dim_length = (
                self.s[i] //
                2 +
                1) if (
                not is_forward and inner_most) else self.s[i]
            input_index = len(x_shape) - fft_rank + i
            if x_shape[input_index] != 0 and x_shape[input_index] < min_input_dim_length:
                raise TypeError(
                    f"For '{self.name}', input 'x' and signal 's' cannot match")
            dim = (
                self.s[i] //
                2 +
                1) if is_forward and inner_most and self.s[i] != 0 else self.s[i]
            y_shape[input_index] = dim
        return y_shape, y_shape

    def infer_dtype(self, x_dtype):
        if (x_dtype != mstype.tensor_type(mstype.float32)
                and x_dtype != mstype.tensor_type(mstype.float64)):
            raise TypeError(
                f"For '{self.name}', dtype of 'x' should be float32 or float64.")
        return x_dtype, x_dtype

    def get_bprop(self):
        s = self.s
        shape = ops.Shape()
        zeros = ops.Zeros()
        ifft1 = IFFT1()

        def bprop(x, out, dout):
            real_type = dout[0].dtype
            shape_x = shape(x)
            rank_x = len(shape_x)
            concat_1 = ops.Concat(rank_x - 1)
            temp_shape = shape_x[0:rank_x - 1] + ((s[0] + 1) // 2 - 1, )
            temp = zeros(temp_shape, mstype.float32)
            dy_real = dout[0]
            dy_imag = dout[1]
            dy_real = F.cast(
                concat_1(
                    (F.cast(
                        dy_real,
                        mstype.float32),
                        temp)),
                mstype.float64)
            dy_imag = F.cast(
                concat_1(
                    (F.cast(
                        dy_imag,
                        mstype.float32),
                        temp)),
                mstype.float64)
            dx = ifft1(dy_real, dy_imag)[0]
            z_dim_1 = shape_x[rank_x - 1] - s[0]
            if z_dim_1:
                temp_dim1_shape = shape_x[0:rank_x - 1] + (z_dim_1, )
                temp_dim1 = zeros(temp_dim1_shape, mstype.float32)
                dx = concat_1((F.cast(dx, mstype.float32), temp_dim1))
            dx = F.cast(dx, mstype.float32)
            dx = dx * s[0]
            dx = F.cast(dx, real_type)
            return (dx,)
        return bprop


rfft1_op_info = AiCPURegOp("RFFT1") .fusion_type("OPAQUE") .input(
    0,
    "x",
    "required") .output(
        0,
        "y_re",
        "required") .output(
            1,
            "y_im",
            "required") .attr(
                "s",
                "listInt") .attr(
                    "cust_aicpu",
                    "str") .dtype_format(
                        DataType.F64_Default,
                        DataType.F64_Default,
                        DataType.F64_Default) .dtype_format(
                            DataType.F32_Default,
                            DataType.F32_Default,
    DataType.F32_Default) .get_op_info()


@op_info_register(rfft1_op_info)
def _rfft1_aicpu():
    """RFFT1 AiCPU register"""
    return
################ IRFFT1D op ################


class IRFFT1(PrimitiveWithInfer):
    @prim_attr_register
    def __init__(self, s):
        """Initialize IRFFT1"""
        fft_rank = 1
        self.init_prim_io_names(inputs=['x_re', 'x_im'], outputs=['y'])
        self.add_prim_attr("cust_aicpu", "FFT_new")
        self.s = s
        validator.check_value_type("s", s, [tuple], self.name)
        for item in self.s:
            validator.check_value_type("item of s", item, [int], self.name)
        validator.check("size of 's'", len(self.s), "fft rank",
                        fft_rank, Rel.EQ, self.name)
        pass

    def infer_shape(self, x_re_shape, x_im_shape):
        rank_re = len(x_re_shape)
        rank_im = len(x_im_shape)
        if rank_re != rank_im:
            raise ValueError(
                f"For '{self.name}', input 'x_re' and 'x_im' should have the same shape.")
        for i in range(rank_re):
            if x_re_shape[i] != x_im_shape[i]:
                raise ValueError(
                    f"For '{self.name}', input 'x_re' and 'x_im' should have the same shape.")
        x_shape = x_re_shape
        fft_rank = 1
        validator.check(
            "rank of 'x",
            len(x_shape),
            "fft rank",
            fft_rank,
            Rel.GE,
            self.name)
        is_forward = False
        y_shape = x_shape
        for i in range(fft_rank):
            inner_most = (i == fft_rank - 1)
            min_input_dim_length = (
                self.s[i] //
                2 +
                1) if (
                not is_forward and inner_most) else self.s[i]
            input_index = len(x_shape) - fft_rank + i
            if x_shape[input_index] != 0 and x_shape[input_index] < min_input_dim_length:
                raise TypeError(
                    f"For '{self.name}', input 'x' and signal 's' cannot match")
            dim = (
                self.s[i] //
                2 +
                1) if is_forward and inner_most and self.s[i] != 0 else self.s[i]
            y_shape[input_index] = dim
        return y_shape

    def infer_dtype(self, x_re_dtype, x_im_dtype):
        if (x_re_dtype != x_im_dtype):
            raise TypeError(
                f"For '{self.name}', dtype of 'x_re' and 'x_im' should be same.")
        if (x_re_dtype != mstype.tensor_type(mstype.float32)
                and x_re_dtype != mstype.tensor_type(mstype.float64)):
            raise TypeError(
                f"For '{self.name}', dtype of 'x_re' and 'x_im' should be float32 or float64.")
        return x_re_dtype

    def get_bprop(self):
        s = self.s
        shape = ops.Shape()
        zeros = ops.Zeros()
        rfft1 = RFFT1(s=s)

        def bprop(x_re, x_im, out, dout):
            real_type = dout.dtype
            shape_x = shape(x_re)
            rank_x = len(shape_x)
            concat_1 = ops.Concat(rank_x - 1)
            factor = s[0] * 0.5
            g_i_real, g_i_imag = rfft1(dout)
            shape_g_i = shape(g_i_real)
            g_i_real = F.cast(g_i_real, mstype.float32) / factor
            g_i_imag = F.cast(g_i_imag, mstype.float32) / factor
            g_i_real_1 = g_i_real[..., 0:1] * 0.5
            g_i_imag_1 = g_i_imag[..., 0:1] * 0.5
            g_i_real_2 = g_i_real[..., 1:(s[0] + 1) // 2]
            g_i_imag_2 = g_i_imag[..., 1:(s[0] + 1) // 2]
            g_i_real_final = concat_1((g_i_real_1, g_i_real_2))
            g_i_imag_final = concat_1((g_i_imag_1, g_i_imag_2))
            if (s[0] % 2 == 0):
                g_i_real_3 = g_i_real[..., s[0] // 2:] * 0.5
                g_i_imag_3 = g_i_imag[..., s[0] // 2:] * 0.5
                g_i_real_final = concat_1((g_i_real_final, g_i_real_3))
                g_i_imag_final = concat_1((g_i_imag_final, g_i_imag_3))
            z_dim_1 = shape_x[rank_x - 1] - shape_g_i[rank_x - 1]
            if z_dim_1:
                temp_dim1_shape = shape_x[0:rank_x - 1] + (z_dim_1, )
                temp_dim1 = zeros(temp_dim1_shape, mstype.float32)
                g_i_real_final = concat_1((g_i_real_final, temp_dim1))
                g_i_imag_final = concat_1((g_i_imag_final, temp_dim1))
            g_i_real_final = F.cast(g_i_real_final, real_type)
            g_i_imag_final = F.cast(g_i_imag_final, real_type)
            return (g_i_real_final, g_i_imag_final)
        return bprop


irfft1_op_info = AiCPURegOp("IRFFT1") .fusion_type("OPAQUE") .input(
    0,
    "x_re",
    "required") .input(
        1,
        "x_im",
        "required") .output(
            0,
            "y",
            "required") .attr(
                "s",
                "listInt") .attr(
                    "cust_aicpu",
                    "str") .dtype_format(
                        DataType.F64_Default,
                        DataType.F64_Default,
                        DataType.F64_Default) .dtype_format(
                            DataType.F32_Default,
                            DataType.F32_Default,
    DataType.F32_Default) .get_op_info()


@op_info_register(irfft1_op_info)
def _irfft1_aicpu():
    """IRFFT1 AiCPU register"""
    return
################ FFT2D op ################


class FFT2(PrimitiveWithInfer):
    @prim_attr_register
    def __init__(self):
        """Initialize FFT2"""
        self.init_prim_io_names(
            inputs=[
                'x_re', 'x_im'], outputs=[
                'y_re', 'y_im'])
        self.add_prim_attr("cust_aicpu", "FFT_new")
        pass

    def infer_shape(self, x_re_shape, x_im_shape):
        rank_re = len(x_re_shape)
        rank_im = len(x_im_shape)
        if rank_re != rank_im:
            raise ValueError(
                f"For '{self.name}', input 'x_re' and 'x_im' should have the same shape.")
        for i in range(rank_re):
            if x_re_shape[i] != x_im_shape[i]:
                raise ValueError(
                    f"For '{self.name}', input 'x_re' and 'x_im' should have the same shape.")
        return x_re_shape, x_re_shape

    def infer_dtype(self, x_re_dtype, x_im_dtype):
        if (x_re_dtype != x_im_dtype):
            raise TypeError(
                f"For '{self.name}', dtype of 'x_re' and 'x_im' should be same.")
        if (x_re_dtype != mstype.tensor_type(mstype.float32)
                and x_re_dtype != mstype.tensor_type(mstype.float64)):
            raise TypeError(
                f"For '{self.name}', dtype of 'x_re' and 'x_im' should be float32 or float64.")
        return x_re_dtype, x_re_dtype

    def get_bprop(self):
        ifft2 = IFFT2()
        shape_op = ops.Shape()

        def bprop(x_re, x_im, out, dout):
            real_type = x_re.dtype
            x_shape = shape_op(x_re)
            rank = len(x_shape)
            n = x_shape[rank - 1] * x_shape[rank - 2]
            dx_re, dx_im = ifft2(dout[0], dout[1])
            dx_re = F.cast(dx_re, ms.float32) * n
            dx_re = F.cast(dx_re, real_type)
            dx_im = F.cast(dx_im, ms.float32) * n
            dx_im = F.cast(dx_im, real_type)
            return (dx_re, dx_im)
        return bprop


fft2_op_info = AiCPURegOp("FFT2") .fusion_type("OPAQUE") .input(
    0,
    "x_re",
    "required") .input(
        1,
        "x_im",
        "required") .output(
            0,
            "y_re",
            "required") .output(
                1,
                "y_im",
                "required") .attr(
                    "cust_aicpu",
                    "str") .dtype_format(
                        DataType.F32_Default,
                        DataType.F32_Default,
                        DataType.F32_Default,
                        DataType.F32_Default) .dtype_format(
                            DataType.F64_Default,
                            DataType.F64_Default,
                            DataType.F64_Default,
    DataType.F64_Default) .get_op_info()


@op_info_register(fft2_op_info)
def _fft2_aicpu():
    """FFT2 AiCPU register"""
    return
################ IFFT2D op ################


class IFFT2(PrimitiveWithInfer):
    @prim_attr_register
    def __init__(self):
        """Initialize IFFT2"""
        self.init_prim_io_names(
            inputs=[
                'x_re', 'x_im'], outputs=[
                'y_re', 'y_im'])
        self.add_prim_attr("cust_aicpu", "FFT_new")
        pass

    def infer_shape(self, x_re_shape, x_im_shape):
        rank_re = len(x_re_shape)
        rank_im = len(x_im_shape)
        if rank_re != rank_im:
            raise ValueError(
                f"For '{self.name}', input 'x_re' and 'x_im' should have the same shape.")
        for i in range(rank_re):
            if x_re_shape[i] != x_im_shape[i]:
                raise ValueError(
                    f"For '{self.name}', input 'x_re' and 'x_im' should have the same shape.")
        return x_re_shape, x_re_shape

    def infer_dtype(self, x_re_dtype, x_im_dtype):
        if (x_re_dtype != x_im_dtype):
            raise TypeError(
                f"For '{self.name}', dtype of 'x_re' and 'x_im' should be same.")
        if (x_re_dtype != mstype.tensor_type(mstype.float32)
                and x_re_dtype != mstype.tensor_type(mstype.float64)):
            raise TypeError(
                f"For '{self.name}', dtype of 'x_re' and 'x_im' should be float32 or float64.")
        return x_re_dtype, x_re_dtype

    def get_bprop(self):
        fft2 = FFT2()
        shape_op = ops.Shape()

        def bprop(x_re, x_im, out, dout):
            real_type = x_re.dtype
            x_shape = shape_op(x_re)
            rank = len(x_shape)
            n = x_shape[rank - 1] * x_shape[rank - 2]
            dx_re, dx_im = fft2(dout[0], dout[1])
            dx_re = F.cast(dx_re, ms.float32) / n
            dx_re = F.cast(dx_re, real_type)
            dx_im = F.cast(dx_im, ms.float32) / n
            dx_im = F.cast(dx_im, real_type)
            return (dx_re, dx_im)
        return bprop


ifft2_op_info = AiCPURegOp("IFFT2") .fusion_type("OPAQUE") .input(
    0,
    "x_re",
    "required") .input(
        1,
        "x_im",
        "required") .output(
            0,
            "y_re",
            "required") .output(
                1,
                "y_im",
                "required") .attr(
                    "cust_aicpu",
                    "str") .dtype_format(
                        DataType.F32_Default,
                        DataType.F32_Default,
                        DataType.F32_Default,
                        DataType.F32_Default) .dtype_format(
                            DataType.F64_Default,
                            DataType.F64_Default,
                            DataType.F64_Default,
    DataType.F64_Default) .get_op_info()


@op_info_register(ifft2_op_info)
def _ifft2_aicpu():
    """IFFT2 AiCPU register"""
    return
################ RFFT2D op ################


class RFFT2(PrimitiveWithInfer):
    @prim_attr_register
    def __init__(self, s):
        """Initialize RFFT2"""
        fft_rank = 2
        self.init_prim_io_names(inputs=['x'], outputs=['y_re', 'y_im'])
        self.add_prim_attr("cust_aicpu", "FFT_new")
        self.s = s
        validator.check_value_type("s", s, [tuple], self.name)
        for item in self.s:
            validator.check_value_type("item of s", item, [int], self.name)
        validator.check("size of 's'", len(self.s), "fft rank",
                        fft_rank, Rel.EQ, self.name)
        pass

    def infer_shape(self, x_shape):
        fft_rank = 2
        validator.check(
            "rank of 'x",
            len(x_shape),
            "fft rank",
            fft_rank,
            Rel.GE,
            self.name)
        is_forward = True
        y_shape = x_shape
        for i in range(fft_rank):
            inner_most = (i == fft_rank - 1)
            min_input_dim_length = (
                self.s[i] //
                2 +
                1) if (
                not is_forward and inner_most) else self.s[i]
            input_index = len(x_shape) - fft_rank + i
            if x_shape[input_index] != 0 and x_shape[input_index] < min_input_dim_length:
                raise TypeError(
                    f"For '{self.name}', input 'x' and signal 's' cannot match")
            dim = (
                self.s[i] //
                2 +
                1) if is_forward and inner_most and self.s[i] != 0 else self.s[i]
            y_shape[input_index] = dim
        return y_shape, y_shape

    def infer_dtype(self, x_dtype):
        if (x_dtype != mstype.tensor_type(mstype.float32)
                and x_dtype != mstype.tensor_type(mstype.float64)):
            raise TypeError(
                f"For '{self.name}', dtype of 'x' should be float32 or float64.")
        return x_dtype, x_dtype

    def get_bprop(self):
        s = self.s
        shape = ops.Shape()
        zeros = ops.Zeros()
        ifft2 = IFFT2()

        def bprop(x, out, dout):
            real_type = dout[0].dtype
            shape_x = shape(x)
            rank_x = len(shape_x)
            concat_2 = ops.Concat(rank_x - 1)
            concat_1 = ops.Concat(rank_x - 2)
            temp_shape = shape_x[0:rank_x - 2] + (s[0], (s[1] + 1) // 2 - 1)
            temp = zeros(temp_shape, mstype.float32)
            dy_real = dout[0]
            dy_imag = dout[1]
            dy_real = F.cast(
                concat_2(
                    (F.cast(
                        dy_real,
                        mstype.float32),
                        temp)),
                mstype.float64)
            dy_imag = F.cast(
                concat_2(
                    (F.cast(
                        dy_imag,
                        mstype.float32),
                        temp)),
                mstype.float64)
            dx = ifft2(dy_real, dy_imag)[0]
            z_dim_1 = shape_x[rank_x - 2] - s[0]
            z_dim_2 = shape_x[rank_x - 1] - s[1]
            if z_dim_2:
                temp_dim2_shape = shape_x[0:rank_x - 2] + (s[0], z_dim_2)
                temp_dim2 = zeros(temp_dim2_shape, mstype.float32)
                dx = concat_2((F.cast(dx, mstype.float32), temp_dim2))
            if z_dim_1:
                temp_dim1_shape = shape_x[0:rank_x -
                                          2] + (z_dim_1, shape_x[rank_x - 1])
                temp_dim1 = zeros(temp_dim1_shape, mstype.float32)
                dx = concat_1((F.cast(dx, mstype.float32), temp_dim1))
            dx = F.cast(dx, mstype.float32)
            dx = dx * s[0] * s[1]
            dx = F.cast(dx, real_type)
            return (dx,)
        return bprop


rfft2_op_info = AiCPURegOp("RFFT2") .fusion_type("OPAQUE") .input(
    0,
    "x",
    "required") .output(
        0,
        "y_re",
        "required") .output(
            1,
            "y_im",
            "required") .attr(
                "s",
                "listInt") .attr(
                    "cust_aicpu",
                    "str") .dtype_format(
                        DataType.F64_Default,
                        DataType.F64_Default,
                        DataType.F64_Default) .dtype_format(
                            DataType.F32_Default,
                            DataType.F32_Default,
    DataType.F32_Default) .get_op_info()


@op_info_register(rfft2_op_info)
def _rfft2_aicpu():
    """RFFT2 AiCPU register"""
    return
################ IRFFT2D op ################


class IRFFT2(PrimitiveWithInfer):
    @prim_attr_register
    def __init__(self, s):
        """Initialize IRFFT2"""
        fft_rank = 2
        self.init_prim_io_names(inputs=['x_re', 'x_im'], outputs=['y'])
        self.add_prim_attr("cust_aicpu", "FFT_new")
        self.s = s
        validator.check_value_type("s", s, [tuple], self.name)
        for item in self.s:
            validator.check_value_type("item of s", item, [int], self.name)
        validator.check("size of 's'", len(self.s), "fft rank",
                        fft_rank, Rel.EQ, self.name)
        pass

    def infer_shape(self, x_re_shape, x_im_shape):
        rank_re = len(x_re_shape)
        rank_im = len(x_im_shape)
        if rank_re != rank_im:
            raise ValueError(
                f"For '{self.name}', input 'x_re' and 'x_im' should have the same shape.")
        for i in range(rank_re):
            if x_re_shape[i] != x_im_shape[i]:
                raise ValueError(
                    f"For '{self.name}', input 'x_re' and 'x_im' should have the same shape.")
        x_shape = x_re_shape
        fft_rank = 2
        validator.check(
            "rank of 'x",
            len(x_shape),
            "fft rank",
            fft_rank,
            Rel.GE,
            self.name)
        is_forward = False
        y_shape = x_shape
        for i in range(fft_rank):
            inner_most = (i == fft_rank - 1)
            min_input_dim_length = (
                self.s[i] //
                2 +
                1) if (
                not is_forward and inner_most) else self.s[i]
            input_index = len(x_shape) - fft_rank + i
            if x_shape[input_index] != 0 and x_shape[input_index] < min_input_dim_length:
                raise TypeError(
                    f"For '{self.name}', input 'x' and signal 's' cannot match")
            dim = (
                self.s[i] //
                2 +
                1) if is_forward and inner_most and self.s[i] != 0 else self.s[i]
            y_shape[input_index] = dim
        return y_shape

    def infer_dtype(self, x_re_dtype, x_im_dtype):
        if (x_re_dtype != x_im_dtype):
            raise TypeError(
                f"For '{self.name}', dtype of 'x_re' and 'x_im' should be same.")
        if (x_re_dtype != mstype.tensor_type(mstype.float32)
                and x_re_dtype != mstype.tensor_type(mstype.float64)):
            raise TypeError(
                f"For '{self.name}', dtype of 'x_re' and 'x_im' should be float32 or float64.")
        return x_re_dtype

    def get_bprop(self):
        s = self.s
        shape = ops.Shape()
        zeros = ops.Zeros()
        rfft2 = RFFT2(s=s)

        def bprop(x_re, x_im, out, dout):
            real_type = dout.dtype
            shape_x = shape(x_re)
            rank_x = len(shape_x)
            concat_2 = ops.Concat(rank_x - 1)
            concat_1 = ops.Concat(rank_x - 2)
            factor = s[0] * s[1] * 0.5
            g_i_real, g_i_imag = rfft2(dout)
            shape_g_i = shape(g_i_real)
            g_i_real = F.cast(g_i_real, mstype.float32) / factor
            g_i_imag = F.cast(g_i_imag, mstype.float32) / factor
            g_i_real_1 = g_i_real[..., 0:1] * 0.5
            g_i_imag_1 = g_i_imag[..., 0:1] * 0.5
            g_i_real_2 = g_i_real[..., 1:(s[1] + 1) // 2]
            g_i_imag_2 = g_i_imag[..., 1:(s[1] + 1) // 2]
            g_i_real_final = concat_2((g_i_real_1, g_i_real_2))
            g_i_imag_final = concat_2((g_i_imag_1, g_i_imag_2))
            if (s[1] % 2 == 0):
                g_i_real_3 = g_i_real[..., s[1] // 2:] * 0.5
                g_i_imag_3 = g_i_imag[..., s[1] // 2:] * 0.5
                g_i_real_final = concat_2((g_i_real_final, g_i_real_3))
                g_i_imag_final = concat_2((g_i_imag_final, g_i_imag_3))
            z_dim_1 = shape_x[rank_x - 2] - shape_g_i[rank_x - 2]
            z_dim_2 = shape_x[rank_x - 1] - shape_g_i[rank_x - 1]
            if z_dim_2:
                temp_dim2_shape = shape_x[0:rank_x -
                                          2] + (shape_g_i[rank_x - 2], z_dim_2)
                temp_dim2 = zeros(temp_dim2_shape, mstype.float32)
                g_i_real_final = concat_2((g_i_real_final, temp_dim2))
                g_i_imag_final = concat_2((g_i_imag_final, temp_dim2))
            if z_dim_1:
                temp_dim1_shape = shape_x[0:rank_x -
                                          2] + (z_dim_1, shape_x[rank_x - 1])
                temp_dim1 = zeros(temp_dim1_shape, mstype.float32)
                g_i_real_final = concat_1((g_i_real_final, temp_dim1))
                g_i_imag_final = concat_1((g_i_imag_final, temp_dim1))
            g_i_real_final = F.cast(g_i_real_final, real_type)
            g_i_imag_final = F.cast(g_i_imag_final, real_type)
            return (g_i_real_final, g_i_imag_final)
        return bprop


irfft2_op_info = AiCPURegOp("IRFFT2") .fusion_type("OPAQUE") .input(
    0,
    "x_re",
    "required") .input(
        1,
        "x_im",
        "required") .output(
            0,
            "y",
            "required") .attr(
                "s",
                "listInt") .attr(
                    "cust_aicpu",
                    "str") .dtype_format(
                        DataType.F64_Default,
                        DataType.F64_Default,
                        DataType.F64_Default) .dtype_format(
                            DataType.F32_Default,
                            DataType.F32_Default,
    DataType.F32_Default) .get_op_info()


@op_info_register(irfft2_op_info)
def _irfft2_aicpu():
    """IRFFT2 AiCPU register"""
    return
################ FFT3D op ################


class FFT3(PrimitiveWithInfer):
    @prim_attr_register
    def __init__(self):
        """Initialize FFT3"""
        self.init_prim_io_names(
            inputs=[
                'x_re', 'x_im'], outputs=[
                'y_re', 'y_im'])
        self.add_prim_attr("cust_aicpu", "FFT_new")
        pass

    def infer_shape(self, x_re_shape, x_im_shape):
        rank_re = len(x_re_shape)
        rank_im = len(x_im_shape)
        if rank_re != rank_im:
            raise ValueError(
                f"For '{self.name}', input 'x_re' and 'x_im' should have the same shape.")
        for i in range(rank_re):
            if x_re_shape[i] != x_im_shape[i]:
                raise ValueError(
                    f"For '{self.name}', input 'x_re' and 'x_im' should have the same shape.")
        return x_re_shape, x_re_shape

    def infer_dtype(self, x_re_dtype, x_im_dtype):
        if (x_re_dtype != x_im_dtype):
            raise TypeError(
                f"For '{self.name}', dtype of 'x_re' and 'x_im' should be same.")
        if (x_re_dtype != mstype.tensor_type(mstype.float32)
                and x_re_dtype != mstype.tensor_type(mstype.float64)):
            raise TypeError(
                f"For '{self.name}', dtype of 'x_re' and 'x_im' should be float32 or float64.")
        return x_re_dtype, x_re_dtype

    def get_bprop(self):
        ifft3 = IFFT3()
        shape_op = ops.Shape()

        def bprop(x_re, x_im, out, dout):
            real_type = x_re.dtype
            x_shape = shape_op(x_re)
            rank = len(x_shape)
            n = x_shape[rank - 1] * x_shape[rank - 2] * x_shape[rank - 3]
            dx_re, dx_im = ifft3(dout[0], dout[1])
            dx_re = F.cast(dx_re, ms.float32) * n
            dx_re = F.cast(dx_re, real_type)
            dx_im = F.cast(dx_im, ms.float32) * n
            dx_im = F.cast(dx_im, real_type)
            return (dx_re, dx_im)
        return bprop


fft3_op_info = AiCPURegOp("FFT3") .fusion_type("OPAQUE") .input(
    0,
    "x_re",
    "required") .input(
        1,
        "x_im",
        "required") .output(
            0,
            "y_re",
            "required") .output(
                1,
                "y_im",
                "required") .attr(
                    "cust_aicpu",
                    "str") .dtype_format(
                        DataType.F32_Default,
                        DataType.F32_Default,
                        DataType.F32_Default,
                        DataType.F32_Default) .dtype_format(
                            DataType.F64_Default,
                            DataType.F64_Default,
                            DataType.F64_Default,
    DataType.F64_Default) .get_op_info()


@op_info_register(fft3_op_info)
def _fft3_aicpu():
    """FFT3 AiCPU register"""
    return
################ IFFT3D op ################


class IFFT3(PrimitiveWithInfer):
    @prim_attr_register
    def __init__(self):
        """Initialize IFFT2"""
        self.init_prim_io_names(
            inputs=[
                'x_re', 'x_im'], outputs=[
                'y_re', 'y_im'])
        self.add_prim_attr("cust_aicpu", "FFT_new")
        pass

    def infer_shape(self, x_re_shape, x_im_shape):
        rank_re = len(x_re_shape)
        rank_im = len(x_im_shape)
        if rank_re != rank_im:
            raise ValueError(
                f"For '{self.name}', input 'x_re' and 'x_im' should have the same shape.")
        for i in range(rank_re):
            if x_re_shape[i] != x_im_shape[i]:
                raise ValueError(
                    f"For '{self.name}', input 'x_re' and 'x_im' should have the same shape.")
        return x_re_shape, x_re_shape

    def infer_dtype(self, x_re_dtype, x_im_dtype):
        if (x_re_dtype != x_im_dtype):
            raise TypeError(
                f"For '{self.name}', dtype of 'x_re' and 'x_im' should be same.")
        if (x_re_dtype != mstype.tensor_type(mstype.float32)
                and x_re_dtype != mstype.tensor_type(mstype.float64)):
            raise TypeError(
                f"For '{self.name}', dtype of 'x_re' and 'x_im' should be float32 or float64.")
        return x_re_dtype, x_re_dtype

    def get_bprop(self):
        fft3 = FFT3()
        shape_op = ops.Shape()

        def bprop(x_re, x_im, out, dout):
            real_type = x_re.dtype
            x_shape = shape_op(x_re)
            rank = len(x_shape)
            n = x_shape[rank - 1] * x_shape[rank - 2] * x_shape[rank - 3]
            dx_re, dx_im = fft3(dout[0], dout[1])
            dx_re = F.cast(dx_re, ms.float32) / n
            dx_re = F.cast(dx_re, real_type)
            dx_im = F.cast(dx_im, ms.float32) / n
            dx_im = F.cast(dx_im, real_type)
            return (dx_re, dx_im)
        return bprop


ifft3_op_info = AiCPURegOp("IFFT3") .fusion_type("OPAQUE") .input(
    0,
    "x_re",
    "required") .input(
        1,
        "x_im",
        "required") .output(
            0,
            "y_re",
            "required") .output(
                1,
                "y_im",
                "required") .attr(
                    "cust_aicpu",
                    "str") .dtype_format(
                        DataType.F32_Default,
                        DataType.F32_Default,
                        DataType.F32_Default,
                        DataType.F32_Default) .dtype_format(
                            DataType.F64_Default,
                            DataType.F64_Default,
                            DataType.F64_Default,
    DataType.F64_Default) .get_op_info()


@op_info_register(ifft3_op_info)
def _ifft3_aicpu():
    """IFFT3 AiCPU register"""
    return
################ RFFT3D op ################


class RFFT3(PrimitiveWithInfer):
    @prim_attr_register
    def __init__(self, s):
        """Initialize RFFT3"""
        fft_rank = 3
        self.init_prim_io_names(inputs=['x'], outputs=['y_re', 'y_im'])
        self.add_prim_attr("cust_aicpu", "FFT_new")
        self.s = s
        validator.check_value_type("s", s, [tuple], self.name)
        for item in self.s:
            validator.check_value_type("item of s", item, [int], self.name)
        validator.check("size of 's'", len(self.s), "fft rank",
                        fft_rank, Rel.EQ, self.name)
        pass

    def infer_shape(self, x_shape):
        fft_rank = 3
        validator.check(
            "rank of 'x",
            len(x_shape),
            "fft rank",
            fft_rank,
            Rel.GE,
            self.name)
        is_forward = True
        y_shape = x_shape
        for i in range(fft_rank):
            inner_most = (i == fft_rank - 1)
            min_input_dim_length = (
                self.s[i] //
                2 +
                1) if (
                not is_forward and inner_most) else self.s[i]
            input_index = len(x_shape) - fft_rank + i
            if x_shape[input_index] != 0 and x_shape[input_index] < min_input_dim_length:
                raise TypeError(
                    f"For '{self.name}', input 'x' and signal 's' cannot match")
            dim = (
                self.s[i] //
                2 +
                1) if is_forward and inner_most and self.s[i] != 0 else self.s[i]
            y_shape[input_index] = dim
        return y_shape, y_shape

    def infer_dtype(self, x_dtype):
        if (x_dtype != mstype.tensor_type(mstype.float32)
                and x_dtype != mstype.tensor_type(mstype.float64)):
            raise TypeError(
                f"For '{self.name}', dtype of 'x' should be float32 or float64.")
        return x_dtype, x_dtype

    def get_bprop(self):
        s = self.s
        shape = ops.Shape()
        zeros = ops.Zeros()
        ifft3 = IFFT3()

        def bprop(x, out, dout):
            real_type = dout[0].dtype
            shape_x = shape(x)
            rank_x = len(shape_x)
            concat_3 = ops.Concat(rank_x - 1)
            concat_2 = ops.Concat(rank_x - 2)
            concat_1 = ops.Concat(rank_x - 3)
            temp_shape = shape_x[0:rank_x - 3] + \
                (s[0], s[1], (s[2] + 1) // 2 - 1)
            temp = zeros(temp_shape, mstype.float32)
            dy_real = dout[0]
            dy_imag = dout[1]
            dy_real = F.cast(
                concat_3(
                    (F.cast(
                        dy_real,
                        mstype.float32),
                        temp)),
                mstype.float64)
            dy_imag = F.cast(
                concat_3(
                    (F.cast(
                        dy_imag,
                        mstype.float32),
                        temp)),
                mstype.float64)
            dx = ifft3(dy_real, dy_imag)[0]
            z_dim_1 = shape_x[rank_x - 3] - s[0]
            z_dim_2 = shape_x[rank_x - 2] - s[1]
            z_dim_3 = shape_x[rank_x - 1] - s[2]
            if z_dim_3:
                temp_dim3_shape = shape_x[0:rank_x - 3] + (s[0], s[1], z_dim_3)
                temp_dim3 = zeros(temp_dim3_shape, mstype.float32)
                dx = concat_3((F.cast(dx, mstype.float32), temp_dim3))
            if z_dim_2:
                temp_dim2_shape = shape_x[0:rank_x - 3] + \
                    (s[0], z_dim_2, shape_x[rank_x - 1])
                temp_dim2 = zeros(temp_dim2_shape, mstype.float32)
                dx = concat_2((F.cast(dx, mstype.float32), temp_dim2))
            if z_dim_1:
                temp_dim1_shape = shape_x[0:rank_x - 3] + \
                    (z_dim_1, shape_x[rank_x - 2], shape_x[rank_x - 1])
                temp_dim1 = zeros(temp_dim1_shape, mstype.float32)
                dx = concat_1((F.cast(dx, mstype.float32), temp_dim1))
            dx = F.cast(dx, mstype.float32)
            dx = dx * s[0] * s[1] * s[2]
            dx = F.cast(dx, real_type)
            return (dx,)
        return bprop


rfft3_op_info = AiCPURegOp("RFFT3") .fusion_type("OPAQUE") .input(
    0,
    "x",
    "required") .output(
        0,
        "y_re",
        "required") .output(
            1,
            "y_im",
            "required") .attr(
                "s",
                "listInt") .attr(
                    "cust_aicpu",
                    "str") .dtype_format(
                        DataType.F64_Default,
                        DataType.F64_Default,
                        DataType.F64_Default) .dtype_format(
                            DataType.F32_Default,
                            DataType.F32_Default,
    DataType.F32_Default) .get_op_info()


@op_info_register(rfft3_op_info)
def _rfft3_aicpu():
    """RFFT3 AiCPU register"""
    return
################ IRFFT3D op ################


class IRFFT3(PrimitiveWithInfer):
    @prim_attr_register
    def __init__(self, s):
        """Initialize IRFFT3"""
        fft_rank = 3
        self.init_prim_io_names(inputs=['x_re', 'x_im'], outputs=['y'])
        self.add_prim_attr("cust_aicpu", "FFT_new")
        self.s = s
        validator.check_value_type("s", s, [tuple], self.name)
        for item in self.s:
            validator.check_value_type("item of s", item, [int], self.name)
        validator.check("size of 's'", len(self.s), "fft rank",
                        fft_rank, Rel.EQ, self.name)
        pass

    def infer_shape(self, x_re_shape, x_im_shape):
        rank_re = len(x_re_shape)
        rank_im = len(x_im_shape)
        if rank_re != rank_im:
            raise ValueError(
                f"For '{self.name}', input 'x_re' and 'x_im' should have the same shape.")
        for i in range(rank_re):
            if x_re_shape[i] != x_im_shape[i]:
                raise ValueError(
                    f"For '{self.name}', input 'x_re' and 'x_im' should have the same shape.")
        x_shape = x_re_shape
        fft_rank = 3
        validator.check(
            "rank of 'x",
            len(x_shape),
            "fft rank",
            fft_rank,
            Rel.GE,
            self.name)
        is_forward = False
        y_shape = x_shape
        for i in range(fft_rank):
            inner_most = (i == fft_rank - 1)
            min_input_dim_length = (
                self.s[i] //
                2 +
                1) if (
                not is_forward and inner_most) else self.s[i]
            input_index = len(x_shape) - fft_rank + i
            if x_shape[input_index] != 0 and x_shape[input_index] < min_input_dim_length:
                raise TypeError(
                    f"For '{self.name}', input 'x' and signal 's' cannot match")
            dim = (
                self.s[i] //
                2 +
                1) if is_forward and inner_most and self.s[i] != 0 else self.s[i]
            y_shape[input_index] = dim
        return y_shape

    def infer_dtype(self, x_re_dtype, x_im_dtype):
        if (x_re_dtype != x_im_dtype):
            raise TypeError(
                f"For '{self.name}', dtype of 'x_re' and 'x_im' should be same.")
        if (x_re_dtype != mstype.tensor_type(mstype.float32)
                and x_re_dtype != mstype.tensor_type(mstype.float64)):
            raise TypeError(
                f"For '{self.name}', dtype of 'x_re' and 'x_im' should be float32 or float64.")
        return x_re_dtype

    def get_bprop(self):
        s = self.s
        shape = ops.Shape()
        zeros = ops.Zeros()
        rfft3 = RFFT3(s=s)

        def bprop(x_re, x_im, out, dout):
            real_type = dout.dtype
            shape_x = shape(x_re)
            rank_x = len(shape_x)
            concat_3 = ops.Concat(rank_x - 1)
            concat_2 = ops.Concat(rank_x - 2)
            concat_1 = ops.Concat(rank_x - 3)
            factor = s[0] * s[1] * s[2] * 0.5
            g_i_real, g_i_imag = rfft3(dout)
            shape_g_i = shape(g_i_real)
            g_i_real = F.cast(g_i_real, mstype.float32) / factor
            g_i_imag = F.cast(g_i_imag, mstype.float32) / factor
            g_i_real_1 = g_i_real[..., 0:1] * 0.5
            g_i_imag_1 = g_i_imag[..., 0:1] * 0.5
            g_i_real_2 = g_i_real[..., 1:(s[2] + 1) // 2]
            g_i_imag_2 = g_i_imag[..., 1:(s[2] + 1) // 2]
            g_i_real_final = concat_3((g_i_real_1, g_i_real_2))
            g_i_imag_final = concat_3((g_i_imag_1, g_i_imag_2))
            if (s[2] % 2 == 0):
                g_i_real_3 = g_i_real[..., s[2] // 2:] * 0.5
                g_i_imag_3 = g_i_imag[..., s[2] // 2:] * 0.5
                g_i_real_final = concat_3((g_i_real_final, g_i_real_3))
                g_i_imag_final = concat_3((g_i_imag_final, g_i_imag_3))
            z_dim_1 = shape_x[rank_x - 3] - shape_g_i[rank_x - 3]
            z_dim_2 = shape_x[rank_x - 2] - shape_g_i[rank_x - 2]
            z_dim_3 = shape_x[rank_x - 1] - shape_g_i[rank_x - 1]
            if z_dim_3:
                temp_dim3_shape = shape_x[0:rank_x - 3] + \
                    (shape_g_i[rank_x - 3], shape_g_i[rank_x - 2], z_dim_3)
                temp_dim3 = zeros(temp_dim3_shape, mstype.float32)
                g_i_real_final = concat_3((g_i_real_final, temp_dim3))
                g_i_imag_final = concat_3((g_i_imag_final, temp_dim3))
            if z_dim_2:
                temp_dim2_shape = shape_x[0:rank_x - 3] + \
                    (shape_g_i[rank_x - 3], z_dim_2, shape_x[rank_x - 1])
                temp_dim2 = zeros(temp_dim2_shape, mstype.float32)
                g_i_real_final = concat_2((g_i_real_final, temp_dim2))
                g_i_imag_final = concat_2((g_i_imag_final, temp_dim2))
            if z_dim_1:
                temp_dim1_shape = shape_x[0:rank_x - 3] + \
                    (z_dim_1, shape_x[rank_x - 2], shape_x[rank_x - 1])
                temp_dim1 = zeros(temp_dim1_shape, mstype.float32)
                g_i_real_final = concat_1((g_i_real_final, temp_dim1))
                g_i_imag_final = concat_1((g_i_imag_final, temp_dim1))
            g_i_real_final = F.cast(g_i_real_final, real_type)
            g_i_imag_final = F.cast(g_i_imag_final, real_type)
            return (g_i_real_final, g_i_imag_final)
        return bprop


irfft3_op_info = AiCPURegOp("IRFFT3") .fusion_type("OPAQUE") .input(
    0,
    "x_re",
    "required") .input(
        1,
        "x_im",
        "required") .output(
            0,
            "y",
            "required") .attr(
                "s",
                "listInt") .attr(
                    "cust_aicpu",
                    "str") .dtype_format(
                        DataType.F64_Default,
                        DataType.F64_Default,
                        DataType.F64_Default) .dtype_format(
                            DataType.F32_Default,
                            DataType.F32_Default,
    DataType.F32_Default) .get_op_info()


@op_info_register(irfft3_op_info)
def _irfft3_aicpu():
    """IRFFT3 AiCPU register"""
    return
################ IFFT1D_SP op ################


class IFFT1_SP(PrimitiveWithInfer):
    @prim_attr_register
    def __init__(self, s, origin):
        """Initialize IFFT1_SP"""
        self.init_prim_io_names(inputs=['x_re', 'x_im'], outputs=['y_re'])
        self.add_prim_attr("cust_aicpu", "FFT_new")
        self.origin = origin
        pass

    def infer_shape(self, shape1, shape2):
        fft_rank = 1
        y_shape = shape1
        for i in range(fft_rank):
            input_index = len(y_shape) - fft_rank + i
            y_shape[input_index] = self.origin[i]
        return y_shape

    def infer_dtype(self, dtype1, dtype2):
        return dtype1


ifft1_sp_op_info = AiCPURegOp("IFFT1_SP") .fusion_type("OPAQUE") .input(
    0,
    "x_re",
    "required") .input(
        1,
        "x_im",
        "required") .output(
            0,
            "y_re",
            "required") .attr(
                "origin",
                "listInt") .attr(
                    "s",
                    "listInt") .attr(
                        "cust_aicpu",
                        "str") .dtype_format(
                            DataType.F32_Default,
                            DataType.F32_Default,
                            DataType.F32_Default) .dtype_format(
                                DataType.F64_Default,
                                DataType.F64_Default,
    DataType.F64_Default) .get_op_info()


@op_info_register(ifft1_sp_op_info)
def _ifft1_sp_aicpu():
    """IFFT1_SP AiCPU register"""
    return
################ RFFT1D_SP op ################


class RFFT1_SP(PrimitiveWithInfer):
    @prim_attr_register
    def __init__(self, s, modes):
        """Initialize RFFT1_SP"""
        fft_rank = 1
        self.init_prim_io_names(inputs=['x'], outputs=['y_re', 'y_im'])
        self.add_prim_attr("cust_aicpu", "FFT_new")
        self.s = s
        self.modes = modes
        validator.check_value_type("s", s, [tuple], self.name)
        validator.check_value_type("modes", modes, [tuple], self.name)
        for item in self.s:
            validator.check_value_type("item of s", item, [int], self.name)
        for item in self.modes:
            validator.check_value_type("item of modes", item, [int], self.name)
        validator.check("size of 's'", len(self.s), "fft rank",
                        fft_rank, Rel.EQ, self.name)
        validator.check("size of 'modes'", len(self.modes),
                        "fft rank", fft_rank, Rel.EQ, self.name)
        pass

    def infer_shape(self, x_shape):
        fft_rank = 1
        validator.check(
            "rank of 'x",
            len(x_shape),
            "fft rank",
            fft_rank,
            Rel.GE,
            self.name)
        is_forward = True
        y_shape = x_shape
        for i in range(fft_rank):
            inner_most = (i == fft_rank - 1)
            min_input_dim_length = (
                self.s[i] //
                2 +
                1) if (
                not is_forward and inner_most) else self.s[i]
            input_index = len(x_shape) - fft_rank + i
            if x_shape[input_index] != 0 and x_shape[input_index] < min_input_dim_length:
                raise TypeError(
                    f"For '{self.name}', input 'x' and signal 's' cannot match")
            dim = (
                self.s[i] //
                2 +
                1) if is_forward and inner_most and self.s[i] != 0 else self.s[i]
            y_shape[input_index] = self.modes[i]
        return y_shape, y_shape

    def infer_dtype(self, x_dtype):
        if (x_dtype != mstype.tensor_type(mstype.float32)
                and x_dtype != mstype.tensor_type(mstype.float64)):
            raise TypeError(
                f"For '{self.name}', dtype of 'x' should be float32 or float64.")
        return x_dtype, x_dtype

    def get_bprop(self):
        s = self.s
        shape = ops.Shape()

        def bprop(x, out, dout):
            x_shape = shape(x)
            x_rank = len(x_shape)
            origin = (x_shape[x_rank - 1], )
            ifft1_sp = IFFT1_SP(s=s, origin=origin)
            dx = ifft1_sp(dout[0], dout[1])
            return (dx,)
        return bprop


rfft1_sp_op_info = AiCPURegOp("RFFT1_SP") .fusion_type("OPAQUE") .input(
    0,
    "x",
    "required") .output(
        0,
        "y_re",
        "required") .output(
            1,
            "y_im",
            "required") .attr(
                "modes",
                "listInt") .attr(
                    "s",
                    "listInt") .attr(
                        "cust_aicpu",
                        "str") .dtype_format(
                            DataType.F64_Default,
                            DataType.F64_Default,
                            DataType.F64_Default) .dtype_format(
                                DataType.F32_Default,
                                DataType.F32_Default,
    DataType.F32_Default) .get_op_info()


@op_info_register(rfft1_sp_op_info)
def _rfft1_sp_aicpu():
    """RFFT1_SP AiCPU register"""
    return
################ IFFT2D_SP op ################


class IFFT2_SP(PrimitiveWithInfer):
    @prim_attr_register
    def __init__(self, s, origin):
        """Initialize IFFT2_SP"""
        self.init_prim_io_names(
            inputs=[
                'x_re1',
                'x_re2',
                'x_im1',
                'x_im2'],
            outputs=['y_re'])
        self.add_prim_attr("cust_aicpu", "FFT_new")
        self.origin = origin
        pass

    def infer_shape(self, shape1, shape2, shape3, shape4):
        fft_rank = 2
        y_shape = shape1
        for i in range(fft_rank):
            input_index = len(y_shape) - fft_rank + i
            y_shape[input_index] = self.origin[i]
        return y_shape

    def infer_dtype(self, dtype1, dtype2, dtype3, dtype4):
        return dtype1


ifft2_sp_op_info = AiCPURegOp("IFFT2_SP") .fusion_type("OPAQUE") .input(
    0,
    "x_re1",
    "required") .input(
        1,
        "x_re2",
        "required") .input(
            2,
            "x_im1",
            "required") .input(
                3,
                "x_im2",
                "required") .output(
                    0,
                    "y_re",
                    "required") .attr(
                        "origin",
                        "listInt") .attr(
                            "s",
                            "listInt") .attr(
                                "cust_aicpu",
                                "str") .dtype_format(
                                    DataType.F32_Default,
                                    DataType.F32_Default,
                                    DataType.F32_Default,
                                    DataType.F32_Default,
                                    DataType.F32_Default) .dtype_format(
                                        DataType.F64_Default,
                                        DataType.F64_Default,
                                        DataType.F64_Default,
                                        DataType.F64_Default,
    DataType.F64_Default) .get_op_info()


@op_info_register(ifft2_sp_op_info)
def _ifft2_sp_aicpu():
    """IFFT2_SP AiCPU register"""
    return
################ RFFT2D_SP op ################


class RFFT2_SP(PrimitiveWithInfer):
    @prim_attr_register
    def __init__(self, s, modes):
        """Initialize RFFT2_SP"""
        fft_rank = 2
        self.init_prim_io_names(
            inputs=['x'], outputs=[
                'y_re1', 'y_re2', 'y_im1', 'y_im2'])
        self.add_prim_attr("cust_aicpu", "FFT_new")
        self.s = s
        self.modes = modes
        validator.check_value_type("s", s, [tuple], self.name)
        validator.check_value_type("modes", modes, [tuple], self.name)
        for item in self.s:
            validator.check_value_type("item of s", item, [int], self.name)
        for item in self.modes:
            validator.check_value_type("item of modes", item, [int], self.name)
        validator.check("size of 's'", len(self.s), "fft rank",
                        fft_rank, Rel.EQ, self.name)
        validator.check("size of 'modes'", len(self.modes),
                        "fft rank", fft_rank, Rel.EQ, self.name)
        pass

    def infer_shape(self, x_shape):
        fft_rank = 2
        validator.check(
            "rank of 'x",
            len(x_shape),
            "fft rank",
            fft_rank,
            Rel.GE,
            self.name)
        is_forward = True
        y_shape = x_shape
        for i in range(fft_rank):
            inner_most = (i == fft_rank - 1)
            min_input_dim_length = (
                self.s[i] //
                2 +
                1) if (
                not is_forward and inner_most) else self.s[i]
            input_index = len(x_shape) - fft_rank + i
            if x_shape[input_index] != 0 and x_shape[input_index] < min_input_dim_length:
                raise TypeError(
                    f"For '{self.name}', input 'x' and signal 's' cannot match")
            dim = (
                self.s[i] //
                2 +
                1) if is_forward and inner_most and self.s[i] != 0 else self.s[i]
            y_shape[input_index] = self.modes[i]
        return y_shape, y_shape, y_shape, y_shape

    def infer_dtype(self, x_dtype):
        if (x_dtype != mstype.tensor_type(mstype.float32)
                and x_dtype != mstype.tensor_type(mstype.float64)):
            raise TypeError(
                f"For '{self.name}', dtype of 'x' should be float32 or float64.")
        return x_dtype, x_dtype, x_dtype, x_dtype

    def get_bprop(self):
        s = self.s
        shape = ops.Shape()

        def bprop(x, out, dout):
            x_shape = shape(x)
            x_rank = len(x_shape)
            origin = (x_shape[x_rank - 2], x_shape[x_rank - 1])
            ifft2_sp = IFFT2_SP(s=s, origin=origin)
            dx = ifft2_sp(dout[0], dout[1], dout[2], dout[3])
            return (dx,)
        return bprop


rfft2_sp_op_info = AiCPURegOp("RFFT2_SP") .fusion_type("OPAQUE") .input(
    0,
    "x",
    "required") .output(
        0,
        "y_re1",
        "required") .output(
            1,
            "y_re2",
            "required") .output(
                2,
                "y_im1",
                "required") .output(
                    3,
                    "y_im2",
                    "required") .attr(
                        "modes",
                        "listInt") .attr(
                            "s",
                            "listInt") .attr(
                                "cust_aicpu",
                                "str") .dtype_format(
                                    DataType.F64_Default,
                                    DataType.F64_Default,
                                    DataType.F64_Default,
                                    DataType.F64_Default,
                                    DataType.F64_Default) .dtype_format(
                                        DataType.F32_Default,
                                        DataType.F32_Default,
                                        DataType.F32_Default,
                                        DataType.F32_Default,
    DataType.F32_Default) .get_op_info()


@op_info_register(rfft2_sp_op_info)
def _rfft2_sp_aicpu():
    """RFFT2_SP AiCPU register"""
    return
################ IFFT3D_SP op ################


class IFFT3_SP(PrimitiveWithInfer):
    @prim_attr_register
    def __init__(self, s, origin):
        """Initialize IFFT3_SP"""
        self.init_prim_io_names(
            inputs=[
                'x_re1',
                'x_re2',
                'x_re3',
                'x_re4',
                'x_im1',
                'x_im2',
                'x_im3',
                'x_im4'],
            outputs=['y_re'])
        self.add_prim_attr("cust_aicpu", "FFT_new")
        self.origin = origin
        pass

    def infer_shape(
            self,
            shape1,
            shape2,
            shape3,
            shape4,
            shape5,
            shape6,
            shape7,
            shape8):
        fft_rank = 3
        y_shape = shape1
        for i in range(fft_rank):
            input_index = len(y_shape) - fft_rank + i
            y_shape[input_index] = self.origin[i]
        return y_shape

    def infer_dtype(
            self,
            dtype1,
            dtype2,
            dtype3,
            dtype4,
            dtype5,
            dtype6,
            dtype7,
            dtype8):
        return dtype1


ifft3_sp_op_info = AiCPURegOp("IFFT3_SP") \
    .fusion_type("OPAQUE") \
    .input(0, "x_re1", "required") \
    .input(1, "x_re2", "required") \
    .input(2, "x_re3", "required") \
    .input(3, "x_re4", "required") \
    .input(4, "x_im1", "required") \
    .input(5, "x_im2", "required") \
    .input(6, "x_im3", "required") \
    .input(7, "x_im4", "required") \
    .output(0, "y_re", "required") \
    .attr("origin", "listInt") \
    .attr("s", "listInt") \
    .attr("cust_aicpu", "str") \
    .dtype_format(DataType.F32_Default, DataType.F32_Default, DataType.F32_Default,
                  DataType.F32_Default, DataType.F32_Default, DataType.F32_Default,
                  DataType.F32_Default, DataType.F32_Default, DataType.F32_Default) \
    .dtype_format(DataType.F64_Default, DataType.F64_Default, DataType.F64_Default,
                  DataType.F64_Default, DataType.F64_Default, DataType.F64_Default,
                  DataType.F64_Default, DataType.F64_Default, DataType.F64_Default) \
    .get_op_info()


@op_info_register(ifft3_sp_op_info)
def _ifft3_sp_aicpu():
    """IFFT3_SP AiCPU register"""
    return
################ RFFT3D_SP op ################


class RFFT3_SP(PrimitiveWithInfer):
    @prim_attr_register
    def __init__(self, s, modes):
        """Initialize RFFT3_SP"""
        fft_rank = 3
        self.init_prim_io_names(
            inputs=['x'],
            outputs=[
                'y_re1',
                'y_re2',
                'y_re3',
                'y_re4',
                'y_im1',
                'y_im2',
                'y_im3',
                'y_im4'])
        self.add_prim_attr("cust_aicpu", "FFT_new")
        self.s = s
        self.modes = modes
        validator.check_value_type("s", s, [tuple], self.name)
        validator.check_value_type("modes", modes, [tuple], self.name)
        for item in self.s:
            validator.check_value_type("item of s", item, [int], self.name)
        for item in self.modes:
            validator.check_value_type("item of modes", item, [int], self.name)
        validator.check("size of 's'", len(self.s), "fft rank",
                        fft_rank, Rel.EQ, self.name)
        validator.check("size of 'modes'", len(self.modes),
                        "fft rank", fft_rank, Rel.EQ, self.name)
        pass

    def infer_shape(self, x_shape):
        fft_rank = 3
        validator.check(
            "rank of 'x",
            len(x_shape),
            "fft rank",
            fft_rank,
            Rel.GE,
            self.name)
        is_forward = True
        y_shape = x_shape
        for i in range(fft_rank):
            inner_most = (i == fft_rank - 1)
            min_input_dim_length = (
                self.s[i] //
                2 +
                1) if (
                not is_forward and inner_most) else self.s[i]
            input_index = len(x_shape) - fft_rank + i
            if x_shape[input_index] != 0 and x_shape[input_index] < min_input_dim_length:
                raise TypeError(
                    f"For '{self.name}', input 'x' and signal 's' cannot match")
            dim = (
                self.s[i] //
                2 +
                1) if is_forward and inner_most and self.s[i] != 0 else self.s[i]
            y_shape[input_index] = self.modes[i]
        return y_shape, y_shape, y_shape, y_shape, y_shape, y_shape, y_shape, y_shape

    def infer_dtype(self, x_dtype):
        if (x_dtype != mstype.tensor_type(mstype.float32)
                and x_dtype != mstype.tensor_type(mstype.float64)):
            raise TypeError(
                f"For '{self.name}', dtype of 'x' should be float32 or float64.")
        return x_dtype, x_dtype, x_dtype, x_dtype, x_dtype, x_dtype, x_dtype, x_dtype

    def get_bprop(self):
        s = self.s
        shape = ops.Shape()

        def bprop(x, out, dout):
            x_shape = shape(x)
            x_rank = len(x_shape)
            origin = (x_shape[x_rank - 3],
                      x_shape[x_rank - 2],
                      x_shape[x_rank - 1])
            ifft3_sp = IFFT3_SP(s=s, origin=origin)
            dx = ifft3_sp(
                dout[0],
                dout[1],
                dout[2],
                dout[3],
                dout[4],
                dout[5],
                dout[6],
                dout[7])
            return (dx,)
        return bprop


rfft3_sp_op_info = AiCPURegOp("RFFT3_SP") \
    .fusion_type("OPAQUE") \
    .input(0, "x", "required") \
    .output(0, "y_re1", "required") \
    .output(1, "y_re2", "required") \
    .output(2, "y_re3", "required") \
    .output(3, "y_re4", "required") \
    .output(4, "y_im1", "required") \
    .output(5, "y_im2", "required") \
    .output(6, "y_im3", "required") \
    .output(7, "y_im4", "required") \
    .attr("modes", "listInt") \
    .attr("s", "listInt") \
    .attr("cust_aicpu", "str") \
    .dtype_format(DataType.F32_Default, DataType.F32_Default, DataType.F32_Default,
                  DataType.F32_Default, DataType.F32_Default, DataType.F32_Default,
                  DataType.F32_Default, DataType.F32_Default, DataType.F32_Default) \
    .dtype_format(DataType.F64_Default, DataType.F64_Default, DataType.F64_Default,
                  DataType.F64_Default, DataType.F64_Default, DataType.F64_Default,
                  DataType.F64_Default, DataType.F64_Default, DataType.F64_Default) \
    .get_op_info()


@op_info_register(rfft3_sp_op_info)
def _rfft3_sp_aicpu():
    """RFFT3_SP AiCPU register"""
    return
################ IRFFT1D_SP op ################


class IRFFT1_SP(PrimitiveWithInfer):
    @prim_attr_register
    def __init__(self, s, origin):
        """Initialize IRFFT1_SP"""
        fft_rank = 1
        self.add_prim_attr("cust_aicpu", "FFT_new")
        self.s = s
        self.origin = origin
        pass

    def infer_shape(self, shape1, shape2):
        fft_rank = 1
        is_forward = False
        y_shape = shape1
        for i in range(fft_rank):
            inner_most = (i == fft_rank - 1)
            min_input_dim_length = (
                self.s[i] //
                2 +
                1) if (
                not is_forward and inner_most) else self.s[i]
            if self.origin[i] != 0 and self.origin[i] < min_input_dim_length:
                raise TypeError(
                    f"For '{self.name}', input 'x' and signal 's' cannot match")
            dim = (
                self.s[i] //
                2 +
                1) if is_forward and inner_most and self.s[i] != 0 else self.s[i]
            y_shape[len(shape1) - fft_rank + i] = dim
        return y_shape

    def infer_dtype(self, dtype1, dtype2):
        return dtype1

    def get_bprop(self):
        s = self.s
        shape = ops.Shape()

        def bprop(x_re, x_im, out, dout):
            x_shape = shape(x_re)
            x_rank = len(x_shape)
            modes = (x_shape[x_rank - 1], )
            rfft1_sp2 = RFFT1_SP2(s=s, modes=modes)
            dx1, dx2 = rfft1_sp2(dout)
            return (dx1, dx2)
        return bprop


irfft1_sp_op_info = AiCPURegOp("IRFFT1_SP") .fusion_type("OPAQUE") .input(
    0,
    "x_re",
    "required") .input(
        1,
        "x_im",
        "required") .output(
            0,
            "y",
            "required") .attr(
                "origin",
                "listInt") .attr(
                    "s",
                    "listInt") .attr(
                        "cust_aicpu",
                        "str") .dtype_format(
                            DataType.F64_Default,
                            DataType.F64_Default,
                            DataType.F64_Default) .dtype_format(
                                DataType.F32_Default,
                                DataType.F32_Default,
    DataType.F32_Default) .get_op_info()


@op_info_register(irfft1_sp_op_info)
def _irfft1_sp_aicpu():
    """IRFFT1_SP AiCPU register"""
    return
################ RFFT1D_SP2 op ################


class RFFT1_SP2(PrimitiveWithInfer):
    @prim_attr_register
    def __init__(self, s, modes):
        """Initialize RFFT1_SP2"""
        fft_rank = 1
        self.add_prim_attr("cust_aicpu", "FFT_new")
        self.s = s
        self.modes = modes
        pass

    def infer_shape(self, x_shape):
        fft_rank = 1
        y_shape = x_shape
        for i in range(fft_rank):
            input_index = len(y_shape) - fft_rank + i
            y_shape[input_index] = self.modes[i]
        return y_shape, y_shape

    def infer_dtype(self, x_dtype):
        return x_dtype, x_dtype


rfft1_sp2_op_info = AiCPURegOp("RFFT1_SP2") .fusion_type("OPAQUE") .input(
    0,
    "x",
    "required") .output(
        0,
        "y_re",
        "required") .output(
            1,
            "y_im",
            "required") .attr(
                "modes",
                "listInt") .attr(
                    "s",
                    "listInt") .attr(
                        "cust_aicpu",
                        "str") .dtype_format(
                            DataType.F64_Default,
                            DataType.F64_Default,
                            DataType.F64_Default) .dtype_format(
                                DataType.F32_Default,
                                DataType.F32_Default,
    DataType.F32_Default) .get_op_info()


@op_info_register(rfft1_sp2_op_info)
def _rfft1_sp2_aicpu():
    """RFFT1_SP2 AiCPU register"""
    return
################ IRFFT2D_SP op ################


class IRFFT2_SP(PrimitiveWithInfer):
    @prim_attr_register
    def __init__(self, s, origin):
        """Initialize IRFFT2_SP"""
        fft_rank = 2
        self.add_prim_attr("cust_aicpu", "FFT_new")
        self.s = s
        self.origin = origin
        pass

    def infer_shape(self, shape1, shape2, shape3, shape4):
        fft_rank = 2
        is_forward = False
        y_shape = shape1
        for i in range(fft_rank):
            inner_most = (i == fft_rank - 1)
            min_input_dim_length = (
                self.s[i] //
                2 +
                1) if (
                not is_forward and inner_most) else self.s[i]
            if self.origin[i] != 0 and self.origin[i] < min_input_dim_length:
                raise TypeError(
                    f"For '{self.name}', input 'x' and signal 's' cannot match")
            dim = (
                self.s[i] //
                2 +
                1) if is_forward and inner_most and self.s[i] != 0 else self.s[i]
            y_shape[len(shape1) - fft_rank + i] = dim
        return y_shape

    def infer_dtype(self, dtype1, dtype2, dtype3, dtype4):
        return dtype1

    def get_bprop(self):
        s = self.s
        shape = ops.Shape()

        def bprop(x_re1, x_re2, x_im1, x_im2, out, dout):
            x_shape = shape(x_re1)
            x_rank = len(x_shape)
            modes = (x_shape[x_rank - 2], x_shape[x_rank - 1])
            rfft2_sp2 = RFFT2_SP2(s=s, modes=modes)
            dx1, dx2, dx3, dx4 = rfft2_sp2(dout)
            return (dx1, dx2, dx3, dx4)
        return bprop


irfft2_sp_op_info = AiCPURegOp("IRFFT2_SP") .fusion_type("OPAQUE") .input(
    0,
    "x_re1",
    "required") .input(
        1,
        "x_re2",
        "required") .input(
            2,
            "x_im1",
            "required") .input(
                3,
                "x_im2",
                "required") .output(
                    0,
                    "y_re",
                    "required") .attr(
                        "origin",
                        "listInt") .attr(
                            "s",
                            "listInt") .attr(
                                "cust_aicpu",
                                "str") .dtype_format(
                                    DataType.F32_Default,
                                    DataType.F32_Default,
                                    DataType.F32_Default,
                                    DataType.F32_Default,
                                    DataType.F32_Default) .dtype_format(
                                        DataType.F64_Default,
                                        DataType.F64_Default,
                                        DataType.F64_Default,
                                        DataType.F64_Default,
    DataType.F64_Default) .get_op_info()


@op_info_register(irfft2_sp_op_info)
def _irfft2_sp_aicpu():
    """IRFFT2_SP AiCPU register"""
    return
################ RFFT2D_SP2 op ################


class RFFT2_SP2(PrimitiveWithInfer):
    @prim_attr_register
    def __init__(self, s, modes):
        """Initialize RFFT2_SP2"""
        fft_rank = 2
        self.add_prim_attr("cust_aicpu", "FFT_new")
        self.s = s
        self.modes = modes
        pass

    def infer_shape(self, x_shape):
        fft_rank = 2
        y_shape = x_shape
        for i in range(fft_rank):
            input_index = len(y_shape) - fft_rank + i
            y_shape[input_index] = self.modes[i]
        return y_shape, y_shape, y_shape, y_shape

    def infer_dtype(self, x_dtype):
        return x_dtype, x_dtype, x_dtype, x_dtype


rfft2_sp2_op_info = AiCPURegOp("RFFT2_SP2") .fusion_type("OPAQUE") .input(
    0,
    "x",
    "required") .output(
        0,
        "y_re1",
        "required") .output(
            1,
            "y_re2",
            "required") .output(
                2,
                "y_im1",
                "required") .output(
                    3,
                    "y_im2",
                    "required") .attr(
                        "modes",
                        "listInt") .attr(
                            "s",
                            "listInt") .attr(
                                "cust_aicpu",
                                "str") .dtype_format(
                                    DataType.F64_Default,
                                    DataType.F64_Default,
                                    DataType.F64_Default,
                                    DataType.F64_Default,
                                    DataType.F64_Default) .dtype_format(
                                        DataType.F32_Default,
                                        DataType.F32_Default,
                                        DataType.F32_Default,
                                        DataType.F32_Default,
    DataType.F32_Default) .get_op_info()


@op_info_register(rfft2_sp2_op_info)
def _rfft2_sp2_aicpu():
    """RFFT2_SP2 AiCPU register"""
    return
################ IRFFT3D_SP op ################


class IRFFT3_SP(PrimitiveWithInfer):
    @prim_attr_register
    def __init__(self, s, origin):
        """Initialize IRFFT3_SP"""
        fft_rank = 3
        self.add_prim_attr("cust_aicpu", "FFT_new")
        self.s = s
        self.origin = origin
        pass

    def infer_shape(
            self,
            shape1,
            shape2,
            shape3,
            shape4,
            shape5,
            shape6,
            shape7,
            shape8):
        fft_rank = 3
        is_forward = False
        y_shape = shape1
        for i in range(fft_rank):
            inner_most = (i == fft_rank - 1)
            min_input_dim_length = (
                self.s[i] //
                2 +
                1) if (
                not is_forward and inner_most) else self.s[i]
            if self.origin[i] != 0 and self.origin[i] < min_input_dim_length:
                raise TypeError(
                    f"For '{self.name}', input 'x' and signal 's' cannot match")
            dim = (
                self.s[i] //
                2 +
                1) if is_forward and inner_most and self.s[i] != 0 else self.s[i]
            y_shape[len(shape1) - fft_rank + i] = dim
        return y_shape

    def infer_dtype(
            self,
            dtype1,
            dtype2,
            dtype3,
            dtype4,
            dtype5,
            dtype6,
            dtype7,
            dtype8):
        return dtype1

    def get_bprop(self):
        s = self.s
        shape = ops.Shape()

        def bprop(
                x_re1,
                x_re2,
                x_re3,
                x_re4,
                x_im1,
                x_im2,
                x_im3,
                x_im4,
                out,
                dout):
            x_shape = shape(x_re1)
            x_rank = len(x_shape)
            modes = (x_shape[x_rank - 3],
                     x_shape[x_rank - 2],
                     x_shape[x_rank - 1])
            rfft3_sp2 = RFFT3_SP2(s=s, modes=modes)
            dx1, dx2, dx3, dx4, dx5, dx6, dx7, dx8 = rfft3_sp2(dout)
            return (dx1, dx2, dx3, dx4, dx5, dx6, dx7, dx8)
        return bprop


irfft3_sp_op_info = AiCPURegOp("IRFFT3_SP") \
    .fusion_type("OPAQUE") \
    .input(0, "x_re1", "required") \
    .input(1, "x_re2", "required") \
    .input(2, "x_re3", "required") \
    .input(3, "x_re4", "required") \
    .input(4, "x_im1", "required") \
    .input(5, "x_im2", "required") \
    .input(6, "x_im3", "required") \
    .input(7, "x_im4", "required") \
    .output(0, "y_re", "required") \
    .attr("origin", "listInt") \
    .attr("s", "listInt") \
    .attr("cust_aicpu", "str") \
    .dtype_format(DataType.F32_Default, DataType.F32_Default, DataType.F32_Default,
                  DataType.F32_Default, DataType.F32_Default, DataType.F32_Default,
                  DataType.F32_Default, DataType.F32_Default, DataType.F32_Default) \
    .dtype_format(DataType.F64_Default, DataType.F64_Default, DataType.F64_Default,
                  DataType.F64_Default, DataType.F64_Default, DataType.F64_Default,
                  DataType.F64_Default, DataType.F64_Default, DataType.F64_Default) \
    .get_op_info()


@op_info_register(irfft3_sp_op_info)
def _irfft3_sp_aicpu():
    """IRFFT3_SP AiCPU register"""
    return
################ RFFT3D_SP2 op ################


class RFFT3_SP2(PrimitiveWithInfer):
    @prim_attr_register
    def __init__(self, s, modes):
        """Initialize RFFT3_SP2"""
        fft_rank = 3
        self.add_prim_attr("cust_aicpu", "FFT_new")
        self.s = s
        self.modes = modes
        pass

    def infer_shape(self, x_shape):
        fft_rank = 3
        y_shape = x_shape
        for i in range(fft_rank):
            input_index = len(y_shape) - fft_rank + i
            y_shape[input_index] = self.modes[i]
        return y_shape, y_shape, y_shape, y_shape, y_shape, y_shape, y_shape, y_shape

    def infer_dtype(self, x_dtype):
        return x_dtype, x_dtype, x_dtype, x_dtype, x_dtype, x_dtype, x_dtype, x_dtype


rfft3_sp2_op_info = AiCPURegOp("RFFT3_SP2") \
    .fusion_type("OPAQUE") \
    .input(0, "x", "required") \
    .output(0, "y_re1", "required") \
    .output(1, "y_re2", "required") \
    .output(2, "y_re3", "required") \
    .output(3, "y_re4", "required") \
    .output(4, "y_im1", "required") \
    .output(5, "y_im2", "required") \
    .output(6, "y_im3", "required") \
    .output(7, "y_im4", "required") \
    .attr("modes", "listInt") \
    .attr("s", "listInt") \
    .attr("cust_aicpu", "str") \
    .dtype_format(DataType.F32_Default, DataType.F32_Default, DataType.F32_Default,
                  DataType.F32_Default, DataType.F32_Default, DataType.F32_Default,
                  DataType.F32_Default, DataType.F32_Default, DataType.F32_Default) \
    .dtype_format(DataType.F64_Default, DataType.F64_Default, DataType.F64_Default,
                  DataType.F64_Default, DataType.F64_Default, DataType.F64_Default,
                  DataType.F64_Default, DataType.F64_Default, DataType.F64_Default) \
    .get_op_info()


@op_info_register(rfft3_sp2_op_info)
def _rfft3_sp2_aicpu():
    """RFFT3_SP2 AiCPU register"""
    return
