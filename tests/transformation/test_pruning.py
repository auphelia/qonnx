# Copyright (c) 2023 Advanced Micro Devices, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of qonnx nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from qonnx.transformation.fold_constants import FoldConstants
from qonnx.transformation.pruning import ApplyMasks, PropagateMasks
from qonnx.util.test import download_model


def test_apply_and_propagate_masks():
    model = download_model("FINN-TFC_W2A2", do_cleanup=True, return_modelwrapper=True)
    # manifest quantized weights as initializers
    model = model.transform(FoldConstants([]))
    mm_nodes = model.get_nodes_by_op_type("MatMul")
    # mark channels 0 and 3 from tensor Mul_0_out0
    # and channel 6 for the input to the 2nd MatMul as well as
    # channel 2 of input and 5 of output from the matmul weight
    # to be pruned
    prune_spec = {"Mul_0_out0": {0, 3}, mm_nodes[0].input[1]: {"i2", "o5"}, mm_nodes[1].input[0]: {6}}
    model = model.transform(ApplyMasks(prune_spec))
    assert model.get_tensor_sparsity("Mul_0_out0") == prune_spec["Mul_0_out0"]
    assert model.get_tensor_sparsity(mm_nodes[1].input[0]) == prune_spec[mm_nodes[1].input[0]]
    assert model.get_tensor_sparsity(mm_nodes[0].input[1]) == prune_spec[mm_nodes[0].input[1]]
    # now apply the propagation
    model = model.transform(PropagateMasks())
    assert model.get_tensor_sparsity("Mul_0_out0") == {0, 2, 3}
    assert model.get_tensor_sparsity(mm_nodes[0].input[1]) == {"i0", "i2", "i3", "o5", "o6"}
    assert model.get_tensor_sparsity("BatchNormalization_0_out0") == {5, 6}
