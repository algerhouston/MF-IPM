import ast
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from crc import ConvReluConv
from decoder import RDecoder
from encoders import REncoder
from meanflow import MeanFlow
from mamba_blocks import CrossModalFusionMamba, FrequencySpatialHomogenizationMamba


class IdentityFFT(nn.Module):
    def forward(self, x, mask):
        return x


def load_basic_block():
    source = Path("").read_text(encoding="utf-8")
    module_ast = ast.parse(source)
    wanted_classes = [
        node for node in module_ast.body
        if isinstance(node, ast.ClassDef) and node.name in {"MeanFlowVelocityModel", "BasicBlock"}
    ]
    namespace = {
        "torch": torch,
        "nn": nn,
        "init": init,
        "F": F,
        "REncoder": REncoder,
        "ConvReluConv": ConvReluConv,
        "MeanFlow": MeanFlow,
        "CrossModalFusionMamba": CrossModalFusionMamba,
        "FrequencySpatialHomogenizationMamba": FrequencySpatialHomogenizationMamba,
        "RDecoder": RDecoder,
    }
    for class_node in wanted_classes:
        ast.fix_missing_locations(class_node)
    exec(compile(ast.Module(wanted_classes, type_ignores=[]), "<basic_block>", "exec"), namespace)
    return namespace["BasicBlock"]


def test_basic_block_inserted_flow_keeps_tensor_shapes():
    block_cls = load_basic_block()
    block = block_cls()

    assert hasattr(block, "x_encoder")
    assert hasattr(block, "f_encoder")
    assert hasattr(block, "x_crc")
    assert hasattr(block, "f_crc")
    assert hasattr(block, "flow_fusion")
    assert hasattr(block.flow_fusion, "spatial_alignment")
    assert hasattr(block.flow_fusion, "channel_alignment")
    assert hasattr(block, "flow_homogenization")
    assert hasattr(block, "flow_decoder")
    assert isinstance(block.meanflow, MeanFlow)
    assert hasattr(block, "meanflow_model")
    assert callable(block.meanflow_process)

    x = torch.rand(2, 1, 16, 16)
    mask = torch.ones(16, 16, 2)

    f = block.inverse_fourier(x)
    x_pred, symloss = block(x, IdentityFFT(), x, mask)

    assert f.shape == x.shape
    assert x_pred.shape == x.shape
    assert symloss.shape == (2, 32, 16, 16)
