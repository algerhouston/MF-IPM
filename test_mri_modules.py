import torch

from alignment_modules import ChannelAlignmentModule, SpatialAlignmentModule
from decoder import RDecoder
from encoders import REncoder
from mamba_blocks import CrossModalFusionMamba, FrequencySpatialHomogenizationMamba
from mri_ops import data_consistency, fft2c, undersample_image


def test_encoder_alignment_fusion_decoder_shapes():
    batch, channels, height, width = 2, 1, 16, 16
    feature_channels = 8
    target = torch.rand(batch, channels, height, width)
    reference = torch.rand(batch, channels, height, width)

    encoder = REncoder(channels, feature_channels)
    spatial_align = SpatialAlignmentModule(feature_channels)
    channel_align = ChannelAlignmentModule(feature_channels)
    fusion = CrossModalFusionMamba(feature_channels, num_blocks=1)
    decoder = RDecoder(feature_channels, channels)

    target_feature = spatial_align(encoder(target))
    reference_feature = channel_align(encoder(reference))
    fused = fusion(target_feature, reference_feature)
    reconstruction = decoder(fused)

    assert target_feature.shape == (batch, feature_channels, height, width)
    assert reference_feature.shape == target_feature.shape
    assert fused.shape == target_feature.shape
    assert reconstruction.shape == target.shape


def test_frequency_spatial_homogenization_shape():
    feature = torch.rand(2, 8, 16, 16)
    frequency = torch.rand(2, 8, 16, 16)
    module = FrequencySpatialHomogenizationMamba(8, num_blocks=1)

    out = module(feature, frequency)

    assert out.shape == feature.shape


def test_data_consistency_keeps_measured_kspace_samples():
    image = torch.rand(1, 1, 16, 16)
    measured_kspace = fft2c(image)
    mask = torch.zeros(1, 1, 16, 16)
    mask[..., ::2, :] = 1.0
    aliased = undersample_image(image, mask)

    corrected = data_consistency(aliased, measured_kspace, mask)
    corrected_kspace = fft2c(corrected)

    assert torch.allclose(
        corrected_kspace * mask,
        measured_kspace * mask,
        atol=1e-5,
        rtol=1e-5,
    )
