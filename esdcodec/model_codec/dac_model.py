# Copyright (c) 2025 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
From DAC: https://github.com/descriptinc/descript-audio-codec/blob/main/dac/model
"""
import math
from typing import List
from typing import Union, Optional

import numpy as np
import torch
from audiotools import AudioSignal
from audiotools.ml import BaseModel
from torch import nn

from .dac_layers import Snake1d
from .dac_layers import WNConv1d
from .dac_layers import WNConvTranspose1d
from .dac_quantize import ResidualVectorQuantize, SimVQ1D
from easydict import EasyDict as edict
import torch.nn.functional as F
from .cnn import ConvNeXtBlock
from .spectral_ops import ISTFT

def init_weights(m):
    if isinstance(m, nn.Conv1d):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias, 0)


def pad_to_length(x, length, pad_value=0):
    # Get the current size along the last dimension
    current_length = x.shape[-1]

    # If the length is greater than current_length, we need to pad
    if length > current_length:
        pad_amount = length - current_length
        # Pad on the last dimension (right side), keeping all other dimensions the same
        x_padded = F.pad(x, (0, pad_amount), value=pad_value)
    else:
        # If no padding is required, simply slice the tensor
        x_padded = x[..., :length]

    return x_padded

class Code2Wavs(torch.nn.Module):
    def __init__(self, input_channels=160, dim=128, n_fft=128, intermediate_dim=1536, num_layers=4, layer_scale_init_value: Optional[float] = None, adanorm_num_embeddings: Optional[int] = None):
        super().__init__()
        self.input_channels = input_channels
        self.dim = dim
        self.embed = nn.Conv1d(input_channels, dim, kernel_size=7, padding=3)
        self.adanorm = adanorm_num_embeddings is not None
        if adanorm_num_embeddings:
            self.norm = AdaLayerNorm(adanorm_num_embeddings, dim, eps=1e-6)
        else:
            self.norm = nn.LayerNorm(dim, eps=1e-6)       
        layer_scale_init_value = layer_scale_init_value or 1 / num_layers
        self.convnext = nn.ModuleList(
            [
                ConvNeXtBlock(
                    dim=dim,
                    intermediate_dim=intermediate_dim,
                    layer_scale_init_value=layer_scale_init_value,
                    adanorm_num_embeddings=adanorm_num_embeddings,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_layer_norm = nn.LayerNorm(dim, eps=1e-6)
        self.mag_out_block = [
                    nn.ReflectionPad1d(1),
                    WNConv1d(dim, n_fft//2+1, kernel_size = 3, padding = 0, bias = False),
                    ]

        self.phase_out_block = [
                    nn.ReflectionPad1d(1),
                    WNConv1d(dim, n_fft // 2 +1, kernel_size = 3, padding = 0, bias = False),
                    ]

        self.mag_out_block = nn.Sequential(*self.mag_out_block)
        self.phase_out_block = nn.Sequential(*self.phase_out_block)

        hop_length = n_fft // 4
        self.istft = ISTFT(n_fft=n_fft, hop_length=hop_length, win_length=n_fft, padding='same')
        self.apply(init_weights)
    
    def forward(self, x: torch.Tensor, bandwidth_id: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.embed(x)
        if self.adanorm:
            assert bandwidth_id is not None
            x = self.norm(x.transpose(1, 2), cond_embedding_id=bandwidth_id)
        else:
            x = self.norm(x.transpose(1, 2))        
        x = x.transpose(1, 2)
        for conv_block in self.convnext:
            x = conv_block(x, cond_embedding_id=bandwidth_id)
        x = self.final_layer_norm(x.transpose(1, 2))
        x = x.transpose(1, 2)  # B, D, T
        mag = self.mag_out_block(x)
        phase = self.phase_out_block(x)
        mag = torch.exp(mag)
        mag = torch.clip(mag, max=1e2)
        source = mag * (torch.cos(phase) + 1j * torch.sin(phase))
        audio = self.istft(source)
        return audio, mag


class ResidualUnit(nn.Module):
    def __init__(self, dim: int = 16, dilation: int = 1):
        super().__init__()
        pad = ((7 - 1) * dilation) // 2
        self.block = nn.Sequential(
            Snake1d(dim),
            WNConv1d(dim, dim, kernel_size=7, dilation=dilation, padding=pad),
            Snake1d(dim),
            WNConv1d(dim, dim, kernel_size=1),
        )

    def forward(self, x):
        y = self.block(x)
        pad = (x.shape[-1] - y.shape[-1]) // 2
        if pad > 0:
            x = x[..., pad:-pad]
        return x + y


class EncoderBlock(nn.Module):
    def __init__(self, dim: int = 16, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            ResidualUnit(dim // 2, dilation=1),
            ResidualUnit(dim // 2, dilation=3),
            ResidualUnit(dim // 2, dilation=9),
            Snake1d(dim // 2),
            WNConv1d(
                dim // 2,
                dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
            ),
        )

    def forward(self, x):
        return self.block(x)


class Encoder(nn.Module):
    def __init__(
        self,
        d_model: int = 64,
        strides: list = [2, 4, 8, 8],
        d_latent: int = 64,
    ):
        super().__init__()
        # Create first convolution
        self.block = [WNConv1d(1, d_model, kernel_size=7, padding=3)]

        # Create EncoderBlocks that double channels as they downsample by `stride`
        for stride in strides:
            d_model *= 2
            self.block += [EncoderBlock(d_model, stride=stride)]

        # Create last convolution
        self.block += [
            Snake1d(d_model),
            WNConv1d(d_model, d_latent, kernel_size=3, padding=1),
        ]

        # Wrap black into nn.Sequential
        self.block = nn.Sequential(*self.block)
        self.enc_dim = d_model

    def forward(self, x):
        return self.block(x)


class DecoderBlock(nn.Module):
    def __init__(self, input_dim: int = 16, output_dim: int = 8, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            Snake1d(input_dim),
            WNConvTranspose1d(
                input_dim,
                output_dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
            ),
            ResidualUnit(output_dim, dilation=1),
            ResidualUnit(output_dim, dilation=3),
            ResidualUnit(output_dim, dilation=9),
        )

    def forward(self, x):
        return self.block(x)


class Decoder(nn.Module):
    def __init__(
        self,
        input_channel,
        channels,
        rates,
        d_out: int = 1,
    ):
        super().__init__()

        # Add first conv layer
        layers = [WNConv1d(input_channel, channels, kernel_size=7, padding=3)]

        # Add upsampling + MRF blocks
        for i, stride in enumerate(rates):
            input_dim = channels // 2**i
            output_dim = channels // 2 ** (i + 1)
            layers += [DecoderBlock(input_dim, output_dim, stride)]

        # Add final conv layer
        layers += [
            Snake1d(output_dim),
            WNConv1d(output_dim, d_out, kernel_size=7, padding=3),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class ConditionDecoder(nn.Module):
    def __init__(
        self,
        input_channel,
        channels,
        rates,
        d_out: int = 1,
    ):
        super().__init__()

        self.rates = rates
        self.out_channels = []
        self.rate_cond = [1] + rates[::-1][1:-1]

        # Add first conv layer
        self.first_layer = WNConv1d(input_channel, channels, kernel_size=7, padding=3)

        # Add upsampling + MRF blocks
        self.ups = nn.ModuleList()
        self.res_blocks = nn.ModuleList()

        for i, stride in enumerate(rates):
            input_dim = channels // 2**i
            output_dim = channels // 2 ** (i + 1)
            self.out_channels.append(output_dim)
            self.ups.append(
                nn.Sequential(
                    Snake1d(input_dim),
                    WNConvTranspose1d(
                        input_dim,
                        output_dim,
                        kernel_size=2 * stride,
                        stride=stride,
                        padding=math.ceil(stride / 2),
                    )
                )
            )
            res_block = nn.Sequential(
                ResidualUnit(output_dim, dilation=1),
                ResidualUnit(output_dim, dilation=3),
                ResidualUnit(output_dim, dilation=9),
            )
            self.res_blocks.append(res_block)

        self.ch_cond = [1] + self.out_channels[::-1][1:]
        self.down_blocks = nn.ModuleList()
        for i, r in enumerate(self.rate_cond):
            if r % 2 == 0:
                down_block = WNConv1d(self.ch_cond[i], self.ch_cond[i+1], kernel_size=2 * r, stride = r, padding = (r+1) // 2)
            else:
                down_block = WNConv1d(self.ch_cond[i], self.ch_cond[i+1], kernel_size=2 * r + 1, stride = r, padding = (r+1) // 2)
            self.down_blocks.append(down_block)

        self.reflection_pad = nn.ReflectionPad1d((1, 0))

        # Add final conv layer
        self.post_layer = nn.Sequential(
            Snake1d(output_dim),
            WNConv1d(output_dim, d_out, kernel_size=7, padding=3),
            nn.Tanh(),
        )


    def forward(self, x, cond):
        x = self.first_layer(x)

        down_cond = []
        for i in range(len(self.rate_cond)):
            cond = self.down_blocks[i](cond)
            down_cond.append(cond)

        down_cond = down_cond[::-1]

        for i, stride in enumerate(self.rates):
            x = self.ups[i](x)
            if i == len(self.rates) - 2:
                x = self.reflection_pad(x)
            if i < len(self.rates) - 1:
                x = x + down_cond[i]
            x = self.res_blocks[i](x)

        output = self.post_layer(x)

        return output

class DAC(BaseModel):
    def __init__(
        self,
        encoder_dim: int = 64,
        encoder_rates: List[int] = [2, 4, 8, 8],
        latent_dim: int = None,
        decoder_dim: int = 1536,
        decoder_rates: List[int] = [8, 8, 4, 2],
        n_codebooks: int = 9,
        codebook_size: int = 1024,
        codebook_dim: Union[int, list] = 8,
        quantizer_dropout: bool = False,
        sample_rate: int = 44100,
        semantic_codebook_size: int = 1024,
        distill_projection_out_dim=1024,
        distill=False,
        convnext=True,
        is_causal=False,
    ):
        super().__init__()

        self.encoder_dim = encoder_dim
        self.encoder_rates = encoder_rates
        self.decoder_dim = decoder_dim
        self.decoder_rates = decoder_rates
        self.sample_rate = sample_rate

        if latent_dim is None:
            latent_dim = encoder_dim * (2 ** len(encoder_rates))

        self.latent_dim = latent_dim

        self.hop_length = np.prod(encoder_rates)
        self.encoder = Encoder(encoder_dim, encoder_rates, latent_dim)

        self.n_codebooks = n_codebooks
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.semantic_vq = SimVQ1D(
            semantic_codebook_size,
            latent_dim,
            beta=0.25,
            legacy=False,
        )
        self.quantizer = ResidualVectorQuantize(
            input_dim=latent_dim,
            n_codebooks=n_codebooks,
            codebook_size=codebook_size,
            codebook_dim=codebook_dim,
            quantizer_dropout=quantizer_dropout,
        )

        self.sub_source = Code2Wavs(
            input_channels=latent_dim,
            dim=256,
            n_fft=960,
            intermediate_dim=1024,
            num_layers=6)
        self.decoder = ConditionDecoder(
            latent_dim,
            decoder_dim,
            decoder_rates,
        )
        self.sample_rate = sample_rate

        self.distill = distill
        if self.distill:
            self.distill_projection = WNConv1d(
                latent_dim,
                distill_projection_out_dim,
                kernel_size=1,
            )
            if convnext:
                self.convnext = nn.Sequential(
                    *[
                        ConvNeXtBlock(
                            dim=distill_projection_out_dim,
                            intermediate_dim=1536,
                            is_causal=is_causal,
                        )
                        for _ in range(4)
                    ],  # Unpack the list directly into nn.Sequential
                    WNConv1d(
                        distill_projection_out_dim,
                        latent_dim,
                        kernel_size=1,
                    ),
                )
            else:
                self.convnext = nn.Identity()
        self.apply(init_weights)

    def preprocess(self, audio_data, sample_rate):
        if sample_rate is None:
            sample_rate = self.sample_rate
        assert sample_rate == self.sample_rate

        length = audio_data.shape[-1]
        right_pad = math.ceil(length / self.hop_length) * self.hop_length - length
        audio_data = nn.functional.pad(audio_data, (0, right_pad))

        return audio_data

    def encode(
        self,
        audio_data: torch.Tensor,
        sample_rate=24000,
        n_quantizers: int = None,
    ):
        """Encode given audio data and return quantized latent codes

        Parameters
        ----------
        audio_data : Tensor[B x 1 x T]
            Audio data to encode
        n_quantizers : int, optional
            Number of quantizers to use, by default None
            If None, all quantizers are used.

        Returns
        -------
        dict
            A dictionary with the following keys:
            "z" : Tensor[B x D x T]
                Quantized continuous representation of input
            "codes" : Tensor[B x N x T]
                Codebook indices for each codebook
                (quantized discrete representation of input)
            "latents" : Tensor[B x N*D x T]
                Projected latents (continuous representation of input before quantization)
            "vq/commitment_loss" : Tensor[1]
                Commitment loss to train encoder to predict vectors closer to codebook
                entries
            "vq/codebook_loss" : Tensor[1]
                Codebook loss to update the codebook
            "length" : int
                Number of samples in input audio
        """
        assert not self.training
        audio_data = self.preprocess(audio_data, sample_rate)
        z = self.encoder(audio_data)
        (semantic, emb_loss, semantic_codes), loss_break = self.semantic_vq(z)
        semantic = self.distill_projection(semantic)
        semantic = self.convnext(semantic)
        if semantic is not None:
            assert (z.shape[-1] - semantic.shape[-1]) <= 2
            z = z[..., : semantic.shape[-1]] - semantic
        z, codes, latents, commitment_loss, codebook_loss, first_layer_quantized = (
            self.quantizer(
                z,
                n_quantizers,
                possibly_no_quantizer=False,
            )
        )
        if semantic is not None:
            z = z + semantic 
        return semantic, codes 

    def decode_from_codes(self, acoustic_codes: torch.Tensor, semantic_latent):
        # acoustic codes should not contain any semantic code
        z = 0.0
        if acoustic_codes is not None:
            z = self.quantizer.from_codes(acoustic_codes)[0]
        z = z + semantic_latent

        sub_wav, sub_mag = self.sub_source(z)
        sub_wav = sub_wav.unsqueeze(1)
        z = self.decoder(z, sub_wav)  # audio
        return z

    def forward(
        self,
        audio_data: torch.Tensor,
        sample_rate: int = None,
        n_quantizers: int = None,
        bypass_quantize=False,
        possibly_no_quantizer=False,
    ):
        """Model forward pass

        Parameters
        ----------
        audio_data : Tensor[B x 1 x T]
            Audio data to encode
        sample_rate : int, optional
            Sample rate of audio data in Hz, by default None
            If None, defaults to `self.sample_rate`
        n_quantizers : int, optional
            Number of quantizers to use, by default None.
            If None, all quantizers are used.

        Returns
        -------
        dict
            A dictionary with the following keys:
            "z" : Tensor[B x D x T]
                Quantized continuous representation of input
            "codes" : Tensor[B x N x T]
                Codebook indices for each codebook
                (quantized discrete representation of input)
            "latents" : Tensor[B x N*D x T]
                Projected latents (continuous representation of input before quantization)
            "vq/commitment_loss" : Tensor[1]
                Commitment loss to train encoder to predict vectors closer to codebook
                entries
            "vq/codebook_loss" : Tensor[1]
                Codebook loss to update the codebook
            "length" : int
                Number of samples in input audio
            "audio" : Tensor[B x 1 x length]
                Decoded audio data.
        """
        length = audio_data.shape[-1]
        audio_data = self.preprocess(audio_data, sample_rate)
        z = self.encoder(audio_data)
        (semantic, emb_loss, semantic_codes), loss_break = self.semantic_vq(z)
        if self.distill:
            semantic = self.distill_projection(semantic)
            semantic = self.convnext(semantic)
        if semantic is not None:
            assert (z.shape[-1] - semantic.shape[-1]) <= 2
            z = z[..., : semantic.shape[-1]] - semantic
        if bypass_quantize:
            codes, latents, commitment_loss, codebook_loss, first_layer_quantized = (
                None,
                None,
                0.0,
                0.0,
                None,
            )
            z = 0.0
        else:
            z, codes, latents, commitment_loss, codebook_loss, first_layer_quantized = (
                self.quantizer(
                    z,
                    n_quantizers,
                    possibly_no_quantizer=possibly_no_quantizer,
                )
            )
        if semantic is not None:
            z = z + semantic

        sub_wav, sub_mag = self.sub_source(z)
        sub_wav = sub_wav.unsqueeze(1)
        x = self.decoder(z, sub_wav)

        x = pad_to_length(x, length)

        semantic_edict = edict(
            {
                "x": semantic,
                "codes": semantic_codes,
                "penalty": loss_break.commitment, 
                "metrics": {},
                "bypassed_quantize": bypass_quantize,
            }
        )
        acoustic_edict = edict(
            {
                "x": x,
                "source": sub_wav,
                "source_mag": sub_mag,
                "z": z,
                "codes": codes,
                "latents": latents,
                "penalty": commitment_loss,
                "vq/codebook_loss": codebook_loss,
                "metrics": {},
            }
        )
        return acoustic_edict, semantic_edict

