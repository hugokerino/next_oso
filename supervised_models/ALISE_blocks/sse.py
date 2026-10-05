import torch
from torch import nn, Tensor

from typing import no_type_check
from einops import rearrange


class PatchEmbedding(nn.Module):
    """
    Integrate position information on feature map extracted
    by the spectro-spatial encoder (SSE)
    """

    def __init__(self, sse: nn.Module, tpe: nn.Module):
        """

        Parameters
        ----------
        sse : Spectro spatial encoder
        tpe : Temporal positional Encoder
        """
        super().__init__()
        self.tpe = tpe
        self.sse = sse

    def forward(self, x: Tensor, positions: Tensor) -> Tensor:
        """

        Parameters
        ----------
        x : time series of size B,T,C,H,W
        positions : B,T
        Returns
        -------
        Tensor, of size B,T,D,H,W, with D equals to C if merge_opt=add
        """
        B, T, C, H, W = x.shape
        x = self.sse(rearrange(x, "B T C H W -> (B T) C H W "))
        x = rearrange(x, "(B T) C H W -> B T C H W", B=B, T=T, H=H, W=W)
        encoded_positions = self.tpe(positions)
        encoded_positions = rearrange(encoded_positions, "b t c -> b t c 1 1")
        return x + encoded_positions


class PositionalEncoder(nn.Module):
    """
    Traditional Positional encoding as defined by Vaswani 2017.
    Implementation inspired by
    https://github.com/VSainteuf/utae-paps/blob/main/src/backbones/positional_encoding.py
    """

    def __init__(self, d: int, T: int = 1000, offset: int = 0):
        """

        Parameters
        ----------
        d : Number of features
        T : The scaling constant
        offset :
        """
        super().__init__()
        self.d = d
        self.T = T
        self.denom = torch.pow(
            T, 2 * (torch.arange(offset, offset + d).float() // 2) / d
        )

    def forward(self, batch_positions: Tensor) -> Tensor:
        """

        Parameters
        ----------    assert False
        batch_positions : (B,T) 2D tensor containing positions.

        Returns
        -------
        a tensor of size B,T,C with C corresponding to self.d
        """
        self.denom = self.denom.to(batch_positions.device)
        sinusoid_table = (
            batch_positions[:, :, None] / self.denom[None, None, :]
        )  # B x T x C
        sinusoid_table[:, :, 0::2] = torch.sin(sinusoid_table[:, :, 0::2])  # dim 2i
        sinusoid_table[:, :, 1::2] = torch.cos(sinusoid_table[:, :, 1::2])  # dim 2i+1

        return sinusoid_table


class Unet(nn.Module):
    """
    Inspired by
    https://github.com/VSainteuf/utae-paps/blob/main/src/backbones/utae.py

    Parameters
    ----------
    inplanes : input number of channels
    planes :
    encoder_widths : list of the number of channels in the Unet encoder
    decoder_widths : list of the number of channels in the Unet decoder
    encoder_norm : normalisation type description
    padding_mode : padding type as defined in Conv2d
    decoding_norm : normalisation in the decoder
    return_maps : wether all intermetdiate eafture maps are returned
    str_conv_k : kernel size
    str_conv_s : stride size
    str_conv_p : padding size
    skip_conv_norm : normalisation in the skip connections
    """

    def __init__(
        self,
        inplanes=10,
        planes=128,
        encoder_widths=[64, 64, 64, 128],
        decoder_widths=[32, 32, 64, 128],
        encoder_norm="group",
        padding_mode="reflect",
        decoding_norm="group",
        return_maps=False,
        str_conv_k=4,
        str_conv_s=2,
        str_conv_p=1,
        skip_conv_norm="group",
    ):
        super().__init__()
        self.return_maps = return_maps
        self.encoder_widths = encoder_widths

        self.decoder_widths = decoder_widths
        self.in_conv = ConvBlock(
            nkernels=[inplanes] + [encoder_widths[0], encoder_widths[0]],
            norm=encoder_norm,
            padding_mode=padding_mode,
        )
        self.out_conv = ConvBlock(
            nkernels=[decoder_widths[0], planes],
            norm=decoding_norm,
            padding_mode=padding_mode,
        )
        self.n_stages = len(encoder_widths)
        self.down_blocks = nn.ModuleList(
            DownConvBlock(
                d_in=encoder_widths[i],
                d_out=encoder_widths[i + 1],
                k=str_conv_k,
                s=str_conv_s,
                p=str_conv_p,
                norm=encoder_norm,
                padding_mode=padding_mode,
            )
            for i in range(self.n_stages - 1)
        )
        self.up_blocks = nn.ModuleList(
            [
                UpConvBlock(
                    d_in=decoder_widths[i],
                    d_out=decoder_widths[i - 1],
                    d_skip=encoder_widths[i - 1],
                    k=str_conv_k,
                    s=str_conv_s,
                    p=str_conv_p,
                    norm=decoding_norm,
                    padding_mode=padding_mode,
                    skip_conv_norm=skip_conv_norm,
                )
                for i in range(self.n_stages - 1, 0, -1)
            ]
        )

    @no_type_check
    def forward(self, input: Tensor):
        """

        Parameters
        ----------
        input : Tensor (B,C,H,W)
        Returns
        -------
        either a Tensor of size (B,C,H,W)  or if returns_map is s
        et to True a Tensor of size
        B,C,H,W and well as a list of all intermediate feature maps
        """
        dtype = input.dtype
        out = self.in_conv(input)
        feature_maps = [out]
        # SPATIAL ENCODER
        for i in range(self.n_stages - 1):
            out = self.down_blocks[i](feature_maps[-1])
            feature_maps.append(out)
            # print(out.shape)
        if self.return_maps:
            maps = [out]
        # print([out.shape for out in feature_maps])
        for i in range(self.n_stages - 1):
            skip = feature_maps[-(i + 2)]
            #  print(skip.shape, out.shape)
            out = self.up_blocks[i](out, skip)
            if self.return_maps:
                maps.append(out)
            #            out = rearrange(out, "b c h w -> b h w c")
        out = self.out_conv(out)
        if self.return_maps:
            return out.to(dtype), maps.to(dtype)

        return out.to(dtype)


class UpConvBlock(nn.Module):
    @no_type_check
    def __init__(
        self,
        d_in,
        d_out,
        k,
        s,
        p,
        final_out=None,
        norm="batch",
        d_skip=None,
        padding_mode="reflect",
        skip_conv_norm="batch",
    ):
        super().__init__()
        d = d_out if d_skip is None else d_skip
        if skip_conv_norm == "batch":
            skip_norm_begin = nn.BatchNorm2d(d)
            skip_norm_end = nn.BatchNorm2d(d_out)
        else:
            skip_norm_begin = nn.GroupNorm(num_groups=4, num_channels=d)
            skip_norm_end = nn.GroupNorm(num_groups=4, num_channels=d_out)
        self.skip_conv = nn.Sequential(
            nn.Conv2d(in_channels=d, out_channels=d, kernel_size=1),
            skip_norm_begin,
            nn.ReLU(),
        )
        self.up = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=d_in,
                out_channels=d_out,
                kernel_size=k,
                stride=s,
                padding=p,
            ),
            skip_norm_end,
            nn.ReLU(),
        )
        self.conv1 = ConvLayer(
            nkernels=[d_out + d, d_out], norm=norm, padding_mode=padding_mode
        )
        if final_out is None:
            final_out = d_out

        self.conv2 = ConvLayer(
            nkernels=[d_out, final_out], norm=norm, padding_mode=padding_mode
        )

    @no_type_check
    def forward(self, input, skip):
        out = self.up(input)
        out = torch.cat([out, self.skip_conv(skip)], dim=1)
        out = self.conv1(out)
        out = out + self.conv2(out)
        return out


class DownConvBlock(nn.Module):
    def __init__(
        self,
        d_in,
        d_out,
        k,
        s,
        p,
        norm="batch",
        padding_mode="reflect",
    ):
        super().__init__()
        self.down = ConvLayer(
            nkernels=[d_in, d_in],
            norm=norm,
            k=k,
            s=s,
            p=p,
            padding_mode=padding_mode,
        )
        self.conv1 = ConvLayer(
            nkernels=[d_in, d_out],
            norm=norm,
            padding_mode=padding_mode,
        )
        self.conv2 = ConvLayer(
            nkernels=[d_out, d_out],
            norm=norm,
            padding_mode=padding_mode,
        )

    def forward(self, input):
        out = self.down(input)
        out = self.conv1(out)
        out = out + self.conv2(out)
        return out


class ConvLayer(nn.Module):
    @no_type_check
    def __init__(
        self,
        nkernels,
        norm="batch",
        k=3,
        s=1,
        p=1,
        n_groups=4,
        last_relu=True,
        padding_mode="reflect",
    ):
        super().__init__()
        layers = []
        if norm == "batch":
            nl = nn.BatchNorm2d
        elif norm == "instance":
            nl = nn.InstanceNorm2d
        elif norm == "group":

            def group_norm(num_feats):
                return nn.GroupNorm(
                    num_channels=num_feats,
                    num_groups=n_groups,
                )

            nl = group_norm
        else:
            nl = None
        for i in range(len(nkernels) - 1):
            layers.append(
                nn.Conv2d(
                    in_channels=nkernels[i],
                    out_channels=nkernels[i + 1],
                    kernel_size=k,
                    padding=p,
                    stride=s,
                    padding_mode=padding_mode,
                )
            )
            if nl is not None:
                layers.append(nl(nkernels[i + 1]))

            if last_relu:
                layers.append(nn.ReLU())
            elif i < len(nkernels) - 2:
                layers.append(nn.ReLU())
        self.conv = nn.Sequential(*layers)

    @no_type_check
    def forward(self, input):
        return self.conv(input)


class ConvBlock(nn.Module):
    def __init__(
        self,
        nkernels,
        norm="batch",
        last_relu=True,
        padding_mode="reflect",
    ):
        super().__init__()
        self.conv = ConvLayer(
            nkernels=nkernels,
            norm=norm,
            last_relu=last_relu,
            padding_mode=padding_mode,
        )

    def forward(self, input):
        return self.conv(input)
