from layers import *

class UNet(nn.Module):
    def __init__(
        self,
        time_steps,
        input_channels=3,
        output_channels=3,
        num_res_blocks=2,
        base_channels=128,
        base_channels_multiples=(1, 2, 4, 8),
        apply_attention=(True, True, True, True),
        dropout_rate=0.1,
        time_multiple=4,
        attn_type="nonlocal"
    ):
        super().__init__()

        time_emb_dims_exp = base_channels * time_multiple
        self.time_embeddings = SinusoidalPositionEmbeddings(total_time_steps= time_steps,time_emb_dims=base_channels, time_emb_dims_exp=time_emb_dims_exp)

        self.first = nn.Conv2d(in_channels=input_channels, out_channels=base_channels, kernel_size=3, stride=1, padding="same")

        num_resolutions = len(base_channels_multiples)

        # Encoder part of the UNet. Dimension reduction.
        self.encoder_blocks = nn.ModuleList()
        curr_channels       = [base_channels]
        in_channels         = base_channels

        for level in range(num_resolutions):
            out_channels = base_channels * base_channels_multiples[level]

            for _ in range(num_res_blocks):

                block = ResBlockWithTime(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    dropout_rate=dropout_rate,
                    time_emb_dims=time_emb_dims_exp,
                    apply_attention=apply_attention[level],
                    attn_type=attn_type
                )
                self.encoder_blocks.append(block)

                in_channels = out_channels
                curr_channels.append(in_channels)

            if level != (num_resolutions - 1):
                self.encoder_blocks.append(DownSample(channels=in_channels))
                curr_channels.append(in_channels)

        # Bottleneck in between
        # attn_type_cls = {"nonlocal":NonlocalAttention, "multihead":AttentionBlock}[attn_type]
        # self.bottleneck_blocks = nn.Sequential(NACBlock(in_channels, in_channels, nn.SiLU()),
        #                                        attn_type_cls(in_channels),
        #                                        NACBlock(in_channels, in_channels, nn.SiLU())
        #                                        )
        self.bottleneck_blocks = nn.ModuleList([ResBlockWithTime(in_channels, in_channels,
                                                dropout_rate=dropout_rate,
                                                time_emb_dims=time_emb_dims_exp,
                                                apply_attention=apply_attention[level],
                                                attn_type=attn_type),
                                               ResBlockWithTime(in_channels, in_channels,
                                                dropout_rate=dropout_rate,
                                                time_emb_dims=time_emb_dims_exp,
                                                apply_attention=False)
                                               ])

        # Decoder part of the UNet. Dimension restoration with skip-connections.
        self.decoder_blocks = nn.ModuleList()

        for level in reversed(range(num_resolutions)):
            out_channels = base_channels * base_channels_multiples[level]

            for _ in range(num_res_blocks + 1):
                encoder_in_channels = curr_channels.pop()
                block = ResBlockWithTime(
                    in_channels=encoder_in_channels + in_channels,
                    out_channels=out_channels,
                    dropout_rate=dropout_rate,
                    time_emb_dims=time_emb_dims_exp,
                    apply_attention=apply_attention[level],
                    attn_type=attn_type
                )

                in_channels = out_channels
                self.decoder_blocks.append(block)

            if level != 0:
                self.decoder_blocks.append(UpSample(in_channels))

        self.final = NACBlock(in_channels, output_channels, nn.SiLU())

    def forward(self, x, t):

        time_emb = self.time_embeddings(t)

        h = self.first(x)
        outs = [h]

        for layer in self.encoder_blocks:
            h = layer(h, time_emb)
            outs.append(h)

        for layer in self.bottleneck_blocks:
            h = layer(h, time_emb)

        for layer in self.decoder_blocks:
            if isinstance(layer, ResBlockWithTime):
                out = outs.pop()
                h = torch.cat([h, out], dim=1)
            h = layer(h, time_emb)

        h = self.final(h)

        return h

class TimestepEmbedSequential(nn.Sequential):
    def forward(self, x, emb, context=None):
        for layer in self:
            if isinstance(layer, ResBlockWithTime):
                x = layer(x, emb)
            elif isinstance(layer, CrossAttention):
                x = layer(x, context=context)
            else:
                x = layer(x)
        return x

class UNetConditionalCat(nn.Module):
    def __init__(
        self,
        time_steps,
        input_channels=3,
        cond_channels=3,
        output_channels=3,
        num_res_blocks=2,
        base_channels=128,
        n_heads=4,
        base_channels_multiples=(1, 2, 4, 8),
        apply_attention=(True, True, True, True),
        dropout_rate=0.0,
        time_multiple=4,
        use_scale_shift_norm=False,
        pixel_shuffle=False
    ):
        super().__init__()

        time_emb_dims_exp = base_channels * time_multiple
        self.time_embeddings = SinusoidalPositionEmbeddings(total_time_steps= time_steps,time_emb_dims=base_channels, time_emb_dims_exp=time_emb_dims_exp)

        self.first = nn.Conv2d(in_channels=input_channels+cond_channels,
                               out_channels=base_channels, kernel_size=3, stride=1, padding="same")

        num_resolutions = len(base_channels_multiples)

        # Encoder part of the UNet. Dimension reduction.
        self.encoder_blocks = nn.ModuleList()
        curr_channels       = [base_channels]
        in_channels         = base_channels

        for level in range(num_resolutions):
            out_channels = base_channels * base_channels_multiples[level]
            
            for _ in range(num_res_blocks):
                layers = []
                block = ResBlockWithTime(
                    in_chs=in_channels,
                    out_chs=out_channels,
                    dropout_rate=dropout_rate,
                    time_emb_dims=time_emb_dims_exp,
                    use_scale_shift_norm=use_scale_shift_norm
                )
                layers.append(block)
                in_channels = out_channels
                if apply_attention[level]:
                    layers.append(AttentionBlock(in_channels, n_heads, dropout=dropout_rate))
                self.encoder_blocks.append(TimestepEmbedSequential(*layers))
                curr_channels.append(in_channels)

            if level != (num_resolutions - 1):
                block = ResBlockWithTime(
                    in_chs=in_channels,
                    out_chs=in_channels,
                    dropout_rate=dropout_rate,
                    time_emb_dims=time_emb_dims_exp,
                    use_scale_shift_norm=use_scale_shift_norm
                )
                self.encoder_blocks.append(TimestepEmbedSequential(block, DownSample(channels=in_channels)))
                curr_channels.append(in_channels)
        
        self.bottleneck_blocks = nn.ModuleList([ResBlockWithTime(in_channels, in_channels,
                                                dropout_rate=dropout_rate,
                                                time_emb_dims=time_emb_dims_exp,
                                                use_scale_shift_norm=use_scale_shift_norm),
                                               AttentionBlock(in_channels, n_heads, dropout=dropout_rate),
                                               ResBlockWithTime(in_channels, in_channels,
                                                dropout_rate=dropout_rate,
                                                time_emb_dims=time_emb_dims_exp,
                                                use_scale_shift_norm=use_scale_shift_norm)
                                               ])

        # Decoder part of the UNet. Dimension restoration with skip-connections.
        self.decoder_blocks = nn.ModuleList()

        for level in reversed(range(num_resolutions)):
            out_channels = base_channels * base_channels_multiples[level]
            for it in range(num_res_blocks+1):
                layers = []
                encoder_in_channels = curr_channels.pop()
                block = ResBlockWithTime(
                    in_chs=encoder_in_channels + in_channels,
                    out_chs=out_channels,
                    dropout_rate=dropout_rate,
                    time_emb_dims=time_emb_dims_exp,
                    use_scale_shift_norm=use_scale_shift_norm
                )
                layers.append(block)
                in_channels = out_channels
                if apply_attention[level]:
                    layers.append(AttentionBlock(in_channels, n_heads, dropout=dropout_rate))
                if level != 0 and it == num_res_blocks:
                    block = ResBlockWithTime(
                        in_chs=in_channels,
                        out_chs=in_channels,
                        dropout_rate=dropout_rate,
                        time_emb_dims=time_emb_dims_exp,
                        use_scale_shift_norm=use_scale_shift_norm
                    )
                    layers.append(block)
                    if pixel_shuffle:
                        layers.append(UpSamplePixelShuffle(in_channels))
                    else:
                        layers.append(UpSample(in_channels))
                self.decoder_blocks.append(TimestepEmbedSequential(*layers))

        self.final = NACBlock(in_channels, output_channels, nn.SiLU())

    def forward(self, x, t, context):

        time_emb = self.time_embeddings(t)

        h = self.first(torch.cat([x, context], dim=1))
        outs = [h]

        for layer in self.encoder_blocks:
            h = layer(h, time_emb)
            outs.append(h)

        for layer in self.bottleneck_blocks:
            h = layer(h, time_emb)
        
        for layer in self.decoder_blocks:
            out = outs.pop()
            h = torch.cat([h, out], dim=1)
            h = layer(h, time_emb)

        h = self.final(h)

        return h

class UNetConditionalCrossAttn(nn.Module):
    def __init__(
        self,
        time_steps,
        input_channels=3,
        output_channels=3,
        num_res_blocks=2,
        base_channels=128,
        base_channels_multiples=(1, 2, 4, 8),
        apply_attention=(True, True, True, True),
        dropout_rate=0.1,
        time_multiple=4,
        context_dim = 4,
        pixel_shuffle=False
    ):
        super().__init__()

        time_emb_dims_exp = base_channels * time_multiple
        self.time_embeddings = SinusoidalPositionEmbeddings(total_time_steps= time_steps,time_emb_dims=base_channels, time_emb_dims_exp=time_emb_dims_exp)

        self.first = nn.Conv2d(in_channels=input_channels, out_channels=base_channels, kernel_size=3, stride=1, padding="same")

        num_resolutions = len(base_channels_multiples)

        # Encoder part of the UNet. Dimension reduction.
        self.encoder_blocks = nn.ModuleList()
        curr_channels       = [base_channels]
        in_channels         = base_channels

        for level in range(num_resolutions):
            out_channels = base_channels * base_channels_multiples[level]
            
            for _ in range(num_res_blocks):
                layers = []
                block = ResBlockWithTime(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    dropout_rate=dropout_rate,
                    time_emb_dims=time_emb_dims_exp,
                    apply_attention=apply_attention[level]
                )
                layers.append(block)
                in_channels = out_channels
                layers.append(CrossAttention(in_channels, condition_dim=context_dim, dropout=dropout_rate))
                self.encoder_blocks.append(TimestepEmbedSequential(*layers))
                curr_channels.append(in_channels)

            if level != (num_resolutions - 1):
                block = ResBlockWithTime(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    dropout_rate=dropout_rate,
                    time_emb_dims=time_emb_dims_exp,
                    apply_attention=False
                )
                self.encoder_blocks.append(TimestepEmbedSequential(block, DownSample(channels=in_channels)))
                curr_channels.append(in_channels)
        
        self.bottleneck_blocks = nn.ModuleList([ResBlockWithTime(in_channels, in_channels,
                                                dropout_rate=dropout_rate,
                                                time_emb_dims=time_emb_dims_exp,
                                                apply_attention=False),
                                               CrossAttention(in_channels, condition_dim=context_dim, dropout=dropout_rate),
                                               ResBlockWithTime(in_channels, in_channels,
                                                dropout_rate=dropout_rate,
                                                time_emb_dims=time_emb_dims_exp,
                                                apply_attention=False)
                                               ])

        # Decoder part of the UNet. Dimension restoration with skip-connections.
        self.decoder_blocks = nn.ModuleList()

        for level in reversed(range(num_resolutions)):
            out_channels = base_channels * base_channels_multiples[level]
            for it in range(num_res_blocks+1):
                layers = []
                encoder_in_channels = curr_channels.pop()
                block = ResBlockWithTime(
                    in_channels=encoder_in_channels + in_channels,
                    out_channels=out_channels,
                    dropout_rate=dropout_rate,
                    time_emb_dims=time_emb_dims_exp,
                    apply_attention=apply_attention[level]
                )
                layers.append(block)
                in_channels = out_channels
                layers.append(CrossAttention(in_channels, condition_dim=context_dim, dropout=dropout_rate))
                if level != 0 and it == num_res_blocks:
                    block = ResBlockWithTime(
                        in_channels=in_channels,
                        out_channels=in_channels,
                        dropout_rate=dropout_rate,
                        time_emb_dims=time_emb_dims_exp,
                        apply_attention=False
                    )
                    layers.append(block)
                    if pixel_shuffle:
                        layers.append(UpSamplePixelShuffle(in_channels))
                    else:
                        layers.append(UpSample(in_channels))
                self.decoder_blocks.append(TimestepEmbedSequential(*layers))

        self.final = NACBlock(in_channels, output_channels, nn.SiLU())

    def forward(self, x, t, context):

        time_emb = self.time_embeddings(t)

        h = self.first(x)
        outs = [h]

        for layer in self.encoder_blocks:
            h = layer(h, time_emb, context=context)
            outs.append(h)

        for layer in self.bottleneck_blocks:
            h = layer(h, time_emb, context=context)
        
        for layer in self.decoder_blocks:
            out = outs.pop()
            h = torch.cat([h, out], dim=1)
            h = layer(h, time_emb, context=context)

        h = self.final(h)

        return h
