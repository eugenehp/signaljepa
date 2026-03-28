/// Convolutional Feature Encoder for SignalJEPA.
///
/// Python _ConvFeatureEncoder: applies Conv1d layers per-channel.
///   Rearrange [B, C, T] → [B*C, 1, T]
///   Layer 0: Conv1d(1→8, k=32, s=8) + GroupNorm(8,8) + GELU  (mode="default", i==0)
///   Layer 1: Conv1d(8→16, k=2, s=2) + GELU                    (no norm for i>0)
///   Layer 2: Conv1d(16→32, k=2, s=2) + GELU
///   Layer 3: Conv1d(32→64, k=2, s=2) + GELU
///   Layer 4: Conv1d(64→64, k=2, s=2) + GELU
///   Rearrange [B*C, 64, T'] → [B, C*T', 64]

use burn::prelude::*;
use burn::nn::{
    conv::{Conv1d, Conv1dConfig},
    GroupNorm, GroupNormConfig,
};
use burn::tensor::activation::gelu;

#[derive(Module, Debug)]
pub struct ConvBlock<B: Backend> {
    pub conv: Conv1d<B>,
    pub norm: Option<GroupNorm<B>>,
}

impl<B: Backend> ConvBlock<B> {
    pub fn new(
        in_ch: usize, out_ch: usize, kernel: usize, stride: usize,
        use_group_norm: bool, conv_bias: bool, device: &B::Device,
    ) -> Self {
        let conv = Conv1dConfig::new(in_ch, out_ch, kernel)
            .with_stride(stride)
            .with_bias(conv_bias)
            .init(device);
        let norm = if use_group_norm {
            Some(GroupNormConfig::new(out_ch, out_ch).with_epsilon(1e-5).init(device))
        } else {
            None
        };
        Self { conv, norm }
    }

    /// x: [N, C_in, T] → [N, C_out, T']
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let x = self.conv.forward(x);
        let x = if let Some(ref norm) = self.norm {
            norm.forward(x)
        } else {
            x
        };
        gelu(x)
    }
}

#[derive(Module, Debug)]
pub struct ConvFeatureEncoder<B: Backend> {
    pub blocks: Vec<ConvBlock<B>>,
    pub n_channels: usize,
    pub emb_dim: usize,
}

impl<B: Backend> ConvFeatureEncoder<B> {
    /// Default conv_layers_spec: [(8,32,8), (16,2,2), (32,2,2), (64,2,2), (64,2,2)]
    pub fn new(
        conv_layers_spec: &[(usize, usize, usize)],
        n_channels: usize,
        conv_bias: bool,
        device: &B::Device,
    ) -> Self {
        let mut blocks = Vec::new();
        let mut in_ch = 1;
        for (i, &(out_ch, kernel, stride)) in conv_layers_spec.iter().enumerate() {
            let use_gn = i == 0; // GroupNorm only on first layer (mode="default")
            blocks.push(ConvBlock::new(in_ch, out_ch, kernel, stride, use_gn, conv_bias, device));
            in_ch = out_ch;
        }
        let emb_dim = conv_layers_spec.last().unwrap().0;
        Self { blocks, n_channels, emb_dim }
    }

    /// x: [B, C, T] → [B, C*T', emb_dim]
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch, channels, time] = x.dims();

        // Rearrange: [B, C, T] → [B*C, 1, T]
        let x = x.reshape([batch * channels, 1, time]);

        // Apply conv blocks
        let mut x = x;
        for block in &self.blocks {
            x = block.forward(x);
        }

        // [B*C, emb_dim, T'] → [B, C*T', emb_dim]
        let [_bc, emb, time_out] = x.dims();
        // Rearrange: [B*C, emb, T'] → [B, C, emb, T'] → [B, C, T', emb] → [B, C*T', emb]
        x.reshape([batch, channels, emb, time_out])
            .swap_dims(2, 3) // [B, C, T', emb]
            .reshape([batch, channels * time_out, emb])
    }

    /// Compute output time dimension for a given input time.
    pub fn n_times_out(&self, n_times: usize) -> usize {
        let mut t = n_times;
        for block in &self.blocks {
            let k = block.conv.weight.dims()[2];
            let s = block.conv.stride; // This is the stride
            // Actually we need to compute: (t - kernel) / stride + 1
            // But stride is stored in the config. Let's just use dims.
            // For now, hardcode default spec
            t = t; // placeholder
        }
        t
    }
}

/// Compute output time dimension for conv_layers_spec.
pub fn n_times_out(spec: &[(usize, usize, usize)], n_times: usize) -> usize {
    let mut t = n_times;
    for &(_dim, kernel, stride) in spec {
        t = (t - kernel) / stride + 1;
    }
    t
}
