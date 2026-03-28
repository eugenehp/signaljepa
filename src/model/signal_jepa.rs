/// SignalJEPA_PreLocal — full model.
///
/// Architecture (PreLocal variant):
///   1. spatial_conv: [B, 1, C, T] → Conv2d(1, n_spat, (C,1)) → [B, n_spat, 1, T] → [B, n_spat, T]
///   2. feature_encoder: [B, n_spat, T] → per-channel conv pipeline → [B, n_spat*T', emb_dim]
///   3. final_layer: Flatten → Linear → [B, n_outputs]

use burn::prelude::*;
use burn::nn::{
    Linear, LinearConfig,
    conv::{Conv2d, Conv2dConfig},
};
use crate::model::conv_encoder::{ConvFeatureEncoder, n_times_out};

pub const DEFAULT_CONV_SPEC: [(usize, usize, usize); 5] = [
    (8, 32, 8),
    (16, 2, 2),
    (32, 2, 2),
    (64, 2, 2),
    (64, 2, 2),
];

#[derive(Module, Debug)]
pub struct SignalJEPAPreLocal<B: Backend> {
    pub spatial_conv: Conv2d<B>,
    pub feature_encoder: ConvFeatureEncoder<B>,
    pub final_linear: Linear<B>,
    pub n_chans: usize,
    pub n_spat_filters: usize,
    pub n_outputs: usize,
}

impl<B: Backend> SignalJEPAPreLocal<B> {
    pub fn new(
        n_outputs: usize, n_chans: usize, n_times: usize,
        n_spat_filters: usize,
        conv_spec: &[(usize, usize, usize)],
        device: &B::Device,
    ) -> Self {
        let spatial_conv = Conv2dConfig::new([1, n_spat_filters], [n_chans, 1])
            .with_bias(true)
            .init(device);

        let feature_encoder = ConvFeatureEncoder::new(
            conv_spec, n_spat_filters, false, device,
        );

        let time_out = n_times_out(conv_spec, n_times);
        let emb_dim = conv_spec.last().unwrap().0;
        let flat_dim = n_spat_filters * time_out * emb_dim;

        let final_linear = LinearConfig::new(flat_dim, n_outputs)
            .with_bias(true)
            .init(device);

        Self {
            spatial_conv, feature_encoder, final_linear,
            n_chans, n_spat_filters, n_outputs,
        }
    }

    /// x: [B, C, T] → [B, n_outputs]
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 2> {
        let [batch, n_chans, n_times] = x.dims();

        // 1. Spatial conv: [B, C, T] → [B, 1, C, T] → Conv2d → [B, n_spat, 1, T] → [B, n_spat, T]
        let x = x.unsqueeze_dim::<4>(1); // [B, 1, C, T]
        let x = self.spatial_conv.forward(x); // [B, n_spat, 1, T]
        let [b, ns, _, t] = x.dims();
        let x = x.reshape([b, ns, t]); // [B, n_spat, T]

        // 2. Feature encoder: [B, n_spat, T] → [B, n_spat*T', emb_dim]
        let x = self.feature_encoder.forward(x);

        // 3. Flatten + Linear
        let [b, seq, emb] = x.dims();
        let x = x.reshape([b, seq * emb]);
        self.final_linear.forward(x)
    }
}
