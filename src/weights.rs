/// Load SignalJEPA_PreLocal weights from safetensors.

use std::collections::HashMap;
use burn::prelude::*;
use half::bf16;
use safetensors::SafeTensors;
use crate::model::signal_jepa::SignalJEPAPreLocal;

pub struct WeightMap {
    pub tensors: HashMap<String, (Vec<f32>, Vec<usize>)>,
}

impl WeightMap {
    pub fn from_file(path: &str) -> anyhow::Result<Self> {
        let bytes = std::fs::read(path)?;
        let st = SafeTensors::deserialize(&bytes)?;
        let mut tensors = HashMap::with_capacity(st.len());
        for (key, view) in st.tensors() {
            let key = key.strip_prefix("model.").unwrap_or(&key).to_string();
            let shape: Vec<usize> = view.shape().to_vec();
            let data = view.data();
            let f32s: Vec<f32> = match view.dtype() {
                safetensors::Dtype::BF16 => data.chunks_exact(2).map(|b| bf16::from_le_bytes([b[0], b[1]]).to_f32()).collect(),
                safetensors::Dtype::F32 => data.chunks_exact(4).map(|b| f32::from_le_bytes([b[0],b[1],b[2],b[3]])).collect(),
                other => anyhow::bail!("unsupported dtype {:?}", other),
            };
            tensors.insert(key, (f32s, shape));
        }
        Ok(Self { tensors })
    }
    pub fn take<B: Backend, const N: usize>(&mut self, key: &str, device: &B::Device) -> anyhow::Result<Tensor<B, N>> {
        let (data, shape) = self.tensors.remove(key).ok_or_else(|| anyhow::anyhow!("key not found: {key}"))?;
        if shape.len() != N { anyhow::bail!("rank mismatch for {key}"); }
        Ok(Tensor::<B, N>::from_data(TensorData::new(data, shape), device))
    }
}

fn set_conv1d_w<B: Backend>(c: &mut burn::nn::conv::Conv1d<B>, w: Tensor<B, 3>) {
    c.weight = c.weight.clone().map(|_| w);
}
fn set_conv2d_wb<B: Backend>(c: &mut burn::nn::conv::Conv2d<B>, w: Tensor<B, 4>, b: Tensor<B, 1>) {
    c.weight = c.weight.clone().map(|_| w);
    if let Some(ref bias) = c.bias { c.bias = Some(bias.clone().map(|_| b)); }
}
fn set_gn<B: Backend>(g: &mut burn::nn::GroupNorm<B>, w: Tensor<B, 1>, b: Tensor<B, 1>) {
    if let Some(ref gamma) = g.gamma { g.gamma = Some(gamma.clone().map(|_| w)); }
    if let Some(ref beta) = g.beta { g.beta = Some(beta.clone().map(|_| b)); }
}
fn set_linear_wb<B: Backend>(l: &mut burn::nn::Linear<B>, w: Tensor<B, 2>, b: Tensor<B, 1>) {
    l.weight = l.weight.clone().map(|_| w.transpose());
    if let Some(ref bias) = l.bias { l.bias = Some(bias.clone().map(|_| b)); }
}

pub fn load_weights<B: Backend>(wm: &mut WeightMap, model: &mut SignalJEPAPreLocal<B>, dev: &B::Device) -> anyhow::Result<()> {
    // spatial_conv: Conv2d
    if let (Ok(w), Ok(b)) = (wm.take::<B,4>("spatial_conv.1.weight", dev), wm.take::<B,1>("spatial_conv.1.bias", dev)) {
        set_conv2d_wb(&mut model.spatial_conv, w, b);
    }

    // feature_encoder: blocks indexed as feature_encoder.{1,2,3,4,5}
    // Block 0 (index 1): Conv1d + GroupNorm + GELU
    // Blocks 1-4 (index 2-5): Conv1d + GELU (no norm)
    for (block_idx, layer_idx) in [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)] {
        let block = &mut model.feature_encoder.blocks[block_idx];
        let key = format!("feature_encoder.{layer_idx}.0.weight");
        if let Ok(w) = wm.take::<B, 3>(&key, dev) {
            set_conv1d_w(&mut block.conv, w);
        }
        // GroupNorm only on first block (layer_idx=1)
        if block_idx == 0 {
            if let (Ok(w), Ok(b)) = (
                wm.take::<B,1>(&format!("feature_encoder.{layer_idx}.2.weight"), dev),
                wm.take::<B,1>(&format!("feature_encoder.{layer_idx}.2.bias"), dev),
            ) {
                if let Some(ref mut norm) = block.norm { set_gn(norm, w, b); }
            }
        }
    }

    // final_layer: Flatten(1) + Linear
    if let (Ok(w), Ok(b)) = (wm.take::<B,2>("final_layer.1.weight", dev), wm.take::<B,1>("final_layer.1.bias", dev)) {
        set_linear_wb(&mut model.final_linear, w, b);
    }

    Ok(())
}
