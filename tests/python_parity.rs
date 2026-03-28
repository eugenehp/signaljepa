use burn::backend::NdArray as B;
use burn::prelude::*;
use std::collections::HashMap;
use signal_jepa_rs::model::signal_jepa::{SignalJEPAPreLocal, DEFAULT_CONV_SPEC};

fn load_data() -> Option<HashMap<String, (Vec<f32>, Vec<usize>)>> {
    let path = "/tmp/sjepa_parity.safetensors";
    if !std::path::Path::new(path).exists() { eprintln!("Skipping: {path} not found"); return None; }
    let bytes = std::fs::read(path).unwrap();
    let st = safetensors::SafeTensors::deserialize(&bytes).unwrap();
    let mut m = HashMap::new();
    for (k, v) in st.tensors() {
        let shape: Vec<usize> = v.shape().to_vec();
        let f32s: Vec<f32> = v.data().chunks_exact(4).map(|b| f32::from_le_bytes([b[0],b[1],b[2],b[3]])).collect();
        m.insert(k.to_string(), (f32s, shape));
    }
    Some(m)
}

fn t1(d: &HashMap<String,(Vec<f32>,Vec<usize>)>, k: &str, dev: &burn::backend::ndarray::NdArrayDevice) -> Tensor<B,1> { let (v,s)=&d[k]; Tensor::from_data(TensorData::new(v.clone(),s.clone()),dev) }
fn t2(d: &HashMap<String,(Vec<f32>,Vec<usize>)>, k: &str, dev: &burn::backend::ndarray::NdArrayDevice) -> Tensor<B,2> { let (v,s)=&d[k]; Tensor::from_data(TensorData::new(v.clone(),s.clone()),dev) }
fn t3(d: &HashMap<String,(Vec<f32>,Vec<usize>)>, k: &str, dev: &burn::backend::ndarray::NdArrayDevice) -> Tensor<B,3> { let (v,s)=&d[k]; Tensor::from_data(TensorData::new(v.clone(),s.clone()),dev) }
fn t4(d: &HashMap<String,(Vec<f32>,Vec<usize>)>, k: &str, dev: &burn::backend::ndarray::NdArrayDevice) -> Tensor<B,4> { let (v,s)=&d[k]; Tensor::from_data(TensorData::new(v.clone(),s.clone()),dev) }

#[test]
fn test_python_parity() {
    let dev = burn::backend::ndarray::NdArrayDevice::Cpu;
    let data = match load_data() { Some(d) => d, None => return };

    let (inp_data, inp_shape) = &data["_input"];
    let (out_data, _) = &data["_output"];
    let n_chans = inp_shape[1];
    let n_times = inp_shape[2];

    let mut model = SignalJEPAPreLocal::<B>::new(4, n_chans, n_times, 4, &DEFAULT_CONV_SPEC, &dev);

    // Load weights
    // spatial_conv
    let w = t4(&data, "spatial_conv.1.weight", &dev);
    let b = t1(&data, "spatial_conv.1.bias", &dev);
    model.spatial_conv.weight = model.spatial_conv.weight.clone().map(|_| w);
    if let Some(ref bias) = model.spatial_conv.bias { model.spatial_conv.bias = Some(bias.clone().map(|_| b)); }

    // feature_encoder blocks
    for (block_idx, layer_idx) in [(0usize, 1usize), (1, 2), (2, 3), (3, 4), (4, 5)] {
        let block = &mut model.feature_encoder.blocks[block_idx];
        let w = t3(&data, &format!("feature_encoder.{layer_idx}.0.weight"), &dev);
        block.conv.weight = block.conv.weight.clone().map(|_| w);
        if block_idx == 0 {
            if let Some(ref mut norm) = block.norm {
                let w = t1(&data, &format!("feature_encoder.{layer_idx}.2.weight"), &dev);
                let b = t1(&data, &format!("feature_encoder.{layer_idx}.2.bias"), &dev);
                if let Some(ref gamma) = norm.gamma { norm.gamma = Some(gamma.clone().map(|_| w)); }
                if let Some(ref beta) = norm.beta { norm.beta = Some(beta.clone().map(|_| b)); }
            }
        }
    }

    // final_layer
    let w = t2(&data, "final_layer.1.weight", &dev);
    let b = t1(&data, "final_layer.1.bias", &dev);
    model.final_linear.weight = model.final_linear.weight.clone().map(|_| w.transpose());
    if let Some(ref bias) = model.final_linear.bias { model.final_linear.bias = Some(bias.clone().map(|_| b)); }

    let input = Tensor::<B, 3>::from_data(TensorData::new(inp_data.clone(), inp_shape.clone()), &dev);
    let output = model.forward(input);
    let out_vec = output.into_data().to_vec::<f32>().unwrap();

    eprintln!("Expected: {:?}", out_data);
    eprintln!("Got:      {:?}", out_vec);
    let max_diff: f32 = out_vec.iter().zip(out_data.iter()).map(|(a,b)|(a-b).abs()).fold(0.0f32, f32::max);
    eprintln!("Max diff: {:.6e}", max_diff);
    assert!(max_diff < 0.01, "Parity failed: {:.6e}", max_diff);
}
