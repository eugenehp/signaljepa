use burn::backend::NdArray as B;
use burn::prelude::*;
use signal_jepa_rs::model::signal_jepa::{SignalJEPAPreLocal, DEFAULT_CONV_SPEC};

#[test]
fn test_forward_basic() {
    let dev = burn::backend::ndarray::NdArrayDevice::Cpu;
    let model = SignalJEPAPreLocal::<B>::new(4, 8, 640, 4, &DEFAULT_CONV_SPEC, &dev);
    let x = Tensor::<B, 3>::ones([1, 8, 640], &dev).mul_scalar(0.1);
    let out = model.forward(x);
    assert_eq!(out.dims(), [1, 4]);
    eprintln!("Output shape: {:?}", out.dims());
}
