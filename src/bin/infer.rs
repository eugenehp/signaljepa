use burn::prelude::*;
use std::time::Instant;
#[cfg(feature = "ndarray")]
mod backend { pub use burn::backend::NdArray as B; pub fn device() -> burn::backend::ndarray::NdArrayDevice { burn::backend::ndarray::NdArrayDevice::Cpu } }
#[cfg(all(feature = "wgpu", not(feature = "ndarray")))]
mod backend { pub use burn::backend::Wgpu as B; pub fn device() -> burn::backend::wgpu::WgpuDevice { burn::backend::wgpu::WgpuDevice::DefaultDevice } }
use backend::{B, device};

fn main() -> anyhow::Result<()> {
    let dev = device();
    let model = signal_jepa_rs::model::signal_jepa::SignalJEPAPreLocal::<B>::new(
        4, 8, 640, 4, &signal_jepa_rs::model::signal_jepa::DEFAULT_CONV_SPEC, &dev,
    );
    let x = Tensor::<B, 3>::ones([1, 8, 640], &dev).mul_scalar(0.1f32);
    let t0 = Instant::now();
    let out = model.forward(x);
    println!("Output: {:?} ({:.1} ms)", out.dims(), t0.elapsed().as_secs_f64()*1000.0);
    Ok(())
}
