# signal-jepa-rs

Pure-Rust inference for the **SignalJEPA** (S-JEPA PreLocal) EEG foundation model, built on [Burn 0.20](https://burn.dev).

SignalJEPA uses a Joint Embedding Predictive Architecture for self-supervised EEG pretraining with dynamic spatial attention. The PreLocal variant applies spatial convolution followed by a multi-layer convolutional feature encoder.

## Architecture

```
EEG [B, C, T]
    │
    ├─ Spatial Conv2d(1, n_spat, (C, 1)) → [B, n_spat, T]
    │
    ├─ Conv Feature Encoder (5 layers, per-channel)
    │  ├─ Conv1d(1→8, k=32, s=8) + GroupNorm + GELU
    │  ├─ Conv1d(8→16, k=2, s=2) + GELU
    │  ├─ Conv1d(16→32, k=2, s=2) + GELU
    │  ├─ Conv1d(32→64, k=2, s=2) + GELU
    │  └─ Conv1d(64→64, k=2, s=2) + GELU
    │  → [B, n_spat × T', 64]
    │
    └─ Flatten + Linear → [B, n_outputs]
```

## Performance

| | Python (PyTorch) | Rust (NdArray) | Speedup |
|---|:---:|:---:|:---:|
| 8ch × 640t | 1.03 ms | **0.38 ms** | **2.7x** |

### Numerical Parity

Python ↔ Rust output difference: **< 1.8×10⁻⁷**

## Build

```bash
cargo build --release
cargo build --release --features blas-accelerate
```

## Pretrained Weights

Available on [HuggingFace](https://huggingface.co/braindecode/SignalJEPA-PreLocal-pretrained).

## Citation

```bibtex
@inproceedings{guetschel2024sjepa,
    title     = {S-{JEPA}: towards seamless cross-dataset transfer through dynamic spatial attention},
    author    = {Guetschel, Pierre and Moreau, Thomas and Tangermann, Michael},
    booktitle = {9th Graz Brain-Computer Interface Conference},
    year      = {2024},
    doi       = {10.3217/978-3-99161-014-4-003}
}

@software{hauptmann2025sjeparustinference,
    title     = {signal-jepa-rs: {SignalJEPA} {EEG} Foundation Model Inference in Rust},
    author    = {Hauptmann, Eugene},
    year      = {2025},
    url       = {https://github.com/eugenehp/signal-jepa-rs},
    version   = {0.0.1}
}
```

## Author

[Eugene Hauptmann](https://github.com/eugenehp)

## License

Apache-2.0
