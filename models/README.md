# models/

Place your TFLite model file here before running benchmarks.

## Recommended model

MobileNet v1 Quantized (INT8) — the same model used in the Arm Performix use-case:

```bash
# Download from TensorFlow Hub (ARM64 compatible)
wget https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_1.0_224_quant.tgz
tar -xzf mobilenet_v1_1.0_224_quant.tgz
mv mobilenet_v1_1.0_224_quant.tflite models/
```

The expected file path is:

```
models/mobilenet_v1_1.0_224_quant.tflite
```

This is the default value for the `--model` flag in the benchmark runner.
