# rsagg

RuSt Algorand GPU vanity address Generator

<p align="center">
  <img width="300" height="300" src="logo.jpeg">
</p>

## Usage

### Optimize batch size

```bash
cargo run --release optimize
```

Note down the best performance batch value for use in the generator.

### Run the generator

```bash
cargo run --release generate --batch BATCH_SIZE PREFIX
```

## Device Compatibility

### AMD Radeon devices
The OpenCL kernel has been optimized for compatibility with AMD Radeon devices (including RX 5500 series and newer). If you encounter compilation errors:

1. The software will automatically try OpenCL 2.0 if OpenCL 3.0 compilation fails
2. You can specify a custom kernel file using `--kernel path/to/kernel.cl` if needed
3. Make sure your AMD drivers are up to date

### NVIDIA devices
All NVIDIA devices with OpenCL support should work out of the box.

## Provider-specific instructions

### vast.ai

1. Go to the vast.ai template: https://cloud.vast.ai/?template_id=85b0923f64acdb8521825c54e369fac3
2. Select a machine of your choice and click "Rent"
3. Connect to the machine
4. Run ./bacon optimize / generate as described above

## Security notes

The utility hasn't been audited so please make sure to rekey any accounts for the addresses generated with it.
