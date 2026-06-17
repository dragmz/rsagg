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

### Passing extra options to the OpenCL compiler

Arbitrary options can be appended verbatim to the `clBuildProgram` call with the
`--cl-build-options` flag. `optimize` persists the value into `bacon.json` so
`generate` reuses it.

```bash
cargo run --release generate --cl-build-options "-cl-nv-maxrregcount=96" --batch BATCH_SIZE PREFIX
# tune & persist into bacon.json
cargo run --release optimize --cl-build-options "-cl-nv-maxrregcount=96" PREFIX
```

> **Note (NVIDIA):** this kernel is register-heavy and occupancy-bound, so
> capping registers-per-thread with `-cl-nv-maxrregcount=<N>` can boost
> throughput a lot (~+30% on a Maxwell GTX 950 at `N=96`). The optimum is
> hardware-specific with a sharp cliff, so tune per GPU (try 80/88/96/104).
> NVIDIA-only — do not pass it on other vendors.

## Provider-specific instructions

### vast.ai

1. Go to the vast.ai template: https://cloud.vast.ai/?template_id=85b0923f64acdb8521825c54e369fac3
2. Select a machine of your choice and click "Rent"
3. Connect to the machine
4. Run ./bacon optimize / generate as described above

## Security notes

The utility hasn't been audited so please make sure to rekey any accounts for the addresses generated with it.
