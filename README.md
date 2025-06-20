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

## Provider-specific instructions

### vast.ai

1. Go to the vast.ai template: https://cloud.vast.ai/?template_id=85b0923f64acdb8521825c54e369fac3
2. Select a machine of your choice and click "Rent"
3. Connect to the machine
4. Run ./bacon optimize / generate as described above

## Security notes

The utility hasn't been audited so please make sure to rekey any accounts for the addresses generated with it.
