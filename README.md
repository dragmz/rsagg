# rsagg

RuSt Algorand GPU vanity address Generator

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