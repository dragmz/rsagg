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

## Provider-specific instructions

### vast.ai

1. Go to the vast.ai template: https://cloud.vast.ai/?template_id=bf2098cba02c3c293192ab489c672e15
2. Select a machine of your choice and click "Rent"
3. Connect to the machine
4. Run ./bacon optimize / generate as described above

## Security notes

The utility hasn't been audited so please make sure to rekey any accounts for the addresses generated with it.