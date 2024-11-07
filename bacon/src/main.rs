use agvg;

use agvg::bacon::{
    align_to_preferred_multiple, max_batch_size, prepare_prefixes, Callback, Context,
};

use algonaut_crypto;
use csv;
use std::io::BufRead;
use std::str::FromStr;

use clap::Parser;
use sha2::{Digest, Sha512_256};

#[derive(Parser)]
enum Cli {
    Generate(GenerateCommand),
    Optimize(OptimizeCommand),
}

#[derive(Parser)]
#[command(version, about, long_about = None)]
struct OptimizeCommand {
    /// prefixes to optimize for
    #[arg(default_value_t = String::from(""))]
    prefixes: String,
    #[arg(long, default_value_t = String::from(""), short = 'f')]
    file: String,
    #[arg(long, default_value_t = 1)]
    min: usize,
    #[arg(long, default_value_t = 0)]
    max: usize,
    #[arg(long, default_value_t = false)]
    ordered: bool,
    #[arg(long, default_value_t = String::from(""))]
    output: String,
    #[arg(long, default_value_t = 0)]
    seed_concurrency: usize,
    #[arg(long, default_value_t = 0)]
    worker_concurrency: usize,
    /// use CPU-assist mode
    #[arg(long, default_value_t = false)]
    cpu: bool,
    #[arg(long, default_value_t = false)]
    all: bool,
    #[arg(long, default_value_t = 0)]
    preheat_time: usize,
    #[arg(long, default_value_t = 0)]
    batch_time: usize,
    #[arg(long, default_value_t = String::from(""))]
    msig: String,
    #[arg(long, default_value_t = 0)]
    device: usize,
}

#[derive(Parser)]
#[command(version, about, long_about = None)]
struct GenerateCommand {
    /// prefixes to search for
    #[arg(default_value_t = String::from(""))]
    prefixes: String,

    #[arg(long, default_value_t = String::from(""), short = 'f')]
    file: String,

    #[arg(long, default_value_t = 0)]
    batch: usize,
    #[arg(long, default_value_t = 0)]
    seed_concurrency: usize,
    #[arg(long, default_value_t = 0)]
    worker_concurrency: usize,

    /// number of keys to generate
    #[arg(short, long, default_value_t = 1)]
    count: usize,
    #[arg(short, long, default_value_t = false)]
    benchmark: bool,
    /// use CPU-assist mode
    #[arg(long, default_value_t = false)]
    cpu: bool,

    #[arg(long, default_value_t = String::from(""))]
    output: String,

    #[arg(long, default_value_t = String::from(""))]
    msig: String,

    #[arg(long, default_value_t = 0)]
    device: usize,
}

fn read_prefixes_from_file(file: &str, prefixes: &mut Vec<String>) {
    if file != "" {
        let file = std::fs::File::open(file).unwrap();
        let reader = std::io::BufReader::new(file);

        for line in reader.lines() {
            prefixes.push(line.unwrap().to_string());
        }
    }
}

struct DummyCallback {}

impl Callback for DummyCallback {
    fn found(&mut self, _: &[u8]) -> bool {
        true
    }
}

struct PrintCallback {
    print: bool,
    writer: Option<csv::Writer<std::fs::File>>,
    found: usize,
    count: usize,
    msig: Option<[u8; 32]>,
}

impl Callback for PrintCallback {
    fn found(&mut self, key: &[u8]) -> bool {
        self.found += 1;

        if self.msig.is_some() {
            let preamble = [b"MultisigAddr\x01\x01", self.msig.unwrap().as_slice()].concat();
            let full_key = [preamble.as_slice(), key].concat();

            let digest = Sha512_256::digest(&full_key);

            let mut target_bytes = [0; 32];
            target_bytes.copy_from_slice(&digest);

            let mut dummy_bytes = [0; 32];
            dummy_bytes.copy_from_slice(&key);

            let base_addr = algonaut::core::Address::new(self.msig.unwrap());
            let dummy_addr = algonaut::core::Address::new(dummy_bytes);
            let target_addr = algonaut::core::Address::new(target_bytes);

            if self.print {
                println!("{},{},{}", target_addr, base_addr, dummy_addr);
            }

            if let Some(ref mut writer) = self.writer {
                writer
                    .write_record(&[target_addr.to_string(), base_addr.to_string(), dummy_addr.to_string()])
                    .unwrap();
                writer.flush().unwrap();
            }
        } else {
            let m = algonaut_crypto::mnemonic::from_key(key).unwrap();
            let acc = algonaut::transaction::account::Account::from_mnemonic(&m).unwrap();

            if self.print {
                println!("{},{}", acc.address(), m);
            }

            if let Some(ref mut writer) = self.writer {
                writer
                    .write_record(&[acc.address().to_string(), m])
                    .unwrap();
                writer.flush().unwrap();
            }
        }

        self.found < self.count
    }
}

fn main() {
    let args = Cli::parse();
    match args {
        Cli::Generate(args) => generate(args),
        Cli::Optimize(args) => optimize(args),
    }
}

fn generate(args: GenerateCommand) {
    let msig = if !args.msig.is_empty() {
        Some(
            algonaut::core::Address::from_str(args.msig.as_str())
                .unwrap()
                .0,
        )
    } else {
        None
    };

    let ctx = Context::new(args.cpu, msig, args.device);

    let mut prefixes = vec![args.prefixes];
    read_prefixes_from_file(&args.file, &mut prefixes);

    let cb: Box<dyn Callback + Send> = Box::new(PrintCallback {
        print: args.output == "",
        writer: if args.output != "" {
            let file = std::fs::File::create(args.output).unwrap();
            let writer = csv::Writer::from_writer(file);
            Some(writer)
        } else {
            None
        },
        found: 0,
        count: args.count,
        msig,
    });

    let init = ctx.prepare(&prefixes);
    unsafe {
        let mut runner = init.prepare(
            args.batch,
            args.seed_concurrency,
            args.worker_concurrency,
            Some(cb),
        );

        let start = std::time::Instant::now();
        let mut total = 0;

        let mut last_benchmark_report = std::time::Instant::now();

        loop {
            let batch_start = std::time::Instant::now();

            let (processed, stop) = runner.step();
            total += processed;

            if args.benchmark
                && !stop
                && total > 0
                && last_benchmark_report.elapsed() >= std::time::Duration::from_secs(1)
            {
                last_benchmark_report = std::time::Instant::now();
                let now = std::time::Instant::now();
                let total_elapsed: std::time::Duration = now.duration_since(start);
                let batch_elapsed: std::time::Duration = now.duration_since(batch_start);

                let performance = total as f64 / total_elapsed.as_secs_f64();
                let batch_performance = runner.batch_size() as f64 / batch_elapsed.as_secs_f64();

                println!(
                    "Elapsed: {}.{:03}s, total: {}, avg/s: {}, last/s: {}",
                    total_elapsed.as_secs(),
                    total_elapsed.subsec_millis(),
                    total,
                    performance as usize,
                    batch_performance as usize,
                );
            }

            if stop {
                break;
            }
        }
    }
}

fn optimize(args: OptimizeCommand) {
    let msig = if !args.msig.is_empty() {
        Some(
            algonaut::core::Address::from_str(args.msig.as_str())
                .unwrap()
                .0,
        )
    } else {
        None
    };

    let ctx = Context::new(args.cpu, msig, args.device);

    let mut prefixes = vec![args.prefixes];
    read_prefixes_from_file(&args.file, &mut prefixes);

    prefixes = prepare_prefixes(&prefixes);

    if args.file != "" {
        let file = std::fs::File::open(args.file).unwrap();
        let reader = std::io::BufReader::new(file);
        for line in reader.lines() {
            prefixes.push(line.unwrap().trim().to_string());
        }
    }

    if prefixes.len() == 0 {
        prefixes.push("AAAAAAAAAA".to_string());
    }

    let preferred_multiple = ctx.preferred_multiple();

    let from_batch_size = align_to_preferred_multiple(args.min, preferred_multiple);
    let to_batch_size = match args.max {
        0 => max_batch_size(&ctx.device(), preferred_multiple),
        _ => align_to_preferred_multiple(args.max, preferred_multiple),
    };

    let mut current_batch_size = from_batch_size;

    let mut best_batch_size = 0;
    let mut best_performance = 0 as f64;

    let mut f = match args.output.as_str() {
        "" => None,
        output_path => Some(csv::WriterBuilder::new().from_path(output_path).unwrap()),
    };

    let init = ctx.prepare(&prefixes);

    unsafe {
        loop {
            if !args.ordered {
                let rnd = rand::random::<usize>();
                let val = match to_batch_size - from_batch_size {
                    0 => 0,
                    x => rnd % x,
                };

                current_batch_size =
                    align_to_preferred_multiple(val + from_batch_size, preferred_multiple);
            }

            let mut runner = init.prepare(
                current_batch_size,
                args.seed_concurrency,
                args.worker_concurrency,
                None,
            );

            let mut total = 0;

            let preheat_start = std::time::Instant::now();
            {
                let mut preheat_processed = 0;

                loop {
                    let (processed, _) = runner.step();

                    preheat_processed += processed;
                    if preheat_processed > runner.batch_size() * 2 {
                        let preheat_time = preheat_start.elapsed();
                        if preheat_time.as_millis() >= args.preheat_time as u128 {
                            break;
                        }
                    }
                }
            }

            let start = std::time::Instant::now();

            loop {
                let (processed, _) = runner.step();
                total += processed;

                let elapsed = start.elapsed();
                if elapsed.is_zero() {
                    continue;
                }

                let performance = total as f64 / elapsed.as_secs_f64();

                if performance as usize > 0
                    && total as f64 >= performance
                    && (args.batch_time == 0 || elapsed.as_millis() >= args.batch_time as u128)
                {
                    if performance > best_performance {
                        best_batch_size = current_batch_size;
                        best_performance = performance;

                        println!(
                            "Best batch size: {}, performance: {}",
                            best_batch_size, best_performance as usize
                        );
                    } else {
                        if args.all {
                            println!(
                                "Batch size: {}, performance: {}",
                                current_batch_size, performance as usize
                            );
                        }
                    }

                    match f {
                        Some(ref mut f) => {
                            f.write_record(&[
                                current_batch_size.to_string(),
                                (performance as usize).to_string(),
                            ])
                            .unwrap();
                            f.flush().unwrap();
                        }
                        _ => {}
                    }

                    break;
                }
            }

            if args.ordered {
                current_batch_size += preferred_multiple;
                if current_batch_size > to_batch_size {
                    break;
                }
            }
        }
    }

    println!(
        "Done. Best batch size: {}, performance: {}",
        best_batch_size, best_performance as usize
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_optimize() {
        let multiple = {
            let ctx = Context::new(false, None, 0);
            ctx.preferred_multiple()
        };

        optimize(OptimizeCommand {
            prefixes: "".to_string(),
            file: "".to_string(),
            min: multiple,
            max: multiple,
            ordered: true,
            output: "".to_string(),
            seed_concurrency: 0,
            worker_concurrency: 0,
            cpu: false,
            all: false,
            preheat_time: 0,
            batch_time: 0,
            msig: String::from(""),
            device: 0,
        });
    }
    #[test]
    fn test_generate() {
        let ctx = Context::new(false, None, 0);
        let init = ctx.prepare(&vec!["A".to_string()]);

        unsafe {
            let mut runner = init.prepare(32, 2, 2, None);
            let first = runner.step();
            assert_eq!(first, (0, false));

            let second = runner.step();
            assert_eq!(second.1, false);
        }
    }
}
