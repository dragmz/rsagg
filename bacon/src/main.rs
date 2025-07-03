use agvg;

use agvg::bacon::{
    BenchmarkCallback, Callback, Context, OptimizeCallback, Optimizer, align_to_preferred_multiple,
    max_batch_size, prepare_prefixes,
};

use algonaut_crypto;
use csv;
use std::io::{BufRead, Read};
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
    #[arg(long, default_value_t = String::from(""))]
    config: String,
    #[arg(long, default_value_t = String::from(""))]
    kernel: String,
    #[arg(long, default_value_t = 0)]
    iterations: usize,
    #[arg(long, default_value_t = 0)]
    time: usize,
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

    #[arg(long, default_value_t = String::from(""))]
    config: String,

    #[arg(long, default_value_t = String::from(""))]
    kernel: String,
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

struct OptimizeUpdateCallback {
    writer: Option<csv::Writer<std::fs::File>>,
}

impl OptimizeCallback for OptimizeUpdateCallback {
    fn result(&mut self, batch_size: usize, performance: usize) {
        match self.writer {
            Some(ref mut f) => {
                f.write_record(&[batch_size.to_string(), (performance as usize).to_string()])
                    .unwrap();
                f.flush().unwrap();
            }
            _ => {}
        }
    }
}

struct OptimizeResultCallback {
    path: String,
}

impl OptimizeCallback for OptimizeResultCallback {
    fn result(&mut self, batch_size: usize, performance: usize) {
        println!(
            "Best batch size: {}, performance: {}",
            batch_size, performance
        );

        let config = Config { batch: batch_size };

        let config_str = serde_json::to_string(&config).unwrap();
        std::fs::write(self.path.clone(), config_str).unwrap();
    }
}

struct PrintBenchmarkCallback {}

impl BenchmarkCallback for PrintBenchmarkCallback {
    fn result(
        &mut self,
        total_elapsed: std::time::Duration,
        total: usize,
        performance: usize,
        batch_performance: usize,
    ) {
        println!(
            "Elapsed: {}.{:03}s, total: {}, avg/s: {}, last/s: {}",
            total_elapsed.as_secs(),
            total_elapsed.subsec_millis(),
            total,
            performance as usize,
            batch_performance as usize,
        );
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
                    .write_record(&[
                        target_addr.to_string(),
                        base_addr.to_string(),
                        dummy_addr.to_string(),
                    ])
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
    match Cli::try_parse() {
        Ok(args) => match args {
            Cli::Generate(args) => generate(args),
            Cli::Optimize(args) => optimize(args),
        },
        Err(_) => {
            if config_exists("bacon.json") {
                generate(GenerateCommand::parse());
            } else {
                optimize(OptimizeCommand::parse());
            }
        }
    };
}

#[derive(serde::Deserialize, serde::Serialize)]
struct Config {
    batch: usize,
}

fn config_exists(config: &str) -> bool {
    std::fs::metadata(config).is_ok()
}

const DEFAULT_KERNEL: &str = include_str!("../../kernel.cl");

fn load_kernel(kernel: &str) -> String {
    if kernel == "" {
        return DEFAULT_KERNEL.to_string();
    }

    let file = match std::fs::File::open(kernel) {
        Ok(file) => file,
        Err(_) => panic!("Kernel file not found: {}", kernel),
    };

    let mut reader = std::io::BufReader::new(file);

    let mut kernel = String::new();
    reader.read_to_string(&mut kernel).unwrap();

    kernel
}

fn load_config(config: &str) -> Option<Config> {
    if config == "" {
        return None;
    }

    let file = match std::fs::File::open(config) {
        Ok(file) => file,
        Err(_) => return None,
    };

    let reader = std::io::BufReader::new(file);

    serde_json::from_reader(reader).unwrap()
}

fn generate(args: GenerateCommand) {
    println!("Running generate");

    let msig = if !args.msig.is_empty() {
        Some(
            algonaut::core::Address::from_str(args.msig.as_str())
                .unwrap()
                .0,
        )
    } else {
        None
    };

    let mut batch = args.batch;

    let config_path = match args.config.as_str() {
        "" => match batch {
            0 => "bacon.json",
            _ => "",
        }
        .to_string(),
        path => path.to_string(),
    };

    if config_path != "" {
        match load_config(&config_path) {
            Some(config) => {
                if args.batch == 0 || !args.config.is_empty() {
                    println!("Loaded config - {}, batch: {}", config_path, config.batch);
                    batch = config.batch;
                }
            }
            None => {
                if args.batch == 0 {
                    panic!("Config file not found: {}", config_path);
                }
            }
        }
    }

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

    let benchmark_cb = match args.benchmark {
        true => Some(Box::new(PrintBenchmarkCallback {}) as Box<dyn BenchmarkCallback + Send>),
        false => None,
    };

    let kernel = load_kernel(&args.kernel);
    let ctx = Context::new(args.cpu, msig, args.device, kernel);
    let init = ctx.prepare(&prefixes);
    let generator = agvg::bacon::Generator::new(init);
    generator.run(
        batch,
        args.seed_concurrency,
        args.worker_concurrency,
        args.benchmark,
        Some(cb),
        benchmark_cb,
    );
}

fn optimize(args: OptimizeCommand) {
    println!("Running optimize");

    let msig = if !args.msig.is_empty() {
        Some(
            algonaut::core::Address::from_str(args.msig.as_str())
                .unwrap()
                .0,
        )
    } else {
        None
    };

    let mut prefixes = vec![args.prefixes];
    read_prefixes_from_file(&args.file, &mut prefixes);

    prefixes = prepare_prefixes(&prefixes);

    if prefixes.len() == 0 {
        prefixes.push("AAAAAAAAAA".to_string());
    }

    let update_cb = match args.output.as_str() {
        "" => None,
        output_path => {
            let cb: Box<dyn OptimizeCallback + Send> = Box::new({
                let f = Some(csv::WriterBuilder::new().from_path(output_path).unwrap());

                OptimizeUpdateCallback { writer: f }
            });

            Some(cb)
        }
    };

    let config_path = match args.config.as_str() {
        "" => "bacon.json".to_string(),
        path => path.to_string(),
    };

    println!("Config path: {}", config_path);

    let result_cb = match config_path.as_str() {
        "" => None,
        path => {
            let cb: Box<dyn OptimizeCallback + Send> = Box::new(OptimizeResultCallback {
                path: path.to_string(),
            });

            Some(cb)
        }
    };

    let kernel = load_kernel(&args.kernel);
    let ctx = Context::new(args.cpu, msig, args.device, kernel);

    let preferred_multiple = ctx.preferred_multiple();
    let from_batch_size = align_to_preferred_multiple(args.min, preferred_multiple);
    let to_batch_size = match args.max {
        0 => max_batch_size(&ctx.device(), preferred_multiple),
        _ => align_to_preferred_multiple(args.max, preferred_multiple),
    };

    let init = ctx.prepare(&prefixes);
    let optimizer = Optimizer::new(init);

    let (best_batch_size, best_performance) = optimizer.run(
        preferred_multiple,
        from_batch_size,
        to_batch_size,
        args.iterations,
        args.batch_time,
        args.all,
        args.ordered,
        args.seed_concurrency,
        args.worker_concurrency,
        args.preheat_time,
        args.time,
        result_cb,
        update_cb,
    );

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
            let ctx = Context::new(false, None, 0, DEFAULT_KERNEL.to_string());
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
            config: String::from(""),
            kernel: String::from(""),
            iterations: 0,
            time: 0,
        });
    }
    #[test]
    fn test_generate() {
        let ctx = Context::new(false, None, 0, DEFAULT_KERNEL.to_string());
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
