use opencl3::{
    self,
    command_queue::{
        CommandQueue, enqueue_nd_range_kernel, enqueue_read_buffer, enqueue_write_buffer,
    },
    device::Device,
    event::{Event, wait_for_events},
    kernel::Kernel,
    memory::{Buffer, CL_MEM_HOST_WRITE_ONLY, CL_MEM_READ_ONLY, CL_MEM_WRITE_ONLY, ClMem},
    program::CL_STD_3_0,
    types::{CL_FALSE, CL_TRUE},
};
use rand::thread_rng;
use rand_core::RngCore;
use rayon::prelude::*;
use std::{
    cmp::max,
    os::raw::c_void,
    ptr,
    sync::{Arc, Barrier, Mutex},
};
use std::{mem::take, sync::mpsc::sync_channel};

struct Job {
    e: Event,
    seeds: Vec<u8>,
    counts: Vec<u8>,
    addrs: Vec<u8>,
}

unsafe impl Send for Worker {}

const KEY_SIZE: usize = 32;

pub struct Generator<'a> {
    init: Initializer<'a>,
}

impl<'a> Generator<'a> {
    pub fn new(init: Initializer<'a>) -> Self {
        Self { init }
    }

    pub fn run(
        &self,
        batch_size: usize,
        seed_concurrency: usize,
        worker_concurrency: usize,
        benchmark: bool,
        cb: Option<Box<dyn Callback + Send>>,
        benchmark_cb: Option<Box<dyn BenchmarkCallback + Send>>,
    ) {
        let benchmark_cb = Arc::new(Mutex::new(benchmark_cb));
        unsafe {
            let mut runner =
                self.init
                    .prepare(batch_size, seed_concurrency, worker_concurrency, cb);

            let start = std::time::Instant::now();
            let mut total = 0;

            let mut last_benchmark_report = std::time::Instant::now();

            loop {
                let batch_start = std::time::Instant::now();

                let (processed, stop) = runner.step();
                total += processed;

                if benchmark
                    && !stop
                    && total > 0
                    && last_benchmark_report.elapsed() >= std::time::Duration::from_secs(1)
                {
                    last_benchmark_report = std::time::Instant::now();
                    let now = std::time::Instant::now();
                    let total_elapsed: std::time::Duration = now.duration_since(start);
                    let batch_elapsed: std::time::Duration = now.duration_since(batch_start);

                    let performance = total as f64 / total_elapsed.as_secs_f64();
                    let batch_performance =
                        runner.batch_size() as f64 / batch_elapsed.as_secs_f64();

                    match *benchmark_cb.lock().unwrap() {
                        Some(ref mut cb) => {
                            cb.result(
                                total_elapsed,
                                total,
                                performance as usize,
                                batch_performance as usize,
                            );
                        }
                        _ => {}
                    }
                }

                if stop {
                    break;
                }
            }
        }
    }
}

pub struct Optimizer<'a> {
    init: Initializer<'a>,
}

impl<'a> Optimizer<'a> {
    pub fn new(init: Initializer<'a>) -> Self {
        Self { init }
    }

    pub fn run(
        &self,
        preferred_multiple: usize,
        from_batch_size: usize,
        to_batch_size: usize,
        iterations: usize,
        batch_time: usize,
        all: bool,
        ordered: bool,
        seed_concurrency: usize,
        worker_concurrency: usize,
        preheat_time: usize,
        time: usize,
        result_cb: Option<Box<dyn OptimizeCallback + Send>>,
        update_cb: Option<Box<dyn OptimizeCallback + Send>>,
    ) -> (usize, usize) {
        let mut current_batch_size = from_batch_size;

        let mut best_batch_size = 0;
        let mut best_performance = 0 as f64;

        let result_cb = Arc::new(Mutex::new(result_cb));
        let update_cb = Arc::new(Mutex::new(update_cb));

        let mut iteration = 0;
        let start = std::time::Instant::now();
        unsafe {
            loop {
                if iterations > 0 {
                    if iteration < iterations {
                        iteration += 1;
                    } else {
                        println!("Reached max iterations: {}", iterations);
                        break;
                    }
                }

                if time > 0 {
                    let elapsed = start.elapsed();
                    if elapsed.as_millis() >= time as u128 {
                        println!("Reached max time: {}ms", time);
                        break;
                    }
                }

                if !ordered {
                    let rnd = rand::random::<usize>();
                    let val = match to_batch_size - from_batch_size {
                        0 => 0,
                        x => rnd % x,
                    };

                    current_batch_size =
                        align_to_preferred_multiple(val + from_batch_size, preferred_multiple);
                }

                let mut runner = self.init.prepare(
                    current_batch_size,
                    seed_concurrency,
                    worker_concurrency,
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
                            let preheat_diff = preheat_start.elapsed();
                            if preheat_diff.as_millis() >= preheat_time as u128 {
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
                        && (batch_time == 0 || elapsed.as_millis() >= batch_time as u128)
                    {
                        if performance > best_performance {
                            best_batch_size = current_batch_size;
                            best_performance = performance;

                            match *result_cb.lock().unwrap() {
                                Some(ref mut cb) => {
                                    cb.result(best_batch_size, best_performance as usize)
                                }
                                _ => {}
                            }
                        } else {
                            if all {
                                println!(
                                    "Batch size: {}, performance: {}",
                                    current_batch_size, performance as usize
                                );
                            }
                        }

                        match *update_cb.lock().unwrap() {
                            Some(ref mut cb) => cb.result(current_batch_size, performance as usize),
                            _ => {}
                        }

                        break;
                    }
                }

                if ordered {
                    current_batch_size += preferred_multiple;
                    if current_batch_size > to_batch_size {
                        break;
                    }
                }
            }
        }
        (best_batch_size, best_performance as usize)
    }
}

pub struct Context {
    device: opencl3::device::Device,
    context: opencl3::context::Context,
    program: opencl3::program::Program,
    cpu: bool,
    msig: Option<[u8; 32]>,
}

impl Context {
    pub fn device(&self) -> opencl3::device::Device {
        self.device
    }
}

fn default_device(index: usize) -> opencl3::device::Device {
    let platforms = opencl3::platform::get_platforms().unwrap();
    let platform = platforms[0];
    let devices = platform
        .get_devices(opencl3::device::CL_DEVICE_TYPE_GPU)
        .unwrap();

    if index >= devices.len() {
        panic!(
            "Invalid device index: {}, devices: {}",
            index,
            devices.len()
        );
    }

    opencl3::device::Device::new(devices[index])
}

pub fn preferred_multiple(device: &Device, kernel: &Kernel) -> usize {
    kernel.get_work_group_size_multiple(device.id()).unwrap()
}

pub fn prepare_prefixes(prefixes: &Vec<String>) -> Vec<String> {
    let mut all_prefixes = Vec::new();

    for prefixes_line in prefixes {
        let prefixes = prefixes_line.split(",").collect::<Vec<_>>().to_vec();

        for raw_prefix in prefixes {
            let prefix = raw_prefix.trim().to_uppercase();

            if prefix.len() == 0 {
                continue;
            }

            for c in prefix.as_bytes() {
                if !BASE32_ALPHABET.contains(c) {
                    panic!(
                        "Invalid prefix: '{}', unexpected character: '{}'",
                        prefix, *c as char
                    );
                }
            }

            all_prefixes.push(prefix);
        }
    }

    all_prefixes
}

const BASE32_ALPHABET: &[u8; 32] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZ234567";

impl Context {
    pub fn new(cpu: bool, msig: Option<[u8; 32]>, device: usize, kernel: String) -> Self {
        let args = {
            let mut args = Vec::from([CL_STD_3_0]);
            if cpu {
                args.push("-D CPU");
            }

            if msig.is_some() {
                args.push("-D MSIG");
            }

            args.join(" ")
        };

        let device = default_device(device);
        let context = opencl3::context::Context::from_device(&device).unwrap();

        let program = opencl3::program::Program::create_and_build_from_source(
            &context,
            &kernel,
            args.as_str(),
        )
        .unwrap();

        Self {
            device,
            context,
            program,
            cpu,
            msig,
        }
    }

    pub fn preferred_multiple(&self) -> usize {
        let kernel = Kernel::create(&self.program, "ed25519_create_keypair").unwrap();
        preferred_multiple(&self.device, &kernel)
    }

    pub fn prepare(&self, prefixes: &Vec<String>) -> Initializer {
        let mut prefix_chunks: Vec<u8> = Vec::new();

        let mut lengths = Vec::new();
        let mut prefix_count = 0;

        for prefix in prepare_prefixes(prefixes) {
            let prefix_bytes = prefix.as_bytes();
            let mut prefix_chunk = vec![0u8; 64];
            prefix_chunk[..prefix_bytes.len()].copy_from_slice(prefix_bytes);

            prefix_chunks.extend(prefix_chunk);

            let len = prefix.len();
            lengths.push(len as u8);
            prefix_count += 1;

            const MAX_PREFIX_COUNT: usize = std::u16::MAX as usize;

            if prefix_count > MAX_PREFIX_COUNT {
                panic!(
                    "Too many prefixes. Maximum allowed is {}.",
                    MAX_PREFIX_COUNT
                );
            }
        }

        if prefix_count == 0 {
            panic!("No prefixes provided");
        }

        Initializer {
            device: &self.device,
            context: &self.context,
            program: &self.program,
            prefix_count,
            lengths,
            prefix_chunks,
            cpu: self.cpu,
            msig: self.msig,
        }
    }
}

pub struct Initializer<'a> {
    device: &'a opencl3::device::Device,
    context: &'a opencl3::context::Context,
    program: &'a opencl3::program::Program,
    prefix_count: usize,
    lengths: Vec<u8>,
    prefix_chunks: Vec<u8>,
    cpu: bool,
    msig: Option<[u8; 32]>,
}

struct Worker {
    queue: CommandQueue,
    kernel: Kernel,
    counts_buffer: Buffer<u8>,
    seed_buffer: Buffer<u8>,
    addrs_buffer: Buffer<u8>,
    job_tx: std::sync::mpsc::SyncSender<Option<Job>>,
    t: std::thread::JoinHandle<()>,
    cpu: bool,
}

struct WorkerStatus {
    id: usize,
    last_batch: usize,
    done: bool,
}

pub struct Runner {
    size: usize,
    batch_size: usize,
    seeds_rx: Option<std::sync::mpsc::Receiver<Vec<u8>>>,
    _size_buffer: opencl3::memory::Buffer<u16>,
    _prefix_length_buffer: opencl3::memory::Buffer<u8>,
    _prefix_chunks_buffer: opencl3::memory::Buffer<u8>,
    _msig_buffer: Buffer<u8>,
    seed_threads: Vec<std::thread::JoinHandle<()>>,
    status_rx: Option<std::sync::mpsc::Receiver<WorkerStatus>>,
    workers: Vec<Worker>,
}

impl Drop for Runner {
    fn drop(&mut self) {
        let seeds_rx = take(&mut self.seeds_rx).unwrap();
        drop(seeds_rx);

        for t in self.seed_threads.drain(..) {
            t.join().unwrap();
        }

        let status_rx = take(&mut self.status_rx).unwrap();
        drop(status_rx);

        for w in self.workers.drain(..) {
            _ = w.job_tx.send(None);
            w.t.join().unwrap();

            w.queue.finish().unwrap();
        }
    }
}

impl Runner {
    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    pub unsafe fn step(&mut self) -> (usize, bool) {
        unsafe {
            let status = self.status_rx.as_ref().unwrap().recv().unwrap();
            let worker = self.workers.get_mut(status.id).unwrap();

            let job = match status.done {
                true => None,
                _ => {
                    let seeds = self.seeds_rx.as_ref().unwrap().recv().unwrap();

                    let copy_seeds_h2d_event = {
                        let e = enqueue_write_buffer(
                            worker.queue.get(),
                            worker.seed_buffer.get(),
                            CL_FALSE,
                            0,
                            seeds.len(),
                            seeds.as_ptr() as *const c_void,
                            0,
                            ptr::null(),
                        )
                        .unwrap();

                        Event::new(e)
                    };

                    let kernel_event = {
                        let events = &[copy_seeds_h2d_event.get()];

                        Event::new(
                            enqueue_nd_range_kernel(
                                worker.queue.get(),
                                worker.kernel.get(),
                                1,
                                ptr::null(),
                                &[self.batch_size] as *const usize,
                                ptr::null(),
                                events.len() as u32,
                                events.as_ptr(),
                            )
                            .unwrap(),
                        )
                    };

                    let mut counts = vec![0u8; self.batch_size * self.size];
                    let mut addrs = vec![0u8; self.batch_size * 54];

                    let copy_wait_events = &[kernel_event.get()];

                    let copy_event = match worker.cpu {
                        true => Event::new(
                            enqueue_read_buffer(
                                worker.queue.get(),
                                worker.addrs_buffer.get(),
                                CL_FALSE,
                                0,
                                addrs.len(),
                                addrs.as_mut_ptr() as *mut c_void,
                                copy_wait_events.len() as u32,
                                copy_wait_events.as_ptr(),
                            )
                            .unwrap(),
                        ),
                        false => Event::new(
                            enqueue_read_buffer(
                                worker.queue.get(),
                                worker.counts_buffer.get(),
                                CL_FALSE,
                                0,
                                counts.len(),
                                counts.as_mut_ptr() as *mut c_void,
                                copy_wait_events.len() as u32,
                                copy_wait_events.as_ptr(),
                            )
                            .unwrap(),
                        ),
                    };

                    Some(Job {
                        e: copy_event,
                        seeds,
                        counts,
                        addrs,
                    })
                }
            };

            worker.job_tx.send(job).unwrap();

            (status.last_batch, status.done)
        }
    }
}

pub trait Callback {
    fn found(&mut self, key: &[u8]) -> bool;
}

pub trait BenchmarkCallback {
    fn result(
        &mut self,
        total_elapsed: std::time::Duration,
        total: usize,
        performance: usize,
        batch_performance: usize,
    );
}

pub trait OptimizeCallback {
    fn result(&mut self, batch_size: usize, performance: usize);
}

impl<'a> Initializer<'a> {
    pub unsafe fn prepare(
        &'a self,
        batch_size: usize,
        seed_concurrency: usize,
        worker_concurrency: usize,
        cb: Option<Box<dyn Callback + Send>>,
    ) -> Runner {
        unsafe {
            let kernel = Kernel::create(&self.program, "ed25519_create_keypair").unwrap();

            let _batch_size = match batch_size {
                0 => max_batch_size(&self.device, preferred_multiple(self.device, &kernel)),
                _ => batch_size,
            };

            let _seed_concurrency = match seed_concurrency {
                0 => num_cpus::get(),
                _ => seed_concurrency,
            };

            let _worker_concurrency = match worker_concurrency {
                0 => num_cpus::get(),
                _ => worker_concurrency,
            };

            let mut seed_threads = Vec::new();

            let (seeds_tx, seeds_rx) = sync_channel(seed_concurrency * 2);

            let seed_workers_start_barrier = Arc::new(Barrier::new(_seed_concurrency + 1));

            for _ in 0.._seed_concurrency {
                let tx = seeds_tx.clone();
                let barrier = seed_workers_start_barrier.clone();

                let t = std::thread::spawn(move || {
                    let mut rng = thread_rng();

                    let mut next_seeds = vec![0u8; _batch_size * KEY_SIZE];
                    rng.fill_bytes(&mut next_seeds);

                    barrier.wait();

                    loop {
                        match tx.send(next_seeds) {
                            Ok(_) => {}
                            Err(_) => break,
                        }

                        next_seeds = vec![0u8; _batch_size * KEY_SIZE];
                        rng.fill_bytes(&mut next_seeds);
                    }
                });

                seed_threads.push(t);
            }

            seed_workers_start_barrier.wait();

            let (status_tx, status_rx) = sync_channel(seed_concurrency);

            let mut size_buffer: Buffer<u16> = Buffer::create(
                &self.context,
                CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY,
                1,
                ptr::null_mut(),
            )
            .unwrap();

            let mut prefix_length_buffer: Buffer<u8> = Buffer::create(
                &self.context,
                CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY,
                self.prefix_count,
                ptr::null_mut(),
            )
            .unwrap();

            let mut prefix_chunks_buffer: Buffer<u8> = Buffer::create(
                &self.context,
                CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY,
                self.prefix_chunks.len(),
                ptr::null_mut(),
            )
            .unwrap();

            let mut msig_buffer: Buffer<u8> = Buffer::create(
                &self.context,
                CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY,
                32,
                ptr::null_mut(),
            )
            .unwrap();

            {
                let queue =
                    CommandQueue::create_with_properties(self.context, self.device.id(), 0, 0)
                        .unwrap();

                {
                    let gpu_addr_test_count = match self.cpu {
                        true => 0,
                        _ => self.prefix_count as u16,
                    };

                    queue
                        .enqueue_write_buffer(
                            &mut size_buffer,
                            CL_TRUE,
                            0,
                            &[gpu_addr_test_count],
                            &[],
                        )
                        .unwrap();
                }

                queue
                    .enqueue_write_buffer(
                        &mut prefix_length_buffer,
                        CL_TRUE,
                        0,
                        self.lengths.as_slice(),
                        &[],
                    )
                    .unwrap();

                queue
                    .enqueue_write_buffer(
                        &mut prefix_chunks_buffer,
                        CL_TRUE,
                        0,
                        self.prefix_chunks.as_slice(),
                        &[],
                    )
                    .unwrap();

                if self.msig.is_some() {
                    queue
                        .enqueue_write_buffer(
                            &mut msig_buffer,
                            CL_TRUE,
                            0,
                            self.msig.unwrap().as_slice(),
                            &[],
                        )
                        .unwrap();
                }
            }

            let mut workers = Vec::new();

            let cb = Arc::new(Mutex::new(cb));

            let workers_start_barrier = Arc::new(Barrier::new(_worker_concurrency + 1));

            for id in 0.._worker_concurrency {
                let status_tx = status_tx.clone();
                let queue =
                    CommandQueue::create_with_properties(self.context, self.device.id(), 0, 0)
                        .unwrap();

                let seed_buffer: Buffer<u8> = Buffer::create(
                    &self.context,
                    CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY,
                    _batch_size * KEY_SIZE,
                    ptr::null_mut(),
                )
                .unwrap();

                let counts_buffer: Buffer<u8> = Buffer::create(
                    &self.context,
                    CL_MEM_WRITE_ONLY,
                    _batch_size * self.prefix_count,
                    ptr::null_mut(),
                )
                .unwrap();

                let addrs_buffer: Buffer<u8> = Buffer::create(
                    &self.context,
                    CL_MEM_WRITE_ONLY,
                    _batch_size * 54,
                    ptr::null_mut(),
                )
                .unwrap();

                let size = self.prefix_count;

                let kernel = Kernel::create(&self.program, "ed25519_create_keypair").unwrap();

                if self.cpu {
                    kernel.set_arg(0, &seed_buffer).unwrap();
                    kernel.set_arg(1, &prefix_length_buffer).unwrap();
                    kernel.set_arg(2, &prefix_chunks_buffer).unwrap();
                    kernel.set_arg(3, &addrs_buffer).unwrap();
                    if self.msig.is_some() {
                        kernel.set_arg(4, &msig_buffer).unwrap();
                    }
                } else {
                    kernel.set_arg(0, &seed_buffer).unwrap();
                    kernel.set_arg(1, &size_buffer).unwrap();
                    kernel.set_arg(2, &prefix_length_buffer).unwrap();
                    kernel.set_arg(3, &prefix_chunks_buffer).unwrap();
                    kernel.set_arg(4, &counts_buffer).unwrap();
                    if self.msig.is_some() {
                        kernel.set_arg(5, &msig_buffer).unwrap();
                    }
                }

                let cb = cb.clone();

                let (job_tx, job_rx) = sync_channel(0);
                let barrier = workers_start_barrier.clone();

                let prefix_chunks = self.prefix_chunks.clone();
                let lengths = self.lengths.clone();

                let cpu = self.cpu;

                let t = std::thread::spawn(move || {
                    let mut last_batch = 0;
                    let mut done = false;

                    barrier.wait();

                    loop {
                        let status = WorkerStatus {
                            id,
                            last_batch,
                            done,
                        };

                        _ = status_tx.send(status);

                        let job: Job = match job_rx.recv() {
                            Ok(Some(job)) => job,
                            _ => break,
                        };

                        wait_for_events(&[job.e.get()]).unwrap();

                        if cpu {
                            if false {
                                let seeds = (0..batch_size)
                                    .into_par_iter()
                                    .filter(|i| {
                                        let addr_chunk = &job.addrs[i * 54..(i + 1) * 54];
                                        (0..size).into_par_iter().any(|j| {
                                            let prefix_chunk = &prefix_chunks
                                                [j * 64..(j * 64) + lengths[j] as usize];

                                            addr_chunk.starts_with(prefix_chunk)
                                        })
                                    })
                                    .map(|i| &job.seeds[i * KEY_SIZE..(i + 1) * KEY_SIZE]);

                                done = seeds.any(|seed| match *cb.lock().unwrap() {
                                    Some(ref mut cb) => !cb.found(seed),
                                    _ => false,
                                });
                            } else {
                                for i in 0..batch_size {
                                    let addr_chunk = &job.addrs[i * 54..(i + 1) * 54];
                                    for j in 0..size {
                                        let prefix_chunk =
                                            &prefix_chunks[j * 64..(j * 64) + lengths[j] as usize];

                                        if addr_chunk.starts_with(prefix_chunk) {
                                            let bytes =
                                                &job.seeds[i * KEY_SIZE..(i + 1) * KEY_SIZE];

                                            match *cb.lock().unwrap() {
                                                Some(ref mut cb) => {
                                                    if !cb.found(bytes) {
                                                        done = true;
                                                    }
                                                }
                                                _ => {}
                                            }
                                        }
                                    }
                                }
                            }
                        } else {
                            for i in 0..batch_size {
                                for j in 0..size {
                                    let index = i * size + j;
                                    if job.counts[index] != 0 {
                                        let bytes = &job.seeds[i * KEY_SIZE..(i + 1) * KEY_SIZE];

                                        match *cb.lock().unwrap() {
                                            Some(ref mut cb) => {
                                                if !cb.found(bytes) {
                                                    done = true;
                                                }
                                            }
                                            _ => {}
                                        }
                                    }
                                }
                            }
                        }

                        last_batch = batch_size;
                    }
                });

                let worker = Worker {
                    queue,
                    kernel,
                    counts_buffer,
                    seed_buffer,
                    addrs_buffer,
                    job_tx,
                    cpu: self.cpu,
                    t,
                };

                workers.push(worker);
            }

            workers_start_barrier.wait();

            Runner {
                size: self.prefix_count,
                batch_size: _batch_size,
                seeds_rx: Some(seeds_rx),

                _size_buffer: size_buffer,
                _prefix_length_buffer: prefix_length_buffer,
                _prefix_chunks_buffer: prefix_chunks_buffer,
                _msig_buffer: msig_buffer,

                status_rx: Some(status_rx),
                seed_threads,
                workers,
            }
        }
    }
}

pub fn max_batch_size(device: &opencl3::device::Device, preferred_multiple: usize) -> usize {
    align_to_preferred_multiple(
        device.max_mem_alloc_size().unwrap() as usize / 2048,
        preferred_multiple,
    )
}

pub fn align_to_preferred_multiple(value: usize, preferred_multiple: usize) -> usize {
    max(
        preferred_multiple,
        value / preferred_multiple * preferred_multiple,
    )
}
