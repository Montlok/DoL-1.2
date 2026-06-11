//! Stream-normalize traditional Mongolian text from stdin to stdout.
//!
//! Usage:
//!   cargo run --example normalize_stream -- [--nominal] [--drop-pua] < input.jsonl > output.jsonl

use std::io::{self, BufRead, Write};

use encoding_mapping::{normalize_to_nominal_unicode, normalize_to_unicode};

fn main() {
    let mut nominal = false;
    let mut drop_pua = false;
    for arg in std::env::args().skip(1) {
        match arg.as_str() {
            "--nominal" => nominal = true,
            "--drop-pua" => drop_pua = true,
            "--help" | "-h" => {
                eprintln!(
                    "Usage: normalize_stream [--nominal] [--drop-pua] < input > output\n\
                     Reads UTF-8 text line by line and writes normalized text."
                );
                return;
            }
            other => {
                eprintln!("normalize_stream: unknown argument {other:?}");
                std::process::exit(2);
            }
        }
    }

    let stdin = io::stdin();
    let mut stdout = io::BufWriter::new(io::stdout().lock());

    for line in stdin.lock().lines() {
        let line = match line {
            Ok(line) => line,
            Err(e) => {
                eprintln!("normalize_stream: read error: {e}");
                std::process::exit(1);
            }
        };
        let out = if nominal {
            normalize_to_nominal_unicode(&line)
        } else {
            normalize_to_unicode(&line)
        };
        let out = if drop_pua {
            drop_private_use(&out)
        } else {
            out
        };
        if let Err(e) = writeln!(stdout, "{out}") {
            eprintln!("normalize_stream: write error: {e}");
            std::process::exit(1);
        }
    }
}

fn drop_private_use(input: &str) -> String {
    input
        .chars()
        .map(|c| {
            let cp = c as u32;
            if (0xE000..=0xF8FF).contains(&cp) {
                ' '
            } else {
                c
            }
        })
        .collect()
}
