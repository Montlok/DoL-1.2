// One-off: convert a Menksoft-PUA text file to nominal Unicode.
// Usage: cargo run --example pua_convert -- <in.txt> <out.txt>
use encoding_mapping::{convert_menksoft_to_unicode, normalize_to_nominal_unicode};
use std::{env, fs};

fn main() {
    let args: Vec<String> = env::args().collect();
    let input = fs::read_to_string(&args[1]).expect("read input");
    let unicode = convert_menksoft_to_unicode(&input);
    let nominal = normalize_to_nominal_unicode(&unicode);
    fs::write(&args[2], nominal).expect("write output");
    eprintln!("converted {} chars", input.chars().count());
}
