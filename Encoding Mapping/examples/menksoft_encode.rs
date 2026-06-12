// nominal Unicode -> Menksoft PUA (shaping baked into code points).
// Usage: cargo run --example menksoft_encode -- <in.txt> <out.txt>
use encoding_mapping::convert_unicode_to_menksoft;
use std::{env, fs};
fn main() {
    let args: Vec<String> = env::args().collect();
    let input = fs::read_to_string(&args[1]).expect("read");
    fs::write(&args[2], convert_unicode_to_menksoft(&input)).expect("write");
}
