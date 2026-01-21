#![no_std]
#![no_main]

#[nightstream_sdk::provable]
fn u64_output() -> u64 {
    0x1122_3344_5566_7788u64
}
