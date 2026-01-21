#![no_std]
#![no_main]

#[derive(nightstream_sdk::NeoAbi)]
struct FibInput {
    n: u32,
}

#[nightstream_sdk::provable]
fn fib(input: FibInput) -> u32 {
    let mut n = input.n;
    let mut a = 0u32;
    let mut b = 1u32;
    while n > 0 {
        let next = a.wrapping_add(b);
        a = b;
        b = next;
        n -= 1;
    }
    a
}
