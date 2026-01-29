use core::arch::asm;

extern "C" {
    static _NEO_INPUT_ADDR: u8;
    static _NEO_OUTPUT_ADDR: u8;
    static _NEO_HEAP_START: u8;
    static _NEO_HEAP_END: u8;
}

#[inline]
pub fn input_addr() -> usize {
    (&raw const _NEO_INPUT_ADDR) as usize
}

#[inline]
pub fn output_addr() -> usize {
    (&raw const _NEO_OUTPUT_ADDR) as usize
}

#[inline]
pub fn heap_start() -> usize {
    (&raw const _NEO_HEAP_START) as usize
}

#[inline]
pub fn heap_end() -> usize {
    (&raw const _NEO_HEAP_END) as usize
}

#[inline]
pub fn read_u32(addr: usize) -> u32 {
    unsafe { core::ptr::read_volatile(addr as *const u32) }
}

#[inline]
pub fn write_u32(addr: usize, value: u32) {
    unsafe {
        core::ptr::write_volatile(addr as *mut u32, value);
    }
}

#[inline]
pub fn read_input_u32() -> u32 {
    read_u32(input_addr())
}

#[inline]
pub fn commit_u32(value: u32) {
    write_u32(output_addr(), value);
}

#[inline]
pub fn halt() -> ! {
    unsafe {
        asm!("li a0, 0", "ecall", options(nostack));
    }
    loop {}
}
