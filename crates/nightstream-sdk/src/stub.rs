pub const INPUT_ADDR: usize = 0x104;
pub const OUTPUT_ADDR: usize = 0x100;
pub const HEAP_START: usize = 0x200;
pub const HEAP_END: usize = 0x700;

#[inline]
pub fn input_addr() -> usize {
    INPUT_ADDR
}

#[inline]
pub fn output_addr() -> usize {
    OUTPUT_ADDR
}

#[inline]
pub fn heap_start() -> usize {
    HEAP_START
}

#[inline]
pub fn heap_end() -> usize {
    HEAP_END
}

#[inline]
pub fn read_u32(_addr: usize) -> u32 {
    #[cfg(feature = "std")]
    panic!("nightstream-sdk::read_u32 is only available on RISC-V targets");
    #[cfg(not(feature = "std"))]
    0
}

#[inline]
pub fn write_u32(_addr: usize, _value: u32) {
    #[cfg(feature = "std")]
    panic!("nightstream-sdk::write_u32 is only available on RISC-V targets");
}

#[inline]
pub fn read_input_u32() -> u32 {
    read_u32(input_addr())
}

#[inline]
pub fn commit_u32(_value: u32) {
    #[cfg(feature = "std")]
    panic!("nightstream-sdk::commit_u32 is only available on RISC-V targets");
}

#[inline]
pub fn halt() -> ! {
    #[cfg(feature = "std")]
    panic!("nightstream-sdk::halt is only available on RISC-V targets");
    #[cfg(not(feature = "std"))]
    loop {}
}
