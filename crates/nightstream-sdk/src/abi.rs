pub trait NeoAbi: Sized {
    const WORDS: usize;
    unsafe fn read_from_words(ptr: *const u32) -> Self;
    unsafe fn write_to_words(&self, ptr: *mut u32);
}

impl NeoAbi for () {
    const WORDS: usize = 0;

    #[inline]
    unsafe fn read_from_words(_ptr: *const u32) -> Self {
        ()
    }

    #[inline]
    unsafe fn write_to_words(&self, _ptr: *mut u32) {}
}

impl NeoAbi for u32 {
    const WORDS: usize = 1;

    #[inline]
    unsafe fn read_from_words(ptr: *const u32) -> Self {
        core::ptr::read_volatile(ptr)
    }

    #[inline]
    unsafe fn write_to_words(&self, ptr: *mut u32) {
        core::ptr::write_volatile(ptr, *self)
    }
}

impl NeoAbi for i32 {
    const WORDS: usize = 1;

    #[inline]
    unsafe fn read_from_words(ptr: *const u32) -> Self {
        core::ptr::read_volatile(ptr) as i32
    }

    #[inline]
    unsafe fn write_to_words(&self, ptr: *mut u32) {
        core::ptr::write_volatile(ptr, *self as u32)
    }
}

impl NeoAbi for bool {
    const WORDS: usize = 1;

    #[inline]
    unsafe fn read_from_words(ptr: *const u32) -> Self {
        core::ptr::read_volatile(ptr) != 0
    }

    #[inline]
    unsafe fn write_to_words(&self, ptr: *mut u32) {
        core::ptr::write_volatile(ptr, if *self { 1 } else { 0 })
    }
}

impl NeoAbi for u64 {
    const WORDS: usize = 2;

    #[inline]
    unsafe fn read_from_words(ptr: *const u32) -> Self {
        let lo = core::ptr::read_volatile(ptr) as u64;
        let hi = core::ptr::read_volatile(ptr.add(1)) as u64;
        lo | (hi << 32)
    }

    #[inline]
    unsafe fn write_to_words(&self, ptr: *mut u32) {
        core::ptr::write_volatile(ptr, *self as u32);
        core::ptr::write_volatile(ptr.add(1), (*self >> 32) as u32);
    }
}

impl NeoAbi for i64 {
    const WORDS: usize = 2;

    #[inline]
    unsafe fn read_from_words(ptr: *const u32) -> Self {
        u64::read_from_words(ptr) as i64
    }

    #[inline]
    unsafe fn write_to_words(&self, ptr: *mut u32) {
        (*self as u64).write_to_words(ptr)
    }
}

impl NeoAbi for u128 {
    const WORDS: usize = 4;

    #[inline]
    unsafe fn read_from_words(ptr: *const u32) -> Self {
        let w0 = core::ptr::read_volatile(ptr) as u128;
        let w1 = core::ptr::read_volatile(ptr.add(1)) as u128;
        let w2 = core::ptr::read_volatile(ptr.add(2)) as u128;
        let w3 = core::ptr::read_volatile(ptr.add(3)) as u128;
        w0 | (w1 << 32) | (w2 << 64) | (w3 << 96)
    }

    #[inline]
    unsafe fn write_to_words(&self, ptr: *mut u32) {
        core::ptr::write_volatile(ptr, *self as u32);
        core::ptr::write_volatile(ptr.add(1), (*self >> 32) as u32);
        core::ptr::write_volatile(ptr.add(2), (*self >> 64) as u32);
        core::ptr::write_volatile(ptr.add(3), (*self >> 96) as u32);
    }
}

impl NeoAbi for i128 {
    const WORDS: usize = 4;

    #[inline]
    unsafe fn read_from_words(ptr: *const u32) -> Self {
        u128::read_from_words(ptr) as i128
    }

    #[inline]
    unsafe fn write_to_words(&self, ptr: *mut u32) {
        (*self as u128).write_to_words(ptr)
    }
}
