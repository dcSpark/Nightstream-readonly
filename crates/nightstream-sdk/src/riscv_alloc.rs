use core::alloc::{GlobalAlloc, Layout};

pub struct BumpAllocator;

impl BumpAllocator {
    #[inline]
    fn ptr_addr() -> usize {
        crate::heap_start()
    }

    #[inline]
    fn heap_data_start() -> usize {
        crate::heap_start().saturating_add(4)
    }

    #[inline]
    fn heap_end() -> usize {
        crate::heap_end()
    }

    #[inline]
    fn align_up(addr: usize, align: usize) -> usize {
        debug_assert!(align.is_power_of_two());
        (addr + align - 1) & !(align - 1)
    }
}

unsafe impl GlobalAlloc for BumpAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if layout.size() == 0 {
            return layout.align() as *mut u8;
        }

        let ptr_addr = Self::ptr_addr();
        let heap_end = Self::heap_end();
        let mut cur = crate::read_u32(ptr_addr) as usize;
        if cur == 0 {
            cur = Self::heap_data_start();
        }

        let aligned = Self::align_up(cur, layout.align().max(4));
        let next = match aligned.checked_add(layout.size()) {
            Some(v) => v,
            None => return core::ptr::null_mut(),
        };
        if next > heap_end {
            return core::ptr::null_mut();
        }

        crate::write_u32(ptr_addr, next as u32);
        aligned as *mut u8
    }

    unsafe fn dealloc(&self, _ptr: *mut u8, _layout: Layout) {}
}

