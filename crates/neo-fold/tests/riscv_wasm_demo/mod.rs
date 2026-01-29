pub mod mini_asm;

pub fn fib_u32(n: u32) -> u32 {
    let mut n = n;
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

pub fn fmt_bytes(bytes: u64) -> String {
    const KB: f64 = 1024.0;
    const MB: f64 = 1024.0 * 1024.0;
    const GB: f64 = 1024.0 * 1024.0 * 1024.0;
    let b = bytes as f64;
    if b < KB {
        return format!("{bytes} B");
    }
    if b < MB {
        return format!("{:.2} KiB", b / KB);
    }
    if b < GB {
        return format!("{:.2} MiB", b / MB);
    }
    format!("{:.2} GiB", b / GB)
}

#[cfg(unix)]
pub fn max_rss_bytes() -> Option<u64> {
    use std::mem::MaybeUninit;

    let mut usage = MaybeUninit::<libc::rusage>::uninit();
    let rc = unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) };
    if rc != 0 {
        return None;
    }
    let usage = unsafe { usage.assume_init() };
    let rss = u64::try_from(usage.ru_maxrss).ok()?;

    // Linux reports KiB; macOS reports bytes.
    #[cfg(target_os = "macos")]
    {
        Some(rss)
    }
    #[cfg(not(target_os = "macos"))]
    {
        Some(rss.saturating_mul(1024))
    }
}

#[cfg(not(unix))]
pub fn max_rss_bytes() -> Option<u64> {
    None
}
