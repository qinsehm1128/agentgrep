//! Test-only helpers shared by unit tests.

use std::path::Path;

/// Whether the filesystem under `dir` accepts non-UTF-8 file names.
///
/// APFS (macOS) rejects invalid UTF-8 names with `EILSEQ`, so tests that
/// build non-UTF-8 fixtures must skip there instead of failing.
#[cfg(unix)]
pub(crate) fn non_utf8_names_supported(dir: &Path) -> bool {
    use std::ffi::OsStr;
    use std::os::unix::ffi::OsStrExt;
    let probe = dir.join(OsStr::from_bytes(b".agentgrep-probe-\xff"));
    match std::fs::write(&probe, b"") {
        Ok(()) => {
            let _ = std::fs::remove_file(&probe);
            true
        }
        Err(_) => false,
    }
}

/// Early-return from a test when the tempdir filesystem rejects non-UTF-8 names.
#[macro_export]
#[doc(hidden)]
macro_rules! skip_unless_non_utf8_fs {
    ($dir:expr) => {
        if !$crate::test_support::non_utf8_names_supported($dir) {
            eprintln!("skipped: filesystem rejects non-UTF-8 file names");
            return;
        }
    };
}
