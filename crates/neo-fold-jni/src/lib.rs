use jni::objects::{JClass, JString};
use jni::sys::{jboolean, jstring};
use jni::JNIEnv;

fn throw_runtime_exception(env: &mut JNIEnv<'_>, message: &str) {
    let _ = env.throw_new("java/lang/RuntimeException", message);
}

fn jstring_from_utf8(env: &mut JNIEnv<'_>, s: &str) -> jstring {
    match env.new_string(s) {
        Ok(s) => s.into_raw(),
        Err(_) => core::ptr::null_mut(),
    }
}

#[no_mangle]
pub extern "system" fn Java_com_midnight_neofold_demo_NeoFoldNative_runWasmDemoWorkflowJson(
    mut env: JNIEnv<'_>,
    _class: JClass<'_>,
    json: JString<'_>,
    do_spartan: jboolean,
) -> jstring {
    let json: String = match env.get_string(&json) {
        Ok(s) => s.into(),
        Err(e) => {
            throw_runtime_exception(&mut env, &format!("invalid input string: {e}"));
            return core::ptr::null_mut();
        }
    };

    let mut out_ptr: *mut u8 = core::ptr::null_mut();
    let mut out_len: usize = 0;
    let mut err_ptr: *mut u8 = core::ptr::null_mut();
    let mut err_len: usize = 0;

    let do_spartan_flag: i32 = if do_spartan != 0 { 1 } else { 0 };

    let rc = neo_fold_ffi::neo_fold_run_wasm_demo_workflow_json(
        json.as_ptr(),
        json.len(),
        do_spartan_flag,
        &mut out_ptr,
        &mut out_len,
        &mut err_ptr,
        &mut err_len,
    );

    let out: &[u8] = unsafe {
        if out_ptr.is_null() || out_len == 0 {
            &[]
        } else {
            core::slice::from_raw_parts(out_ptr, out_len)
        }
    };
    let err: &[u8] = unsafe {
        if err_ptr.is_null() || err_len == 0 {
            &[]
        } else {
            core::slice::from_raw_parts(err_ptr, err_len)
        }
    };

    let out_string = if rc == 0 {
        match core::str::from_utf8(out) {
            Ok(s) => Some(s.to_string()),
            Err(_) => {
                throw_runtime_exception(&mut env, "neo-fold returned invalid UTF-8 output");
                None
            }
        }
    } else {
        let msg = core::str::from_utf8(err).unwrap_or("<non-utf8 error>");
        throw_runtime_exception(&mut env, &format!("neo-fold error (code {rc}): {msg}"));
        None
    };

    if !out_ptr.is_null() {
        neo_fold_ffi::neo_fold_free_bytes(out_ptr, out_len);
    }
    if !err_ptr.is_null() {
        neo_fold_ffi::neo_fold_free_bytes(err_ptr, err_len);
    }

    match out_string {
        Some(s) => jstring_from_utf8(&mut env, &s),
        None => core::ptr::null_mut(),
    }
}
