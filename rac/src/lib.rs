use std::{
    ffi::CStr,
    os::raw::c_char,
    sync::{Arc, Mutex},
};

use agvg::{
    self,
    bacon::{
        Callback, Context, DEFAULT_KERNEL, Generator, Optimizer, Session,
        align_to_preferred_multiple, max_batch_size,
    },
};

pub struct Rac {}

pub struct RacSession {
    session: Session,
    found_key: Arc<Mutex<Option<Vec<u8>>>>,
}

#[unsafe(no_mangle)]
pub extern "C" fn rac_new() -> *mut Rac {
    let rac = Box::new(Rac {});
    Box::into_raw(rac)
}

#[unsafe(no_mangle)]
pub extern "C" fn rac_free(c_rac: *mut Rac) {
    if c_rac.is_null() {
        return;
    }

    unsafe {
        let _ = Box::from_raw(c_rac);
    }
}

struct RacGenerateCallback {
    found: Arc<Mutex<Option<Vec<u8>>>>,
}

impl Callback for RacGenerateCallback {
    fn found(&mut self, key: &[u8]) -> bool {
        let mut found = self.found.lock().unwrap();
        if found.is_none() {
            *found = Some(key.to_vec());
        }
        false
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn rac_session_new(
    c_rac: *mut Rac,
    c_prefix: *const c_char,
    batch_size: usize,
) -> *mut RacSession {
    if c_rac.is_null() || c_prefix.is_null() {
        return std::ptr::null_mut();
    }

    let _rac = unsafe { &*c_rac };
    let ctx = Context::new(false, None, 0, DEFAULT_KERNEL.to_string());

    let prefix = unsafe { CStr::from_ptr(c_prefix) }.to_str().unwrap();
    let prefixes = vec![prefix.to_string()];
    let init = ctx.prepare(&prefixes);

    let generator = Generator::new(init);

    let found_key = Arc::new(Mutex::new(None));
    let cb = Box::new(RacGenerateCallback {
        found: found_key.clone(),
    });

    let session = generator.start(batch_size, 0, 0, false, Some(cb), None);

    let rac_session = Box::new(RacSession { session, found_key });

    Box::into_raw(rac_session)
}

#[unsafe(no_mangle)]
pub extern "C" fn rac_session_result(c_session: *mut RacSession) -> *mut c_char {
    if !c_session.is_null() {
        let session = unsafe { &mut *c_session };

        if session.session.step() {
            if let Some(key) = &*session.found_key.lock().unwrap() {
                let c_string = unsafe { std::ffi::CString::from_vec_unchecked(key.clone()) };
                return c_string.into_raw();
            }
        }
    }

    std::ptr::null_mut()
}

#[unsafe(no_mangle)]
pub extern "C" fn rac_session_free(c_session: *mut RacSession) {
    if c_session.is_null() {
        return;
    }

    unsafe {
        let _ = Box::from_raw(c_session);
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn rac_optimize(c_rac: *mut Rac, c_prefix: *const c_char, time: usize) -> usize {
    if c_rac.is_null() || c_prefix.is_null() {
        return 0;
    }

    let _rac = unsafe { &*c_rac };
    let ctx = Context::new(false, None, 0, DEFAULT_KERNEL.to_string());

    let prefix = unsafe { CStr::from_ptr(c_prefix) }.to_str().unwrap();
    let prefixes = vec![prefix.to_string()];
    let init = ctx.prepare(&prefixes);

    let optimizer = Optimizer::new(init);

    let preferred_multiple = ctx.preferred_multiple();
    let from_batch_size = align_to_preferred_multiple(0, preferred_multiple);
    let to_batch_size = max_batch_size(&ctx.device(), preferred_multiple);

    let (batch, _) = optimizer.run(
        ctx.preferred_multiple(),
        from_batch_size,
        to_batch_size,
        0,
        0,
        false,
        0,
        0,
        0,
        time,
        None,
        None,
    );

    batch
}

#[unsafe(no_mangle)]
pub extern "C" fn rac_session_result_free(c_str: *mut c_char) {
    if c_str.is_null() {
        return;
    }

    unsafe {
        let _ = std::ffi::CString::from_raw(c_str);
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {}
}
