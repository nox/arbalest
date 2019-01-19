extern crate arbalest;

use arbalest::Arbalest;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering::{Acquire, SeqCst};
use std::sync::mpsc;
use std::thread;

#[test]
fn manually_share_arbalest() {
    let v = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    let arbalest_v = Arbalest::new(v);

    let (tx, rx) = mpsc::channel();

    let _t = thread::spawn(move || {
        let arbalest_v: Arbalest<Vec<i32>> = rx.recv().unwrap();
        assert_eq!((*arbalest_v)[3], 4);
    });

    tx.send(arbalest_v.clone()).unwrap();

    assert_eq!((*arbalest_v)[2], 3);
    assert_eq!((*arbalest_v)[4], 5);
}

#[test]
fn try_unwrap() {
    let x = Arbalest::new(3);
    assert_eq!(Arbalest::try_unwrap(x), Ok(3));
    let x = Arbalest::new(4);
    let _y = x.clone();
    assert_eq!(Arbalest::try_unwrap(x), Err(Arbalest::new(4)));
    let x = Arbalest::new(5);
    let _w = Arbalest::downgrade(&x);
    assert_eq!(Arbalest::try_unwrap(x), Ok(5));
}

#[test]
fn into_from_raw() {
    let x = Arbalest::new(Box::new("hello"));
    let y = x.clone();

    let x_ptr = Arbalest::into_raw(x);
    drop(y);
    unsafe {
        assert_eq!(**x_ptr, "hello");

        let x = Arbalest::from_raw(x_ptr);
        assert_eq!(**x, "hello");

        assert_eq!(Arbalest::try_unwrap(x).map(|x| *x), Ok("hello"));
    }
}

#[test]
fn test_live() {
    let x = Arbalest::new(5);
    let y = Arbalest::downgrade(&x);
    assert!(y.upgrade().is_some());
}

#[test]
fn test_dead() {
    let x = Arbalest::new(5);
    let y = Arbalest::downgrade(&x);
    drop(x);
    assert!(y.upgrade().is_none());
}

#[test]
fn drop_arbalest() {
    let mut canary = AtomicUsize::new(0);
    let x = Arbalest::new(Canary(&mut canary));
    drop(x);
    assert!(canary.load(Acquire) == 1);
}

#[test]
fn drop_arbalest_fragile() {
    let mut canary = AtomicUsize::new(0);
    let arbalest = Arbalest::new(Canary(&mut canary));
    let arbalest_fragile = Arbalest::downgrade(&arbalest);
    assert!(canary.load(Acquire) == 0);
    drop(arbalest);
    assert!(canary.load(Acquire) == 1);
    drop(arbalest_fragile);
}

#[test]
fn test_strong_count() {
    let a = Arbalest::new(0);
    assert!(Arbalest::strong_count(&a) == 1);
    let w = Arbalest::downgrade(&a);
    assert!(Arbalest::strong_count(&a) == 1);
    let b = w.upgrade().expect("");
    assert!(Arbalest::strong_count(&b) == 2);
    assert!(Arbalest::strong_count(&a) == 2);
    drop(w);
    drop(a);
    assert!(Arbalest::strong_count(&b) == 1);
    let c = b.clone();
    assert!(Arbalest::strong_count(&b) == 2);
    assert!(Arbalest::strong_count(&c) == 2);
}

#[test]
fn test_fragile_count() {
    let a = Arbalest::new(0);
    assert!(Arbalest::strong_count(&a) == 1);
    assert!(Arbalest::fragile_count(&a) == 0);
    let w = Arbalest::downgrade(&a);
    assert!(Arbalest::strong_count(&a) == 1);
    assert!(Arbalest::fragile_count(&a) == 1);
    let x = w.clone();
    assert!(Arbalest::fragile_count(&a) == 2);
    drop(w);
    drop(x);
    assert!(Arbalest::strong_count(&a) == 1);
    assert!(Arbalest::fragile_count(&a) == 0);
    let c = a.clone();
    assert!(Arbalest::strong_count(&a) == 2);
    assert!(Arbalest::fragile_count(&a) == 0);
    let d = Arbalest::downgrade(&c);
    assert!(Arbalest::fragile_count(&c) == 1);
    assert!(Arbalest::strong_count(&c) == 2);

    drop(a);
    drop(c);
    drop(d);
}

#[test]
fn show_arbalest() {
    let a = Arbalest::new(5);
    assert_eq!(format!("{:?}", a), "5");
}

#[test]
fn test_from_owned() {
    let foo = 123;
    let foo_arbalest = Arbalest::from(foo);
    assert!(123 == *foo_arbalest);
}

// Make sure deriving works with Arbalest<T>.
#[derive(Eq, Ord, PartialEq, PartialOrd, Clone, Debug, Default)]
struct _Foo {
    _inner: Arbalest<i32>,
}

struct Canary(*mut AtomicUsize);

impl Drop for Canary {
    fn drop(&mut self) {
        unsafe {
            (*self.0).fetch_add(1, SeqCst);
        }
    }
}
