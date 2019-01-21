extern crate arbalest;

use arbalest::sync::Strong;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering::{Acquire, SeqCst};
use std::sync::mpsc;
use std::thread;

#[test]
fn manually_share_strong() {
    let v = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    let arbalest_v = Strong::new(v);

    let (tx, rx) = mpsc::channel();

    let _t = thread::spawn(move || {
        let arbalest_v: Strong<Vec<i32>> = rx.recv().unwrap();
        assert_eq!((*arbalest_v)[3], 4);
    });

    tx.send(arbalest_v.clone()).unwrap();

    assert_eq!((*arbalest_v)[2], 3);
    assert_eq!((*arbalest_v)[4], 5);
}

#[test]
fn try_unwrap() {
    let x = Strong::new(3);
    assert_eq!(Strong::try_unwrap(x), Ok(3));
    let x = Strong::new(4);
    let _y = x.clone();
    assert_eq!(Strong::try_unwrap(x), Err(Strong::new(4)));
    let x = Strong::new(5);
    let _w = Strong::downgrade(&x);
    assert_eq!(Strong::try_unwrap(x), Ok(5));
}

#[test]
fn into_from_raw() {
    let x = Strong::new(Box::new("hello"));
    let y = x.clone();

    let x_ptr = Strong::into_raw(x);
    drop(y);
    unsafe {
        assert_eq!(**x_ptr, "hello");

        let x = Strong::from_raw(x_ptr);
        assert_eq!(**x, "hello");

        assert_eq!(Strong::try_unwrap(x).map(|x| *x), Ok("hello"));
    }
}

#[test]
fn test_live() {
    let x = Strong::new(5);
    let y = Strong::downgrade(&x);
    assert!(y.upgrade().is_some());
}

#[test]
fn test_dead() {
    let x = Strong::new(5);
    let y = Strong::downgrade(&x);
    drop(x);
    assert!(y.upgrade().is_none());
}

#[test]
fn drop_strong() {
    let mut canary = AtomicUsize::new(0);
    let x = Strong::new(Canary(&mut canary));
    drop(x);
    assert!(canary.load(Acquire) == 1);
}

#[test]
fn drop_strong_frail() {
    let mut canary = AtomicUsize::new(0);
    let arbalest = Strong::new(Canary(&mut canary));
    let arbalest_frail = Strong::downgrade(&arbalest);
    assert!(canary.load(Acquire) == 0);
    drop(arbalest);
    assert!(canary.load(Acquire) == 1);
    drop(arbalest_frail);
}

#[test]
fn test_strong_count() {
    let a = Strong::new(0);
    assert!(Strong::strong_count(&a) == 1);
    let w = Strong::downgrade(&a);
    assert!(Strong::strong_count(&a) == 1);
    let b = w.upgrade().expect("");
    assert!(Strong::strong_count(&b) == 2);
    assert!(Strong::strong_count(&a) == 2);
    drop(w);
    drop(a);
    assert!(Strong::strong_count(&b) == 1);
    let c = b.clone();
    assert!(Strong::strong_count(&b) == 2);
    assert!(Strong::strong_count(&c) == 2);
}

#[test]
fn test_frail_count() {
    let a = Strong::new(0);
    assert!(Strong::strong_count(&a) == 1);
    assert!(Strong::frail_count(&a) == 0);
    let w = Strong::downgrade(&a);
    assert!(Strong::strong_count(&a) == 1);
    assert!(Strong::frail_count(&a) == 1);
    let x = w.clone();
    assert!(Strong::frail_count(&a) == 2);
    drop(w);
    drop(x);
    assert!(Strong::strong_count(&a) == 1);
    assert!(Strong::frail_count(&a) == 0);
    let c = a.clone();
    assert!(Strong::strong_count(&a) == 2);
    assert!(Strong::frail_count(&a) == 0);
    let d = Strong::downgrade(&c);
    assert!(Strong::frail_count(&c) == 1);
    assert!(Strong::strong_count(&c) == 2);

    drop(a);
    drop(c);
    drop(d);
}

#[test]
fn show_strong() {
    let a = Strong::new(5);
    assert_eq!(format!("{:?}", a), "5");
}

#[test]
fn test_from_owned() {
    let foo = 123;
    let foo_strong = Strong::from(foo);
    assert!(123 == *foo_strong);
}

// Make sure deriving works with Strong<T>.
#[derive(Eq, Ord, PartialEq, PartialOrd, Clone, Debug, Default)]
struct _Foo {
    _inner: Strong<i32>,
}

struct Canary(*mut AtomicUsize);

impl Drop for Canary {
    fn drop(&mut self) {
        unsafe {
            (*self.0).fetch_add(1, SeqCst);
        }
    }
}
