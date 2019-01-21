// Copyright 2012-2018 The Rust Project Developers
// Copyright 2018 Anthony Ramine
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Thread-safe reference-counting pointers.
//!
//! See the [`Strong<T>`][strong] documentation for more details.
//!
//! “Arbalest” is just a cute name, an `Arc<T>` with a twist, and “arc” is
//! French for “bow”.
//!
//! [strong]: struct.Strong.html

use std::alloc::{self, Layout};
use std::borrow::Borrow;
use std::cell::UnsafeCell;
use std::cmp::Ordering;
use std::error::Error;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::isize;
use std::marker::PhantomData;
use std::mem;
use std::ops::{Deref, DerefMut};
use std::panic::{RefUnwindSafe, UnwindSafe};
use std::process;
use std::ptr::{self, NonNull};
use std::sync::atomic::Ordering::{Acquire, Relaxed, Release, SeqCst};
use std::sync::atomic::{self, AtomicUsize};
use std::usize;

/// A thread-safe reference-counting pointer.
///
/// The type `Strong<T>` provides shared ownership of a value of type `T`,
/// allocated in the heap. It behaves mostly like `Arc<T>`, except that it
/// provides a way to mutably borrow the `T` that doesn't take into account
/// any frail references to it. Instead, frail references fail to upgrade
/// when the `Strong<T>` is mutably borrowed.
///
/// ## Thread Safety
///
/// `Strong<T>` will implement `Send` and `Sync` as long as the `T` implements
/// `Send` and `Sync`, just like `Arc<T>`.
///
/// ## Breaking cycles with `Frail`
///
/// The [`downgrade`][downgrade] method can be used to create a non-owning
/// [`Frail`][frail] pointer. A [`Frail`][frail] pointer can be
/// [`upgrade`][upgrade]d to an `Strong`, but this will return `None` if the
/// value has already been dropped, or panic if the `T` is mutably borrowed
/// by another `Strong`.
///
/// A cycle between `Strong` pointers will never be deallocated. For this
/// reason, [`Frail`][frail] is used to break cycles. For example, a tree
/// could have strong `Strong` pointers from parent nodes to children, and
/// [`Frail`][frail] pointers from children back to their parents.
///
/// # Cloning references
///
/// Creating a new reference from an existing reference-counted pointer is done
/// using the `Clone` trait implemented for `Strong<T>` and
/// [`Frail<T>`][frail].
///
/// ```
/// use arbalest::Strong;
/// let foo = Strong::new(vec![1.0, 2.0, 3.0]);
/// // The two syntaxes below are equivalent.
/// let a = foo.clone();
/// let b = Strong::clone(&foo);
/// // a, b, and foo are all Arcs that point to the same memory location.
/// ```
///
/// The [`Strong::clone(&from)`] syntax is the most idiomatic because it
/// conveys more explicitly the meaning of the code. In the example above, this
/// syntax makes it easier to see that this code is creating a new reference
/// rather than copying the whole content of `foo`.
///
/// ## `Deref` behavior
///
/// `Strong<T>` automatically dereferences to `T` (via the `Deref` trait),
/// so you can call `T`'s methods on a value of type `Strong<T>`. To avoid
/// name clashes with `T`'s methods, the methods of `Strong<T>` itself are
/// associated functions, called using function-like syntax:
///
/// ```
/// use arbalest::Strong;
/// let my_Arbalest = Strong::new(());
///
/// Strong::downgrade(&my_Arbalest);
/// ```
///
/// [`Frail<T>`][frail] does not auto-dereference to `T`, because the value
/// may currently be mutably borrowed or have already been destroyed.
///
/// [frail]: struct.Frail.html
/// [downgrade]: #method.downgrade
/// [upgrade]: struct.Frail.html#method.upgrade
/// [`Strong::clone(&from)`]: #method.clone
pub struct Strong<T: ?Sized> {
    phantom: PhantomData<T>,
    ptr: NonNull<ArbalestInner<T>>,
}

unsafe impl<T: ?Sized + Sync + Send> Send for Strong<T> {}
unsafe impl<T: ?Sized + Sync + Send> Sync for Strong<T> {}

/// A mutable memory location with dynamically checked borrow rules.
pub struct RefMut<'b, T: ?Sized + 'b> {
    value: &'b mut T,
}

/// An error returned by [`Strong::try_borrow_mut`](struct.Strong.html#method.try_borrow_mut).
pub struct BorrowMutError {
    _private: (),
}

/// An error returned by [`Frail::try_upgrade`](struct.Frail.html#method.try_upgrade).
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum UpgradeError {
    /// The value has been dropped.
    Dropped,
    /// The value is currently mutably borrowed.
    MutablyBorrowed,
}

/// `Frail` is a version of [`Strong`] that holds a non-owning reference
/// to the managed value.
///
/// The value is accessed by calling [`upgrade`] on the `Frail`
/// pointer, which returns an `Option<`[`Strong`]`<T>>`.
///
/// Since a `Frail` reference does not count towards ownership, it will not
/// prevent the inner value from being dropped, and `Frail` itself makes no
/// guarantees about the value still being present and may return `None`
/// when [`upgrade`]d.
///
/// A `Frail` pointer is useful for keeping a temporary reference to the value
/// within [`Strong`] without extending its lifetime. It is also used to
/// prevent circular references between [`Strong`] pointers, since mutual
/// owning references would never allow either [`Strong`] to be dropped.
/// For example, a tree could have strong [`Strong`] pointers from parent
/// nodes to children, and `Frail` pointers from children back to their
/// parents.
///
/// The typical way to obtain a `Frail` pointer is to call
/// [`Strong::downgrade`].
///
/// [`Strong`]: struct.Strong.html
/// [`Strong::downgrade`]: struct.Strong.html#method.downgrade
/// [`upgrade`]: #method.upgrade
pub struct Frail<T: ?Sized> {
    // This is a `NonNull` to allow optimizing the size of this type in enums,
    // but it is not necessarily a valid pointer.
    // `Frail::new` sets this to `usize::MAX` so that it doesn’t need
    // to allocate space on the heap.  That's not a value a real pointer
    // will ever have because RcBox has alignment at least 2.
    ptr: NonNull<ArbalestInner<T>>,
}

unsafe impl<T: ?Sized + Sync + Send> Send for Frail<T> {}
unsafe impl<T: ?Sized + Sync + Send> Sync for Frail<T> {}

struct ArbalestInner<T: ?Sized> {
    strong: AtomicUsize,
    frail: AtomicUsize,
    data: UnsafeCell<T>,
}

unsafe impl<T: ?Sized + Sync + Send> Send for ArbalestInner<T> {}
unsafe impl<T: ?Sized + Sync + Send> Sync for ArbalestInner<T> {}

impl<T> Strong<T> {
    /// Constructs a new `Strong<T>`.
    ///
    /// # Examples
    ///
    /// ```
    /// use arbalest::Strong;
    ///
    /// let five = Strong::new(5);
    /// ```
    #[inline]
    pub fn new(data: T) -> Self {
        // Start the frail pointer count as 1 which is the frail pointer
        // that is held by all the strong pointers (kinda).
        let inner = Box::new(ArbalestInner {
            strong: AtomicUsize::new(1),
            frail: AtomicUsize::new(1),
            data: UnsafeCell::new(data),
        });
        Self {
            ptr: NonNull::new(Box::into_raw(inner)).unwrap(),
            phantom: PhantomData,
        }
    }

    /// Returns the contained value, if the `Strong` has exactly one strong
    /// reference.
    ///
    /// Otherwise, an error is returned with the same value that was passed in.
    ///
    /// This will succeed even if there are outstanding frail references.
    ///
    /// # Examples
    ///
    /// ```
    /// use arbalest::Strong;
    ///
    /// let x = Strong::new(3);
    /// assert_eq!(Strong::try_unwrap(x), Ok(3));
    ///
    /// let x = Strong::new(4);
    /// let y = Strong::clone(&x);
    /// assert_eq!(*Strong::try_unwrap(x).unwrap_err(), 4);
    /// ```
    pub fn try_unwrap(this: Self) -> Result<T, Self> {
        // See `drop` for why all these atomics are like this.
        if this.inner().strong.compare_and_swap(1, 0, Release) != 1 {
            return Err(this);
        }

        atomic::fence(Acquire);

        unsafe {
            let elem = ptr::read(&this.ptr.as_ref().data);

            // Make a frail pointer to clean up the implicit strong-frail reference.
            let _frail = Frail { ptr: this.ptr };
            mem::forget(this);

            Ok(elem.into_inner())
        }
    }
}

impl<T: ?Sized> Strong<T> {
    /// Mutably borrows the wrapped value.
    ///
    /// The borrow lasts until the returned `RefMut` exits scope. Frail
    /// references to that `Strong` cannot be upgraded while this borrow
    /// is active.
    ///
    /// # Panics
    ///
    /// Panics if the value is currently shared. For a non-panicking variant,
    /// use [`try_borrow_mut`](#method.try_borrow_mut).
    ///
    /// # Examples
    ///
    /// ```
    /// use arbalest::Strong;
    ///
    /// let mut c = Strong::new(5);
    ///
    /// *Strong::borrow_mut(&mut c) = 7;
    ///
    /// assert_eq!(*c, 7);
    /// ```
    ///
    /// An example of panic:
    ///
    /// ```
    /// use arbalest::Strong;
    /// use std::thread;
    ///
    /// let mut five = Strong::new(5);
    /// let same_five = Strong::clone(&five);
    /// let result = thread::spawn(move || {
    ///    let b = Strong::borrow_mut(&mut five); // this causes a panic
    /// }).join();
    ///
    /// assert!(result.is_err());
    /// ```
    #[inline]
    pub fn borrow_mut(this: &mut Self) -> RefMut<T> {
        match Self::try_borrow_mut(this) {
            Ok(value) => value,
            Err(_) => panic!("value is shared"),
        }
    }

    /// Mutably borrows the wrapped value, returning an error if the value
    /// is currently borrowed.
    ///
    /// The borrow lasts until the returned `RefMut` exits scope. Frail
    /// references to that `Strong` cannot be upgraded while this borrow
    /// is active.
    ///
    /// This is the non-panicking variant of [`borrow_mut`](#method.borrow_mut).
    ///
    /// # Examples
    ///
    /// ```
    /// use arbalest::Strong;
    ///
    /// let mut five = Strong::new(5);
    ///
    /// {
    ///     let same_five = Strong::clone(&five);
    ///     assert!(Strong::try_borrow_mut(&mut five).is_err());
    /// }
    ///
    /// assert!(Strong::try_borrow_mut(&mut five).is_ok());
    /// ```
    #[inline]
    pub fn try_borrow_mut(this: &mut Self) -> Result<RefMut<T>, BorrowMutError> {
        let inner = this.inner();
        // If this is true, that means there was only one Arbalete<T> for this
        // value, and given we have a &mut of it, there is no way another thread
        // is holding a reference to it.
        //
        // This Acquire load synchronises with the Release write in Strong::drop.
        if inner.strong.compare_and_swap(1, MUTABLE_REFCOUNT, Acquire) != 1 {
            return Err(BorrowMutError { _private: () });
        }
        Ok(RefMut {
            value: unsafe { &mut *inner.data.get() },
        })
    }

    /// Consumes the `Strong`, returning the wrapped pointer.
    ///
    /// To avoid a memory leak the pointer must be converted back to an
    /// `Strong` using [`Strong::from_raw`][from_raw].
    ///
    /// [from_raw]: #method.from_raw
    ///
    /// # Examples
    ///
    /// ```
    /// use arbalest::Strong;
    ///
    /// let x = Strong::new(10);
    /// let x_ptr = Strong::into_raw(x);
    /// assert_eq!(unsafe { *x_ptr }, 10);
    /// ```
    pub fn into_raw(this: Self) -> *const T {
        let ptr: *const T = &*this;
        mem::forget(this);
        ptr
    }

    /// Constructs an `Strong` from a raw pointer.
    ///
    /// The raw pointer must have been previously returned by a call to a
    /// [`Strong::into_raw`][into_raw].
    ///
    /// This function is unsafe because improper use may lead to memory problems.
    /// For example, a double-free may occur if the function is called twice on
    /// the same raw pointer.
    ///
    /// [into_raw]: #method.into_raw
    ///
    /// # Examples
    ///
    /// ```
    /// use arbalest::Strong;
    ///
    /// let x = Strong::new(10);
    /// let x_ptr = Strong::into_raw(x);
    ///
    /// unsafe {
    ///     // Convert back to an `Arc` to prevent leak.
    ///     let x = Strong::from_raw(x_ptr);
    ///     assert_eq!(*x, 10);
    ///
    ///     // Further calls to `Strong::from_raw(x_ptr)` would be memory-unsafe.
    /// }
    ///
    /// // The memory was freed when `x` went out of scope above, so `x_ptr` is now dangling!
    /// ```
    pub unsafe fn from_raw(ptr: *const T) -> Self {
        Self {
            ptr: NonNull::new_unchecked(ArbalestInner::from_raw(ptr)),
            phantom: PhantomData,
        }
    }

    /// Creates a new [`Frail`][frail] pointer to this value.
    ///
    /// [frail]: struct.Frail.html
    ///
    /// # Examples
    ///
    /// ```
    /// use arbalest::Strong;
    ///
    /// let five = Strong::new(5);
    ///
    /// let frail_five = Strong::downgrade(&five);
    /// ```
    #[inline]
    pub fn downgrade(this: &Self) -> Frail<T> {
        this.inner().downgrade()
    }

    /// Gets the number of [`Frail`][frail] pointers to this value.
    ///
    /// [frail]: struct.Frail.html
    ///
    /// # Safety
    ///
    /// This method by itself is safe, but using it correctly requires extra
    /// care. Another thread can change the frail count at any time, including
    /// potentially between calling this method and acting on
    /// the result.
    ///
    /// # Examples
    ///
    /// ```
    /// use arbalest::Strong;
    ///
    /// let five = Strong::new(5);
    /// let frail_five = Strong::downgrade(&five);
    ///
    /// // This assertion is deterministic because we haven't shared
    /// // the `Strong` or `Frail` between threads.
    /// assert_eq!(1, Strong::frail_count(&five));
    /// ```
    #[inline]
    pub fn frail_count(this: &Self) -> usize {
        this.inner().frail.load(SeqCst) - 1
    }

    /// Gets the number of strong (`Strong`) pointers to this value.
    ///
    /// # Safety
    ///
    /// This method by itself is safe, but using it correctly requires extra
    /// care. Another thread can change the strong count at any time, including
    /// potentially between calling this method and acting on the result.
    ///
    /// # Examples
    ///
    /// ```
    /// use arbalest::Strong;
    ///
    /// let five = Strong::new(5);
    /// let also_five = Strong::clone(&five);
    ///
    /// // This assertion is deterministic because we haven't shared
    /// // the `Strong` between threads.
    /// assert_eq!(2, Strong::strong_count(&five));
    /// ```
    #[inline]
    pub fn strong_count(this: &Self) -> usize {
        this.inner().strong.load(SeqCst)
    }

    /// Returns true if the two `Strong`s point to the same value (not
    /// just values that compare as equal).
    ///
    /// # Examples
    ///
    /// ```
    /// use arbalest::Strong;
    ///
    /// let five = Strong::new(5);
    /// let same_five = Strong::clone(&five);
    /// let other_five = Strong::new(5);
    ///
    /// assert!(Strong::ptr_eq(&five, &same_five));
    /// assert!(!Strong::ptr_eq(&five, &other_five));
    /// ```
    pub fn ptr_eq(this: &Self, other: &Self) -> bool {
        this.ptr.as_ptr() == other.ptr.as_ptr()
    }
}

impl<T: ?Sized> AsRef<T> for Strong<T> {
    #[inline]
    fn as_ref(&self) -> &T {
        self
    }
}

impl<T: ?Sized> Borrow<T> for Strong<T> {
    #[inline]
    fn borrow(&self) -> &T {
        self
    }
}

impl<T: ?Sized> Deref for Strong<T> {
    type Target = T;

    #[inline]
    fn deref(&self) -> &Self::Target {
        // This does not require any synchronisation, given that the value
        // can only be mutated when there is no other Strong pointing to it,
        // so if we deref this one, it's either the same one or a clone
        // on the same thread (in which case things are obviously in order),
        // or an access through a different thread, to which the data had to be
        // sent there in the first place, which required synchronisation on its
        // own.
        unsafe { &*self.inner().data.get() }
    }
}

impl<T: Default> Default for Strong<T> {
    /// Creates a new `Strong<T>`, with the `Default` value for `T`.
    ///
    /// # Examples
    ///
    /// ```
    /// use arbalest::Strong;
    ///
    /// let x: Strong<i32> = Default::default();
    /// assert_eq!(*x, 0);
    /// ```
    #[inline]
    fn default() -> Self {
        Self::new(Default::default())
    }
}

impl<T: ?Sized> Clone for Strong<T> {
    /// Makes a clone of the `Strong` pointer.
    ///
    /// This creates another pointer to the same inner value, increasing the
    /// strong reference count.
    ///
    /// # Examples
    ///
    /// ```
    /// use arbalest::Strong;
    ///
    /// let five = Strong::new(5);
    ///
    /// let _ = Strong::clone(&five);
    /// ```
    #[inline]
    fn clone(&self) -> Self {
        // FIXME(nox): If this is called after a RefMut is forgotten,
        // this will currently abort the program. Is there any way to recover
        // decently from a forgotten RefMut, taking advantage of the fact
        // that at the moment the RefMut was forgotten, there couldn't be more
        // than a single Strong<T>?

        // Using a relaxed ordering is alright here, as knowledge of the
        // original reference prevents other threads from erroneously deleting
        // the object.
        //
        // As explained in the [Boost documentation][1], Increasing the
        // reference counter can always be done with memory_order_relaxed: New
        // references to an object can only be formed from an existing
        // reference, and passing an existing reference from one thread to
        // another must already provide any required synchronization.
        //
        // [1]: (www.boost.org/doc/libs/1_55_0/doc/html/atomic/usage_examples.html)
        let old_strong_refcount = self.inner().strong.fetch_add(1, Relaxed);

        // However we need to guard against massive refcounts in case someone
        // is `mem::forget`ing Arbaletes. If we don't do this the count can
        // overflow and users will use-after free. We racily saturate to
        // `isize::MAX` on the assumption that there aren't ~2 billion threads
        // incrementing the reference count at once. This branch will never be
        // taken in any realistic program.
        //
        // We abort because such a program is incredibly degenerate, and we
        // don't care to support it.
        if old_strong_refcount > MAX_REFCOUNT {
            process::abort();
        }

        Self {
            ptr: self.ptr,
            phantom: PhantomData,
        }
    }
}

impl<T> From<T> for Strong<T> {
    #[inline]
    fn from(t: T) -> Self {
        Self::new(t)
    }
}

impl<T: ?Sized + Hash> Hash for Strong<T> {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        (**self).hash(state)
    }
}

impl<T: ?Sized + PartialEq> PartialEq for Strong<T> {
    /// Equality for two `Strong`s.
    ///
    /// Two `Strong`s are equal if their inner values are equal.
    ///
    /// # Examples
    ///
    /// ```
    /// use arbalest::Strong;
    ///
    /// let five = Strong::new(5);
    ///
    /// assert!(five == Strong::new(5));
    /// ```
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        **self == **other
    }

    /// Inequality for two `Strong`s.
    ///
    /// Two `Strong`s are unequal if their inner values are unequal.
    ///
    /// # Examples
    ///
    /// ```
    /// use arbalest::Strong;
    ///
    /// let five = Strong::new(5);
    ///
    /// assert!(five != Strong::new(6));
    /// ```
    #[inline]
    fn ne(&self, other: &Self) -> bool {
        **self != **other
    }
}

impl<T: ?Sized + Eq> Eq for Strong<T> {}

impl<T: ?Sized + PartialOrd> PartialOrd for Strong<T> {
    /// Partial comparison for two `Strong`s.
    ///
    /// The two are compared by calling `partial_cmp()` on their inner values.
    ///
    /// # Examples
    ///
    /// ```
    /// use arbalest::Strong;
    /// use std::cmp::Ordering;
    ///
    /// let five = Strong::new(5);
    ///
    /// assert_eq!(Some(Ordering::Less), five.partial_cmp(&Strong::new(6)));
    /// ```
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        (**self).partial_cmp(&**other)
    }

    /// Less-than comparison for two `Strong`s.
    ///
    /// The two are compared by calling `<` on their inner values.
    ///
    /// # Examples
    ///
    /// ```
    /// use arbalest::Strong;
    ///
    /// let five = Strong::new(5);
    ///
    /// assert!(five < Strong::new(6));
    /// ```
    fn lt(&self, other: &Self) -> bool {
        **self < **other
    }

    /// “Less than or equal to” comparison for two `Strong`s.
    ///
    /// The two are compared by calling `<=` on their inner values.
    ///
    /// # Examples
    ///
    /// ```
    /// use arbalest::Strong;
    ///
    /// let five = Strong::new(5);
    ///
    /// assert!(five <= Strong::new(5));
    /// ```
    fn le(&self, other: &Self) -> bool {
        **self <= **other
    }

    /// Greater-than comparison for two `Strong`s.
    ///
    /// The two are compared by calling `>` on their inner values.
    ///
    /// # Examples
    ///
    /// ```
    /// use arbalest::Strong;
    ///
    /// let five = Strong::new(5);
    ///
    /// assert!(five > Strong::new(4));
    /// ```
    fn gt(&self, other: &Self) -> bool {
        **self > **other
    }

    /// “Greater than or equal to” comparison for two `Strong`s.
    ///
    /// The two are compared by calling `>=` on their inner values.
    ///
    /// # Examples
    ///
    /// ```
    /// use arbalest::Strong;
    ///
    /// let five = Strong::new(5);
    ///
    /// assert!(five >= Strong::new(5));
    /// ```
    fn ge(&self, other: &Self) -> bool {
        **self >= **other
    }
}

impl<T: ?Sized + Ord> Ord for Strong<T> {
    /// Comparison for two `Strong`s.
    ///
    /// The two are compared by calling `cmp()` on their inner values.
    ///
    /// # Examples
    ///
    /// ```
    /// use arbalest::Strong;
    /// use std::cmp::Ordering;
    ///
    /// let five = Strong::new(5);
    ///
    /// assert_eq!(Ordering::Less, five.cmp(&Strong::new(6)));
    /// ```
    fn cmp(&self, other: &Self) -> Ordering {
        (**self).cmp(&**other)
    }
}

impl<T: ?Sized + fmt::Debug> fmt::Debug for Strong<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Debug::fmt(&**self, f)
    }
}

impl<T: ?Sized + fmt::Display> fmt::Display for Strong<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(&**self, f)
    }
}

impl<T: ?Sized> fmt::Pointer for Strong<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        (&**self as *const T).fmt(f)
    }
}

impl<T: RefUnwindSafe + ?Sized> UnwindSafe for Strong<T> {}

impl<T: ?Sized> Drop for Strong<T> {
    /// Drops the `Strong`.
    ///
    /// This will decrement the strong reference count. If the strong reference
    /// count reaches zero then the only other references (if any) are
    /// [`Frail`], so we `drop` the inner value.
    ///
    /// # Examples
    ///
    /// ```
    /// use arbalest::Strong;
    ///
    /// struct Foo;
    ///
    /// impl Drop for Foo {
    ///     fn drop(&mut self) {
    ///         println!("dropped!");
    ///     }
    /// }
    ///
    /// let foo  = Strong::new(Foo);
    /// let foo2 = Strong::clone(&foo);
    ///
    /// drop(foo);    // Doesn't print anything
    /// drop(foo2);   // Prints "dropped!"
    /// ```
    ///
    /// [`Frail`]: struct.Frail.html
    #[inline]
    fn drop(&mut self) {
        // FIXME(nox): If a RefMut was forgotten, this is the only strong
        // reference to the value but the strong reference counter is
        // MUTABLE_REFCOUNT, so the heap allocation will be leaked, and
        // the program may abort in Frail::upgrade.

        // Because `fetch_sub` is already atomic, we do not need to synchronize
        // with other threads unless we are going to delete the object. This
        // same logic applies to the below `fetch_sub` to the `frail` count.
        if self.inner().strong.fetch_sub(1, Release) != 1 {
            return;
        }

        // This fence is needed to prevent reordering of use of the data and
        // deletion of the data.  Because it is marked `Release`, the decreasing
        // of the reference count synchronizes with this `Acquire` fence. This
        // means that use of the data happens before decreasing the reference
        // count, which happens before this fence, which happens before the
        // deletion of the data.
        //
        // As explained in the [Boost documentation][1],
        //
        // > It is important to enforce any possible access to the object in one
        // > thread (through an existing reference) to *happen before* deleting
        // > the object in a different thread. This is achieved by a "release"
        // > operation after dropping a reference (any access to the object
        // > through this reference must obviously happened before), and an
        // > "acquire" operation before deleting the object.
        //
        // In particular, while the contents of an Arc are usually immutable,
        // it's possible to have interior writes to something like a Mutex<T>.
        // Since a Mutex is not acquired when it is deleted, we can't rely on
        // its synchronization logic to make writes in thread A visible to a
        // destructor running in thread B.
        //
        // Also note that the Acquire fence here could probably be replaced with
        // an Acquire load, which could improve performance in highly-contended
        // situations. See [2].
        //
        // [1]: (www.boost.org/doc/libs/1_55_0/doc/html/atomic/usage_examples.html)
        // [2]: (https://github.com/rust-lang/rust/pull/41714)
        atomic::fence(Acquire);

        #[inline(never)]
        unsafe fn drop_slow<T: ?Sized>(this: &mut Strong<T>) {
            // Destroy the data at this time, even though we may not free the box
            // allocation itself (there may still be frail pointers lying around).
            ptr::drop_in_place(&mut this.ptr.as_mut().data);

            if this.inner().frail.fetch_sub(1, Release) == 1 {
                atomic::fence(Acquire);
                alloc::dealloc(
                    this.ptr.cast().as_ptr(),
                    Layout::for_value(this.ptr.as_ref()),
                );
            }
        }

        unsafe {
            drop_slow(self);
        }
    }
}

impl<'b, T: ?Sized> RefMut<'b, T> {
    /// Creates a new [`Frail`][frail] pointer to this value.
    ///
    /// [frail]: struct.Frail.html
    ///
    /// # Examples
    ///
    /// ```
    /// use arbalest::Strong;
    ///
    /// let five = Strong::new(5);
    ///
    /// let frail_five = Strong::downgrade(&five);
    /// ```
    #[inline]
    pub fn downgrade(this: &Self) -> Frail<T> {
        this.inner().downgrade()
    }
}

impl<'b, T: ?Sized> Deref for RefMut<'b, T> {
    type Target = T;

    #[inline]
    fn deref(&self) -> &Self::Target {
        self.value
    }
}

impl<'b, T: ?Sized> DerefMut for RefMut<'b, T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.value
    }
}

impl<'b, T: ?Sized + fmt::Debug> fmt::Debug for RefMut<'b, T> {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.value.fmt(f)
    }
}

impl<'b, T: ?Sized + fmt::Display> fmt::Display for RefMut<'b, T> {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.value.fmt(f)
    }
}

impl<'b, T: ?Sized> Drop for RefMut<'b, T> {
    #[inline]
    fn drop(&mut self) {
        self.inner().strong.store(1, Release);
    }
}

impl fmt::Debug for BorrowMutError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("BorrowMutError").finish()
    }
}

impl fmt::Display for BorrowMutError {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.description().fmt(f)
    }
}

impl Error for BorrowMutError {
    fn description(&self) -> &str {
        "value is shared"
    }
}

impl<T> Frail<T> {
    /// Constructs a new `Frail<T>`, without allocating any memory.
    ///
    /// Calling [`upgrade`] on the return value always gives `None`.
    ///
    /// [`upgrade`]: #method.upgrade
    ///
    /// # Examples
    ///
    /// ```
    /// use arbalest::Frail;
    ///
    /// let empty: Frail<i64> = Frail::new();
    /// assert!(empty.upgrade().is_none());
    /// ```
    #[inline]
    pub fn new() -> Self {
        Self {
            ptr: NonNull::new(usize::MAX as *mut ArbalestInner<T>).expect("MAX is not 0"),
        }
    }
}

impl<T: ?Sized> Frail<T> {
    /// Attempts to upgrade the `Frail` pointer to an [`Strong`], extending
    /// the lifetime of the value if successful.
    ///
    /// Returns `None` if the value has since been dropped.
    ///
    /// [`Strong`]: struct.Strong.html
    ///
    /// # Panics
    ///
    /// Panics if the value is currently mutably borrowed by its single
    /// `Strong` reference.
    ///
    /// # Examples
    ///
    /// ```
    /// use arbalest::Strong;
    ///
    /// let five = Strong::new(5);
    ///
    /// let frail_five = Strong::downgrade(&five);
    ///
    /// let strong_five: Option<Strong<_>> = frail_five.upgrade();
    /// assert!(strong_five.is_some());
    ///
    /// // Destroy all strong pointers.
    /// drop(strong_five);
    /// drop(five);
    ///
    /// assert!(frail_five.upgrade().is_none());
    /// ```
    ///
    /// An example of panic:
    ///
    /// ```
    /// use arbalest::{Strong, Frail};
    /// use std::thread;
    ///
    /// let mut five = Strong::new(5);
    /// let frail_five = Strong::downgrade(&five);
    /// let b = Strong::borrow_mut(&mut five);
    /// let result = thread::spawn(move || {
    ///    let maybe_same_five = frail_five.upgrade(); // this causes a panic
    /// }).join();
    ///
    /// assert!(result.is_err());
    /// ```
    pub fn upgrade(&self) -> Option<Strong<T>> {
        match self.try_upgrade() {
            Ok(value) => Some(value),
            Err(UpgradeError::Dropped) => None,
            Err(UpgradeError::MutablyBorrowed) => panic!("already mutably borrowed"),
        }
    }

    /// Attempts to upgrade the `Frail` pointer to an [`Strong`], extending
    /// the lifetime of the value if successful.
    ///
    /// This is the non-panicking variant of [`upgrade`](#method.upgrade).
    ///
    /// Returns an error if the value has since been dropped or it is currently
    /// mutably borrowed.
    ///
    /// # Examples
    ///
    /// ```
    /// use arbalest::{Strong, Frail, UpgradeError};
    /// use std::mem;
    ///
    /// let mut five = Strong::new(5);
    /// let frail_five = Strong::downgrade(&five);
    /// assert!(frail_five.try_upgrade().is_ok());
    ///
    /// {
    ///     let b = Strong::borrow_mut(&mut five);
    ///     assert_eq!(frail_five.try_upgrade(), Err(UpgradeError::MutablyBorrowed));
    /// }
    ///
    /// drop(five);
    /// assert_eq!(frail_five.try_upgrade(), Err(UpgradeError::Dropped));
    /// ```
    /// [`Strong`]: struct.Strong.html
    pub fn try_upgrade(&self) -> Result<Strong<T>, UpgradeError> {
        // We use a CAS loop to increment the strong count instead of a
        // fetch_add because once the count hits 0 it must never be above 0.
        let inner = self.inner().ok_or(UpgradeError::Dropped)?;

        // This needs an Acquire load to synchronize with the Release write
        // in RefMut::drop.
        let mut n = inner.strong.load(Acquire);

        loop {
            if n == 0 {
                return Err(UpgradeError::Dropped);
            }

            if n == MUTABLE_REFCOUNT {
                return Err(UpgradeError::MutablyBorrowed);
            }

            // See comments in `Arc::clone` for why we do this (for `mem::forget`).
            // FIXME(nox): This may also happen if a Strong is mutably borrowed,
            // the RefMut is forgotten, then the Strong is cloned and a
            // a frail reference is upgraded.
            if n > MAX_REFCOUNT {
                process::abort();
            }

            // Relaxed is valid for the same reason it is on Arc's Clone impl.
            match inner
                .strong
                .compare_exchange_weak(n, n + 1, Relaxed, Relaxed)
            {
                Ok(_) => {
                    return Ok(Strong {
                        ptr: self.ptr,
                        phantom: PhantomData,
                    });
                }
                Err(old) => n = old,
            }
        }
    }

    /// Returns true if the two `Frail`s point to the same value (not just
    /// values that compare as equal).
    ///
    /// # Notes
    ///
    /// Since this compares pointers it means that `Frail::new()` will equal
    /// each other, even though they don't point to any value.
    ///
    /// # Examples
    ///
    /// ```
    /// use arbalest::{Strong, Frail};
    ///
    /// let first_rc = Strong::new(5);
    /// let first = Strong::downgrade(&first_rc);
    /// let second = Strong::downgrade(&first_rc);
    ///
    /// assert!(Frail::ptr_eq(&first, &second));
    ///
    /// let third_rc = Strong::new(5);
    /// let third = Strong::downgrade(&third_rc);
    ///
    /// assert!(!Frail::ptr_eq(&first, &third));
    /// ```
    ///
    /// Comparing `Frail::new`.
    ///
    /// ```
    /// use arbalest::{Strong, Frail};
    ///
    /// let first = Frail::new();
    /// let second = Frail::new();
    /// assert!(Frail::ptr_eq(&first, &second));
    ///
    /// let third_rc = Strong::new(());
    /// let third = Strong::downgrade(&third_rc);
    /// assert!(!Frail::ptr_eq(&first, &third));
    /// ```
    #[inline]
    pub fn ptr_eq(this: &Self, other: &Self) -> bool {
        this.ptr.as_ptr() == other.ptr.as_ptr()
    }
}

impl<T: ?Sized> Clone for Frail<T> {
    /// Makes a clone of the `Frail` pointer that points to the same value.
    ///
    /// # Examples
    ///
    /// ```
    /// use arbalest::{Strong, Frail};
    ///
    /// let frail_five = Strong::downgrade(&Strong::new(5));
    ///
    /// let _ = Frail::clone(&frail_five);
    /// ```
    #[inline]
    fn clone(&self) -> Self {
        self.inner()
            .map_or_else(|| Self { ptr: self.ptr }, ArbalestInner::downgrade)
    }
}

impl<T> Default for Frail<T> {
    /// Constructs a new `Frail<T>`, without allocating memory.
    ///
    /// Calling [`upgrade`] on the return value always gives `None`.
    ///
    /// [`upgrade`]: #method.upgrade
    ///
    /// # Examples
    ///
    /// ```
    /// use arbalest::Frail;
    ///
    /// let empty: Frail<i64> = Default::default();
    /// assert!(empty.upgrade().is_none());
    /// ```
    fn default() -> Self {
        Self::new()
    }
}

impl<T: ?Sized + fmt::Debug> fmt::Debug for Frail<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "(Frail)")
    }
}

impl<T: ?Sized> Drop for Frail<T> {
    /// Drops the `Frail` pointer.
    ///
    /// # Examples
    ///
    /// ```
    /// use arbalest::{Strong, Frail};
    ///
    /// struct Foo;
    ///
    /// impl Drop for Foo {
    ///     fn drop(&mut self) {
    ///         println!("dropped!");
    ///     }
    /// }
    ///
    /// let foo = Strong::new(Foo);
    /// let frail_foo = Strong::downgrade(&foo);
    /// let other_frail_foo = Frail::clone(&frail_foo);
    ///
    /// drop(frail_foo);   // Doesn't print anything
    /// drop(foo);        // Prints "dropped!"
    ///
    /// assert!(other_frail_foo.upgrade().is_none());
    /// ```
    fn drop(&mut self) {
        // If we find out that we were the last frail pointer, then its time
        // to deallocate the data entirely. See the discussion in Strong::drop()
        // about the memory orderings.
        let inner = if let Some(inner) = self.inner() {
            inner
        } else {
            return;
        };

        if inner.frail.fetch_sub(1, Release) != 1 {
            return;
        }

        atomic::fence(Acquire);
        unsafe {
            alloc::dealloc(
                self.ptr.cast().as_ptr(),
                Layout::for_value(self.ptr.as_ref()),
            );
        }
    }
}

impl fmt::Display for UpgradeError {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.description().fmt(f)
    }
}

impl Error for UpgradeError {
    fn description(&self) -> &str {
        match *self {
            UpgradeError::Dropped => "value was dropped",
            UpgradeError::MutablyBorrowed => "value is mutably borrowed",
        }
    }
}

impl<T: ?Sized> Strong<T> {
    fn inner(&self) -> &ArbalestInner<T> {
        // This unsafety is ok because while this arc is alive we're guaranteed
        // that the inner pointer is valid. Furthermore, we know that the
        // `ArbalestInner` structure itself is `Sync` because the inner data is
        // `Sync` as well, so we're ok loaning out an immutable pointer to these
        // contents.
        unsafe { self.ptr.as_ref() }
    }
}

impl<'b, T: ?Sized> RefMut<'b, T> {
    fn inner(&self) -> &ArbalestInner<T> {
        // FIXME(nox): this.inner() is an immutable reference to the structure
        // which holds the UnsafeCell<T> of which this RefMut holds a mutable
        // reference. Is this ok? Given how RefCell<T> is implemented, I am
        // pretty sure it is.
        unsafe { &*ArbalestInner::from_raw(self.value) }
    }
}

impl<T: ?Sized> Frail<T> {
    /// Return `None` when the pointer is dangling and there is no allocated
    /// `ArbalestInner`, i.e., this `Frail` was created by `Frail::new`.
    fn inner(&self) -> Option<&ArbalestInner<T>> {
        if is_dangling(self.ptr) {
            None
        } else {
            Some(unsafe { self.ptr.as_ref() })
        }
    }
}

/// Sets the data pointer of a `?Sized` raw pointer.
///
/// For a slice/trait object, this sets the `data` field and leaves the rest
/// unchanged. For a sized raw pointer, this simply sets the pointer.
unsafe fn set_data_ptr<T: ?Sized, U>(mut ptr: *mut T, data: *mut U) -> *mut T {
    ptr::write(&mut ptr as *mut _ as *mut *mut u8, data as *mut u8);
    ptr
}

impl<T: ?Sized> ArbalestInner<T> {
    fn downgrade(&self) -> Frail<T> {
        // Using a relaxed ordering is alright here, as knowledge of the
        // original reference prevents other threads from erroneously deleting
        // the object.
        //
        // As explained in the [Boost documentation][1], Increasing the
        // reference counter can always be done with memory_order_relaxed: New
        // references to an object can only be formed from an existing
        // reference, and passing an existing reference from one thread to
        // another must already provide any required synchronization.
        //
        // [1]: (www.boost.org/doc/libs/1_55_0/doc/html/atomic/usage_examples.html)
        let old_frail_refcount = self.frail.fetch_add(1, Relaxed);

        // However we need to guard against massive refcounts in case someone
        // is `mem::forget`ing Arbalests. If we don't do this the count can
        // overflow and users will use-after free. We racily saturate to
        // `isize::MAX` on the assumption that there aren't ~2 billion threads
        // incrementing the reference count at once. This branch will never be
        // taken in any realistic program.
        //
        // We abort because such a program is incredibly degenerate, and we
        // don't care to support it.
        if old_frail_refcount > MAX_REFCOUNT {
            process::abort();
        }

        Frail { ptr: self.into() }
    }

    unsafe fn from_raw(ptr: *const T) -> *mut Self {
        // Align the unsized value to the end of the ArcInner.
        // Because it is ?Sized, it will always be the last field in memory.
        let align = mem::align_of_val(&*ptr);
        let layout = Layout::new::<ArbalestInner<()>>();
        let offset = (layout.size() + padding_needed_for(&layout, align)) as isize;

        // Reverse the offset to find the original ArcInner.
        let fake_ptr = ptr as *mut ArbalestInner<T>;
        set_data_ptr(fake_ptr, (ptr as *mut u8).offset(-offset))
    }
}

/// Returns the amount of padding we must insert after `layout` to ensure that
/// the following address will satisfy `align` (measured in bytes).
///
/// E.g., if `self.size()` is 9, then `self.padding_needed_for(4)` returns 3,
/// because that is the minimum number of bytes of padding required to get a
/// 4-aligned address (assuming that the corresponding memory block starts at
/// a 4-aligned address).
///
/// The return value of this function has no meaning if `align` is not
/// a power-of-two.
///
/// Note that the utility of the returned value requires `align` to be less than
/// or equal to the alignment of the starting address for the whole allocated
/// block of memory. One way to satisfy this constraint is to ensure
/// `align <= layout.align()`.
fn padding_needed_for(layout: &Layout, align: usize) -> usize {
    let len = layout.size();

    // Rounded up value is:
    //   len_rounded_up = (len + align - 1) & !(align - 1);
    // and then we return the padding difference: `len_rounded_up - len`.
    //
    // We use modular arithmetic throughout:
    //
    // 1. align is guaranteed to be > 0, so align - 1 is always
    //    valid.
    //
    // 2. `len + align - 1` can overflow by at most `align - 1`,
    //    so the &-mask wth `!(align - 1)` will ensure that in the
    //    case of overflow, `len_rounded_up` will itself be 0.
    //    Thus the returned padding, when added to `len`, yields 0,
    //    which trivially satisfies the alignment `align`.
    //
    // (Of course, attempts to allocate blocks of memory whose
    // size and padding overflow in the above manner should cause
    // the allocator to yield an error anyway.)

    let len_rounded_up = len.wrapping_add(align).wrapping_sub(1) & !align.wrapping_sub(1);
    len_rounded_up.wrapping_sub(len)
}

fn is_dangling<T: ?Sized>(ptr: NonNull<T>) -> bool {
    let address = ptr.as_ptr() as *mut () as usize;
    address == usize::MAX
}

/// A soft limit on the amount of references that may be made.
///
/// Going above this limit will abort your program (although not
/// necessarily) at _exactly_ `MAX_REFCOUNT + 1` references.
const MAX_REFCOUNT: usize = (isize::MAX) as usize;

/// A sentinel value for a strong reference counter of a mutably borrowed value.
const MUTABLE_REFCOUNT: usize = (!MAX_REFCOUNT) | ((!MAX_REFCOUNT) >> 1) | 1;
