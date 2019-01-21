# Arbalest

This is a thread-safe reference-counted smart pointer with weak references.

## How is it different from `Arc<T>`?

The weak references don't prevent mutable access to the inner value,
`Strong::borrow_mut` will always succeed as long as there is a single
`Strong<T>` handle to the value that should be mutated.

## How can I help?

Improve documentation, review the code (most importantly the atomic operations),
make use of it.
