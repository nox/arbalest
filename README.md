# Arbalest

This is a thread-safe reference-counted smart pointer with weak references.

## How is it different from `Arc<T>`?

The weak references don't prevent mutable access to the inner value,
`Arbalest::borrow_mut` will always succeed as long as there is a single
`Arbalest<T>` handle to the value that should be mutated.

## How can I help?

I have no clue whether my atomic orderings are correct, notably in
`Arbalest::borrow_mut` and `Fragile::upgrade`.
