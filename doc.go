// Package vecindex is an in-process vector index over 32-dimensional
// unit vectors, supporting online rolling rebuilds while reads and
// writes proceed concurrently.
//
// The index keeps two real, separate structures:
//
//   - an authoritative store of committed document records, and
//   - a materialized search generation used to answer Search queries.
//
// Every committed modification is an immutable version node carrying
// the document's tags, revision, and chunk vectors from a single
// commit. Commits are serialized and applied to the store and the
// active generation (plus the generation under construction during a
// Rebuild). Search pins the current committed revision and reads only
// immutable nodes from the generation it selected, so every result is
// equivalent to a query executed at a single instant, never holds
// locks while scoring, and never observes partial commits.
//
// Rebuild constructs a fresh generation from the authoritative records
// while live commits are dual-written into it; stale backfilled
// revisions always lose to newer live commits. The new generation is
// swapped in atomically, and in-flight queries finish on the view they
// selected. Generations may share immutable vector bytes but never
// mutable search state.
package vecindex
