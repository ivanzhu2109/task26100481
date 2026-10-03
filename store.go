package vecindex

import "sync"

// tagSet is a per-document immutable set of tags, shared (read-only)
// between the authoritative record and generation version nodes.
type tagSet map[string]struct{}

func makeTagSet(tags []string) tagSet {
	s := make(tagSet, len(tags))
	for _, t := range tags {
		s[t] = struct{}{}
	}
	return s
}

func (s tagSet) has(tag string) bool {
	_, ok := s[tag]
	return ok
}

// store is the authoritative record of all committed documents. Each
// document owns a version chain; every successful modification prepends
// an immutable node, so the rebuilder can read a consistent as-of
// snapshot without holding any lock while it traverses.
//
// Mutations happen only under Index.mu. The revision counter is also
// driven by Index (deletes consume revisions too).
type store struct {
	cells sync.Map // docID -> *versionCell
}

func newStore() *store {
	return &store{}
}

// cellFor returns the cell for docID, creating it if needed.
// Must be called under Index.mu.
func (s *store) cellFor(docID string) *versionCell {
	c, _ := s.cells.LoadOrStore(docID, &versionCell{})
	return c.(*versionCell)
}

// upsert commits a new version of docID with the given revision. The
// caller must have copied vectors already; the slice is shared
// (read-only) with the generation version nodes. Must be called under
// Index.mu.
func (s *store) upsert(docID string, rev uint64, tags tagSet, vectors []float32, floor uint64) {
	c := s.cellFor(docID)
	c.prepend(&verNode{rev: rev, tags: tags, vec: vectors})
	c.prune(floor)
}

// remove commits a tombstone for docID. Must be called under Index.mu.
func (s *store) remove(docID string, rev uint64, floor uint64) {
	c := s.cellFor(docID)
	c.prepend(&verNode{rev: rev, tombstone: true})
	c.prune(floor)
}

// latest returns the newest committed node for docID, or nil.
func (s *store) latest(docID string) *verNode {
	c, ok := s.cells.Load(docID)
	if !ok {
		return nil
	}
	return c.(*versionCell).head.Load()
}

// forEachAsOf invokes fn for every document whose newest version with
// rev <= r is a live (non-tombstone) record. Runs lock-free; safe to
// call concurrently with commits and to block inside fn.
func (s *store) forEachAsOf(r uint64, fn func(docID string, n *verNode)) {
	s.cells.Range(func(key, value any) bool {
		n := value.(*versionCell).asOf(r)
		if n != nil && !n.tombstone {
			fn(key.(string), n)
		}
		return true
	})
}
