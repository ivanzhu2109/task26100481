package vecindex

import (
	"errors"
	"math"
	"sync"
	"sync/atomic"
)

// Tuning constants for Search.
const (
	minScore = 0.35
	maxHits  = 20
)

var (
	// ErrEmptyDocID is returned when an operation gets an empty docID.
	ErrEmptyDocID = errors.New("vecindex: empty docID")
	// ErrBadVectorLen is returned when the vector payload is not a
	// multiple of Dim (Upsert) or not exactly Dim (Search).
	ErrBadVectorLen = errors.New("vecindex: vector length mismatch")
)

// readerSlot tracks one in-flight search: rev is the snapshot revision
// the search reads at, or math.MaxUint64 while not yet determined.
type readerSlot struct {
	rev atomic.Uint64
}

// readerRegistry tracks in-flight searches so version-chain pruning
// never drops nodes a paused search may still need.
type readerRegistry struct {
	mu    sync.Mutex
	slots map[*readerSlot]struct{}
}

func newReaderRegistry() *readerRegistry {
	return &readerRegistry{slots: make(map[*readerSlot]struct{})}
}

func (r *readerRegistry) enter() *readerSlot {
	s := &readerSlot{}
	s.rev.Store(math.MaxUint64)
	r.mu.Lock()
	r.slots[s] = struct{}{}
	r.mu.Unlock()
	return s
}

func (r *readerRegistry) leave(s *readerSlot) {
	r.mu.Lock()
	delete(r.slots, s)
	r.mu.Unlock()
}

// minRev returns the smallest snapshot revision any registered reader
// may still need, or 0 if some reader has not pinned its revision yet
// (which disables pruning), or math.MaxUint64 when no reader exists.
func (r *readerRegistry) minRev() uint64 {
	r.mu.Lock()
	defer r.mu.Unlock()
	m := uint64(math.MaxUint64)
	for s := range r.slots {
		v := s.rev.Load()
		if v == math.MaxUint64 {
			return 0
		}
		if v < m {
			m = v
		}
	}
	return m
}

// Index is an in-process vector index with online rolling rebuild.
//
// Concurrency model: commits (Upsert/Delete) are serialized by mu and
// applied to the authoritative store plus the active generation (and
// the building generation during Rebuild) as immutable version nodes.
// Searches pin a revision and read only immutable nodes, so they never
// hold locks while scoring and never observe partial commits.
type Index struct {
	mu       sync.Mutex
	docs     *store
	nextRev  uint64
	docCount int
	genSeq   uint64
	building *generation

	committed atomic.Uint64
	active    atomic.Pointer[generation]
	// rebuildRb is the snapshot revision of the in-flight rebuild, or
	// math.MaxUint64 when no rebuild is running.
	rebuildRb atomic.Uint64

	rebuildMu sync.Mutex
	readers   *readerRegistry

	hooks Hooks
}

// New creates an empty index.
func New(opts ...Option) *Index {
	ix := &Index{
		docs:    newStore(),
		readers: newReaderRegistry(),
	}
	ix.active.Store(newGeneration(0))
	ix.rebuildRb.Store(math.MaxUint64)
	for _, o := range opts {
		o(ix)
	}
	return ix
}

// pruneFloor returns the revision floor safe for chain pruning: the
// minimum of in-flight reader revisions and the rebuild snapshot.
// Must be called under ix.mu.
func (ix *Index) pruneFloor() uint64 {
	floor := ix.readers.minRev()
	if rb := ix.rebuildRb.Load(); rb < floor {
		floor = rb
	}
	return floor
}

// Upsert atomically replaces the document's tags and all chunk vectors,
// returning the new strictly increasing revision number.
func (ix *Index) Upsert(docID string, tags []string, vectors []float32) (uint64, error) {
	if docID == "" {
		return 0, ErrEmptyDocID
	}
	if len(vectors)%Dim != 0 {
		return 0, ErrBadVectorLen
	}
	vec := make([]float32, len(vectors))
	copy(vec, vectors)
	ts := makeTagSet(tags)

	ix.mu.Lock()
	defer ix.mu.Unlock()
	ix.nextRev++
	rev := ix.nextRev
	floor := ix.pruneFloor()

	if prev := ix.docs.latest(docID); prev == nil || prev.tombstone {
		ix.docCount++
	}
	ix.docs.upsert(docID, rev, ts, vec, floor)

	node := &verNode{rev: rev, tags: ts, vec: vec}
	ix.active.Load().insertLive(docID, node, floor)
	if ix.building != nil {
		ix.building.insertLive(docID, &verNode{rev: rev, tags: ts, vec: vec}, floor)
	}
	ix.committed.Store(rev)
	return rev, nil
}

// Delete atomically removes the document. The docID may be reused later.
// Deleting a non-existent document is a no-op.
func (ix *Index) Delete(docID string) error {
	if docID == "" {
		return ErrEmptyDocID
	}
	ix.mu.Lock()
	defer ix.mu.Unlock()
	prev := ix.docs.latest(docID)
	if prev == nil || prev.tombstone {
		return nil
	}
	ix.nextRev++
	rev := ix.nextRev
	floor := ix.pruneFloor()

	ix.docCount--
	ix.docs.remove(docID, rev, floor)

	node := &verNode{rev: rev, tombstone: true}
	ix.active.Load().insertLive(docID, node, floor)
	if ix.building != nil {
		ix.building.insertLive(docID, &verNode{rev: rev, tombstone: true}, floor)
	}
	ix.committed.Store(rev)
	return nil
}

// Search returns the top chunks of documents carrying tag, with cosine
// score >= 0.35, ranked by score desc then (docID, chunk) asc. The
// result is equivalent to a query executed at a single instant between
// the call's start and return.
func (ix *Index) Search(q []float32, tag string) ([]Result, error) {
	if len(q) != Dim {
		return nil, ErrBadVectorLen
	}
	slot := ix.readers.enter()
	defer ix.readers.leave(slot)

	// Pin the snapshot revision first, then select the generation.
	// Both are atomic loads; the selected generation always contains
	// every commit with rev <= r (commits publish to the generation
	// before bumping committed, and a swapped-in generation contains
	// the full prefix up to the swap).
	r := ix.committed.Load()
	slot.rev.Store(r)
	g := ix.active.Load()

	if ix.hooks.OnSearchViewSelected != nil {
		ix.hooks.OnSearchViewSelected(g.id)
	}
	return g.search(q, tag, r, minScore, maxHits), nil
}

// Rebuild constructs a fresh independent search generation from the
// authoritative records and atomically swaps it in, returning after the
// swap. At most one Rebuild runs at a time; concurrent Rebuild calls
// are serialized. Reads and writes proceed throughout.
func (ix *Index) Rebuild() error {
	ix.rebuildMu.Lock()
	defer ix.rebuildMu.Unlock()

	ix.mu.Lock()
	rb := ix.committed.Load()
	ix.genSeq++
	ng := newGeneration(ix.genSeq)
	ix.building = ng
	ix.rebuildRb.Store(rb)
	ix.mu.Unlock()

	// Backfill from the authoritative records as of rb. Holds no
	// locks; live commits are dual-written into ng and always win
	// against stale backfill via revision comparison.
	ix.docs.forEachAsOf(rb, func(docID string, n *verNode) {
		if ix.hooks.OnBackfillDoc != nil {
			ix.hooks.OnBackfillDoc(docID, n.rev)
		}
		ng.insertBackfill(docID, &verNode{rev: n.rev, tags: n.tags, vec: n.vec})
	})

	ix.mu.Lock()
	ix.active.Store(ng)
	ix.building = nil
	ix.rebuildRb.Store(math.MaxUint64)
	ix.mu.Unlock()
	return nil
}

// Stats returns diagnostic information about the index.
func (ix *Index) Stats() Stats {
	ix.mu.Lock()
	defer ix.mu.Unlock()
	return Stats{
		GenerationID: ix.active.Load().id,
		Rebuilding:   ix.rebuildRb.Load() != math.MaxUint64,
		Docs:         ix.docCount,
	}
}
