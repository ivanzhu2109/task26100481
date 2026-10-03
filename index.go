package vecindex

import (
	"sort"
	"sync"
	"sync/atomic"
)

// Index is an in-process vector index supporting online rolling rebuilds.
//
// Concurrency model: commits (Upsert/Delete) are serialized by mu, stamped
// with a strictly increasing revision, recorded in the authoritative docs
// map, and pushed as immutable versions into every live generation (the
// serving one plus the one being rebuilt, if any). Search never takes mu:
// it reads a commit watermark, selects the serving generation, and scores a
// lock-free snapshot of immutable versions.
type Index struct {
	mu          sync.Mutex
	seq         atomic.Uint64
	docs        map[string]*docNode
	head        *docNode // oldest appended; guarded by mu
	tail        *docNode // guarded by mu
	serving     atomic.Pointer[generation]
	building    *generation            // guarded by mu
	rebuilding  bool                   // guarded by mu
	rebuildDone chan struct{}          // guarded by mu; closed when rebuild finishes
	gens        map[uint64]*generation // guarded by mu; diagnostic registry
	hooks       hooks
}

// New returns an empty index.
func New() *Index {
	ix := &Index{docs: make(map[string]*docNode), gens: make(map[uint64]*generation)}
	g := newGeneration(1)
	ix.gens[g.id] = g
	ix.serving.Store(g)
	return ix
}

// Upsert atomically replaces the tags and all chunk vectors of docID and
// returns the new strictly increasing revision number.
func (ix *Index) Upsert(docID string, tags []string, vectors [][]float32) uint64 {
	tagSet := make(map[string]struct{}, len(tags))
	for _, t := range tags {
		tagSet[t] = struct{}{}
	}
	chunks := make([][]float32, len(vectors))
	for i, v := range vectors {
		cp := make([]float32, len(v))
		copy(cp, v)
		chunks[i] = cp
	}

	ix.mu.Lock()
	rev := ix.seq.Add(1)
	rec := &record{rev: rev, tags: tagSet, chunks: chunks}
	if n, ok := ix.docs[docID]; ok {
		n.rec = rec
	} else {
		n := &docNode{id: docID, rec: rec, prev: ix.tail}
		if ix.tail != nil {
			ix.tail.next = n
		} else {
			ix.head = n
		}
		ix.tail = n
		ix.docs[docID] = n
	}
	v := &docVer{seq: rev, rev: rev, tags: tagSet, chunks: chunks}
	ix.serving.Load().applyDelta(docID, v)
	if ix.building != nil {
		ix.building.applyDelta(docID, v)
	}
	ix.mu.Unlock()
	return rev
}

// Delete atomically removes docID. The same docID may be re-upserted later.
// Deleting a non-existent document is a no-op.
func (ix *Index) Delete(docID string) {
	ix.mu.Lock()
	n, ok := ix.docs[docID]
	if !ok {
		ix.mu.Unlock()
		return
	}
	rev := ix.seq.Add(1)
	delete(ix.docs, docID)
	if n.prev != nil {
		n.prev.next = n.next
	} else {
		ix.head = n.next
	}
	if n.next != nil {
		n.next.prev = n.prev
	} else {
		ix.tail = n.prev
	}
	v := &docVer{seq: rev, rev: rev, deleted: true}
	ix.serving.Load().applyDelta(docID, v)
	if ix.building != nil {
		ix.building.applyDelta(docID, v)
	}
	ix.mu.Unlock()
}

// Search returns the top chunks of documents carrying tag: cosine score at
// least ScoreThreshold, at most MaxResults, ties broken by docID then chunk
// index ascending. The result is equivalent to a search over the fully
// committed document set at a single instant within the call.
func (ix *Index) Search(q []float32, tag string) []SearchResult {
	// Watermark first, then generation: any generation observable after the
	// watermark read contains every commit up to the watermark.
	wm := ix.seq.Load()
	g := ix.serving.Load()
	g.searches.Add(1)
	if ix.hooks.onSearchView != nil {
		ix.hooks.onSearchView(g.id)
	}

	entries := g.snapshotEntries()
	var hits []SearchResult
	for _, e := range entries {
		v := viewOf(e.head, wm)
		if v == nil || v.deleted {
			continue
		}
		if _, ok := v.tags[tag]; !ok {
			continue
		}
		for ci, vec := range v.chunks {
			var dot float32
			for d := 0; d < len(vec) && d < len(q); d++ {
				dot += q[d] * vec[d]
			}
			if dot >= ScoreThreshold {
				hits = append(hits, SearchResult{
					DocID: e.docID,
					Chunk: ci,
					Rev:   v.rev,
					Score: dot,
				})
			}
		}
	}
	sort.Slice(hits, func(i, j int) bool {
		a, b := hits[i], hits[j]
		if a.Score != b.Score {
			return a.Score > b.Score
		}
		if a.DocID != b.DocID {
			return a.DocID < b.DocID
		}
		return a.Chunk < b.Chunk
	})
	if len(hits) > MaxResults {
		hits = hits[:MaxResults]
	}
	return hits
}
