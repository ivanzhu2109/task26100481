package vecindex

import (
	"hash/fnv"
	"sort"
	"sync/atomic"
)

// numBuckets is the fixed bucket count of a generation's cell table.
// The table only grows by prepending immutable cells, so readers can
// traverse it concurrently with commits.
const numBuckets = 1024

// genCell is a per-document entry inside a generation.
type genCell struct {
	docID string
	chain versionCell
	next  *genCell // immutable after publication
}

// generation is a materialized search view: its own cell table and
// version chains. A generation may share immutable vector bytes with
// other generations, but never mutable search state. Once a generation
// is swapped out it receives no further writes and serves as a stable
// snapshot for in-flight queries.
type generation struct {
	id      uint64
	buckets [numBuckets]atomic.Pointer[genCell]
}

func newGeneration(id uint64) *generation {
	return &generation{id: id}
}

func bucketOf(docID string) uint32 {
	h := fnv.New32a()
	_, _ = h.Write([]byte(docID))
	return h.Sum32() % numBuckets
}

// cellFor returns the cell for docID, creating and publishing it if
// needed. Safe for concurrent use; creation is idempotent under CAS.
func (g *generation) cellFor(docID string) *genCell {
	b := &g.buckets[bucketOf(docID)]
	for c := b.Load(); c != nil; c = c.next {
		if c.docID == docID {
			return c
		}
	}
	nc := &genCell{docID: docID}
	for {
		h := b.Load()
		nc.next = h
		if b.CompareAndSwap(h, nc) {
			return nc
		}
		// Re-check: a concurrent insert may have published the same docID.
		for c := b.Load(); c != nil; c = c.next {
			if c.docID == docID {
				return c
			}
		}
	}
}

// insertLive commits a live mutation into the generation. Callers are
// serialized by Index.mu, so rev exceeds every rev already present.
func (g *generation) insertLive(docID string, n *verNode, floor uint64) {
	c := g.cellFor(docID)
	c.chain.prepend(n)
	c.chain.prune(floor)
}

// insertBackfill installs a backfilled historical version, losing to any
// newer version already committed into this generation.
func (g *generation) insertBackfill(docID string, n *verNode) {
	c := g.cellFor(docID)
	c.chain.prependIfNewer(n)
}

// search scores q against the newest version of every document with
// rev <= r, filtered by tag, and returns the ranked top results.
// Runs entirely against immutable nodes; holds no locks.
func (g *generation) search(q []float32, tag string, r uint64, minScore float32, limit int) []Result {
	var out []Result
	for i := range g.buckets {
		for c := g.buckets[i].Load(); c != nil; c = c.next {
			n := c.chain.asOf(r)
			if n == nil || n.tombstone || !n.tags.has(tag) {
				continue
			}
			chunks := len(n.vec) / Dim
			for ch := 0; ch < chunks; ch++ {
				s := dot(q, n.vec[ch*Dim:(ch+1)*Dim])
				if s >= minScore {
					out = append(out, Result{DocID: c.docID, Chunk: ch, Rev: n.rev, Score: s})
				}
			}
		}
	}
	sort.Slice(out, func(i, j int) bool {
		a, b := out[i], out[j]
		if a.Score != b.Score {
			return a.Score > b.Score
		}
		if a.DocID != b.DocID {
			return a.DocID < b.DocID
		}
		return a.Chunk < b.Chunk
	})
	if len(out) > limit {
		out = out[:limit]
	}
	return out
}

// dot computes the dot product of two Dim-length vectors. Inputs are
// unit vectors, so this is the cosine similarity.
func dot(a, b []float32) float32 {
	var s float64
	for i := 0; i < Dim; i++ {
		s += float64(a[i]) * float64(b[i])
	}
	return float32(s)
}
