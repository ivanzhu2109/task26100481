package vecindex

import (
	"fmt"
	"math"
	"math/rand"
	"sort"
	"sync"
	"testing"
)

type refDoc struct {
	rev    uint64
	tags   map[string]struct{}
	chunks [][]float32
}

// reference is an independent sequential model of the index.
type reference struct {
	mu      sync.Mutex
	docs    map[string]refDoc
	history map[string]map[uint64]refDoc
}

func (r *reference) upsert(id string, tags []string, vecs [][]float32, rev uint64) {
	tagSet := map[string]struct{}{}
	for _, t := range tags {
		tagSet[t] = struct{}{}
	}
	d := refDoc{rev: rev, tags: tagSet, chunks: vecs}
	r.mu.Lock()
	r.docs[id] = d
	if r.history[id] == nil {
		r.history[id] = map[uint64]refDoc{}
	}
	r.history[id][rev] = d
	r.mu.Unlock()
}

func (r *reference) delete(id string) {
	r.mu.Lock()
	delete(r.docs, id)
	r.mu.Unlock()
}

func (r *reference) search(q []float32, tag string) []SearchResult {
	r.mu.Lock()
	var hits []SearchResult
	for id, d := range r.docs {
		if _, ok := d.tags[tag]; !ok {
			continue
		}
		for ci, vec := range d.chunks {
			var dot float32
			for i := range vec {
				dot += q[i] * vec[i]
			}
			if dot >= ScoreThreshold {
				hits = append(hits, SearchResult{DocID: id, Chunk: ci, Rev: d.rev, Score: dot})
			}
		}
	}
	r.mu.Unlock()
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

// checkHit verifies a hit against every revision the document ever had:
// it must match one committed state exactly (no torn or fabricated state).
func (r *reference) checkHit(t *testing.T, h SearchResult, tag string, q []float32) {
	t.Helper()
	r.mu.Lock()
	vers := r.history[h.DocID]
	d, ok := vers[h.Rev]
	r.mu.Unlock()
	if !ok {
		t.Fatalf("hit references never-committed revision: %+v", h)
	}
	if _, ok := d.tags[tag]; !ok {
		t.Fatalf("hit doc lacks tag at its revision: %+v", h)
	}
	if h.Chunk >= len(d.chunks) {
		t.Fatalf("hit chunk out of range at its revision: %+v", h)
	}
	var dot float32
	for i := range d.chunks[h.Chunk] {
		dot += q[i] * d.chunks[h.Chunk][i]
	}
	if math.Abs(float64(dot-h.Score)) > 1e-5 {
		t.Fatalf("hit score %v recomputed %v", h.Score, dot)
	}
}

func randUnitVec(rng *rand.Rand) []float32 {
	v := make([]float32, Dim)
	var norm float64
	for i := range v {
		f := rng.NormFloat64()
		v[i] = float32(f)
		norm += f * f
	}
	norm = math.Sqrt(norm)
	for i := range v {
		v[i] = float32(float64(v[i]) / norm)
	}
	return v
}

// blend returns a unit vector close to q (cosine ~= mix with q dominant).
func blend(rng *rand.Rand, q []float32) []float32 {
	v := randUnitVec(rng)
	out := make([]float32, Dim)
	var norm float64
	for i := range out {
		f := 0.85*float64(q[i]) + 0.15*float64(v[i])
		out[i] = float32(f)
		norm += f * f
	}
	norm = math.Sqrt(norm)
	for i := range out {
		out[i] = float32(float64(out[i]) / norm)
	}
	return out
}

func docID(i int) string { return fmt.Sprintf("doc%06d", i) }

func TestStressRebuildUnderLoad(t *testing.T) {
	const numDocs = 50000
	const numChunks = 4
	const numTags = 8

	rng := rand.New(rand.NewSource(42))
	q := randUnitVec(rng)
	tags := make([]string, numTags)
	for i := range tags {
		tags[i] = fmt.Sprintf("tag%d", i)
	}

	ix := New()
	ref := &reference{docs: map[string]refDoc{}, history: map[string]map[uint64]refDoc{}}

	genVecs := func(i int) [][]float32 {
		vecs := make([][]float32, numChunks)
		for c := range vecs {
			if i%50 == 0 {
				vecs[c] = blend(rng, q) // correlated: clears the threshold
			} else {
				vecs[c] = randUnitVec(rng)
			}
		}
		return vecs
	}

	for i := 0; i < numDocs; i++ {
		vecs := genVecs(i)
		tag := []string{tags[i%numTags]}
		rev := ix.Upsert(docID(i), tag, vecs)
		ref.upsert(docID(i), tag, vecs, rev)
	}

	stop := make(chan struct{})
	var searchWG sync.WaitGroup
	for g := 0; g < 4; g++ {
		searchWG.Add(1)
		go func(g int) {
			defer searchWG.Done()
			for {
				select {
				case <-stop:
					return
				default:
				}
				res := ix.Search(q, tags[g%numTags])
				if len(res) > MaxResults {
					t.Errorf("too many hits: %d", len(res))
					return
				}
				for i, h := range res {
					if h.Score < ScoreThreshold {
						t.Errorf("hit below threshold: %+v", h)
						return
					}
					if i > 0 {
						p := res[i-1]
						if p.Score < h.Score ||
							(p.Score == h.Score && (p.DocID > h.DocID ||
								(p.DocID == h.DocID && p.Chunk > h.Chunk))) {
							t.Errorf("misordered hits: %+v then %+v", p, h)
							return
						}
					}
					ref.checkHit(t, h, tags[g%numTags], q)
				}
			}
		}(g)
	}

	// Concurrent mutations while a rebuild is in flight.
	rebuildDone := make(chan struct{})
	go func() {
		ix.Rebuild()
		close(rebuildDone)
	}()

	mutRng := rand.New(rand.NewSource(7))
	for op := 0; op < 3000; op++ {
		i := mutRng.Intn(numDocs)
		id := docID(i)
		switch mutRng.Intn(3) {
		case 0:
			ix.Delete(id)
			ref.delete(id)
		default:
			vecs := genVecs(mutRng.Intn(1 << 30))
			tag := []string{tags[mutRng.Intn(numTags)]}
			rev := ix.Upsert(id, tag, vecs)
			ref.upsert(id, tag, vecs, rev)
		}
	}
	<-rebuildDone
	close(stop)
	searchWG.Wait()

	// Quiesced: the index must exactly match the sequential reference.
	for _, tag := range tags {
		got := ix.Search(q, tag)
		want := ref.search(q, tag)
		if len(got) != len(want) {
			t.Fatalf("tag %s: got %d hits, want %d", tag, len(got), len(want))
		}
		for i := range want {
			if got[i] != want[i] {
				t.Fatalf("tag %s hit %d: got %+v want %+v", tag, i, got[i], want[i])
			}
		}
	}
}
