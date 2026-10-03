package vecindex

import (
	"fmt"
	"math"
	"math/rand"
	"sort"
	"sync"
	"sync/atomic"
	"testing"
)

// splitmix64 is a deterministic PRNG used to derive reproducible
// vectors from (doc, chunk, salt) without storing them.
func splitmix64(x *uint64) uint64 {
	*x += 0x9E3779B97F4A7C15
	z := *x
	z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
	z = (z ^ (z >> 27)) * 0x94D049BB133111EB
	return z ^ (z >> 31)
}

// stressVec returns a deterministic pseudo-random unit vector.
func stressVec(doc, chunk, salt uint64) []float32 {
	seed := doc*0x100000001B3 + chunk*0xC2B2AE3D27D4EB4F + salt*0x165667B19E3779F9
	v := make([]float32, Dim)
	var norm float64
	for i := range v {
		f := float64(splitmix64(&seed)>>11)*(1.0/(1<<53)) - 0.5
		v[i] = float32(f)
		norm += float64(v[i]) * float64(v[i])
	}
	norm = math.Sqrt(norm)
	for i := range v {
		v[i] = float32(float64(v[i]) / norm)
	}
	return v
}

func stressChunks(doc, salt uint64, n int) []float32 {
	var out []float32
	for c := 0; c < n; c++ {
		out = append(out, stressVec(doc, uint64(c), salt)...)
	}
	return out
}

func stressDocID(i uint64) string { return fmt.Sprintf("doc%06d", i) }

func stressTags(doc, salt uint64) []string {
	return []string{
		fmt.Sprintf("tag%d", (doc+salt)%7),
		fmt.Sprintf("grp%d", (doc+2*salt)%3),
	}
}

// refIndex is an independent sequential reference implementation used
// to verify the concurrent index after the dust settles.
type refIndex struct {
	docs map[string]*refDoc
	rev  uint64
}

type refDoc struct {
	tags map[string]bool
	vecs []float32
	rev  uint64
}

func newRefIndex() *refIndex { return &refIndex{docs: make(map[string]*refDoc)} }

func (r *refIndex) upsert(id string, tags []string, vecs []float32) uint64 {
	r.rev++
	ts := make(map[string]bool, len(tags))
	for _, t := range tags {
		ts[t] = true
	}
	r.docs[id] = &refDoc{tags: ts, vecs: vecs, rev: r.rev}
	return r.rev
}

func (r *refIndex) delete(id string) {
	if _, ok := r.docs[id]; !ok {
		return
	}
	r.rev++
	delete(r.docs, id)
}

// search mirrors the index ranking exactly: cosine >= 0.35, score desc,
// then (docID, chunk) asc, limit 20.
func (r *refIndex) search(q []float32, tag string) []Result {
	var out []Result
	for id, d := range r.docs {
		if !d.tags[tag] {
			continue
		}
		for c := 0; c < len(d.vecs)/Dim; c++ {
			var s float64
			for i := 0; i < Dim; i++ {
				s += float64(q[i]) * float64(d.vecs[c*Dim+i])
			}
			sc := float32(s)
			if sc >= minScore {
				out = append(out, Result{DocID: id, Chunk: c, Rev: d.rev, Score: sc})
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
	if len(out) > maxHits {
		out = out[:maxHits]
	}
	return out
}

// checkSearchInvariants validates properties every search result must
// satisfy regardless of the exact snapshot it read.
func checkSearchInvariants(t *testing.T, res []Result) {
	t.Helper()
	if len(res) > maxHits {
		t.Fatalf("more than %d results: %d", maxHits, len(res))
	}
	revByDoc := make(map[string]uint64)
	for i, r := range res {
		if r.Score < minScore {
			t.Fatalf("result %d below cutoff: %+v", i, r)
		}
		if prev, ok := revByDoc[r.DocID]; ok && prev != r.Rev {
			t.Fatalf("doc %s appears with revisions %d and %d in one result", r.DocID, prev, r.Rev)
		}
		revByDoc[r.DocID] = r.Rev
		if i > 0 {
			a, b := res[i-1], r
			if a.Score < b.Score ||
				(a.Score == b.Score && (a.DocID > b.DocID ||
					(a.DocID == b.DocID && a.Chunk >= b.Chunk))) {
				t.Fatalf("results out of order at %d: %+v then %+v", i, a, b)
			}
		}
	}
}

// TestRebuildUnderLoad runs 50k documents x 4 chunks through a rebuild
// that overlaps a stream of modifications and queries, then verifies
// the post-switch state against the sequential reference.
func TestRebuildUnderLoad(t *testing.T) {
	const (
		numDocs   = 50000
		numChunks = 4
		numOps    = 2000
		searchTag = "tag3"
	)
	query := stressVec(0xFFFFFFFFFFFFFFFF, 0, 0xDEADBEEF)

	// The first backfill hook call blocks until the writer has
	// completed half of its operations, guaranteeing overlap between
	// the backfill and the modification stream.
	halfway := make(chan struct{})
	var hooked atomic.Bool
	ix := New(WithHooks(Hooks{
		OnBackfillDoc: func(string, uint64) {
			if hooked.CompareAndSwap(false, true) {
				<-halfway
			}
		},
	}))
	ref := newRefIndex()

	// Initial load: 50k docs, 4 chunks each.
	for i := uint64(0); i < numDocs; i++ {
		tags := stressTags(i, 0)
		vec := stressChunks(i, 0, numChunks)
		if _, err := ix.Upsert(stressDocID(i), tags, vec); err != nil {
			t.Fatal(err)
		}
		ref.upsert(stressDocID(i), tags, vec)
	}

	// Concurrent searcher: hammers the index throughout the rebuild and
	// the modification stream, checking per-result invariants.
	stopSearch := make(chan struct{})
	searcherDone := make(chan struct{})
	go func() {
		defer close(searcherDone)
		for {
			select {
			case <-stopSearch:
				return
			default:
			}
			res, err := ix.Search(query, searchTag)
			if err != nil {
				t.Error(err)
				return
			}
			checkSearchInvariants(t, res)
		}
	}()

	rebuildDone := make(chan error, 1)
	go func() { rebuildDone <- ix.Rebuild() }()

	// Modification stream: upserts (varying chunk counts), deletes, and
	// ID reuse, applied to both the index and the reference.
	rng := rand.New(rand.NewSource(42))
	for op := uint64(1); op <= numOps; op++ {
		if op == numOps/2 {
			close(halfway)
		}
		id := stressDocID(rng.Uint64() % (numDocs + 500))
		switch rng.Intn(10) {
		case 0, 1:
			if err := ix.Delete(id); err != nil {
				t.Fatal(err)
			}
			ref.delete(id)
		default:
			tags := stressTags(rng.Uint64()%numDocs, op)
			vec := stressChunks(rng.Uint64(), op, 1+rng.Intn(6))
			if _, err := ix.Upsert(id, tags, vec); err != nil {
				t.Fatal(err)
			}
			ref.upsert(id, tags, vec)
		}
	}

	if err := <-rebuildDone; err != nil {
		t.Fatal(err)
	}
	close(stopSearch)
	<-searcherDone

	// After the switch, the index must match the sequential reference
	// exactly, and it must be served by the new generation.
	want := ref.search(query, searchTag)
	got := mustSearch(t, ix, query, searchTag)
	assertIdenticalResults(t, got, want)

	// A second, quiesced rebuild must not change anything either.
	if err := ix.Rebuild(); err != nil {
		t.Fatal(err)
	}
	got2 := mustSearch(t, ix, query, searchTag)
	assertIdenticalResults(t, got2, want)

	if ix.Stats().Docs != len(ref.docs) {
		t.Fatalf("doc count: index %d, reference %d", ix.Stats().Docs, len(ref.docs))
	}
}

func assertIdenticalResults(t *testing.T, got, want []Result) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("result count: got %d, want %d\ngot:  %v\nwant: %v",
			len(got), len(want), keysOf(got), keysOf(want))
	}
	for i := range got {
		if got[i] != want[i] {
			t.Fatalf("result %d differs:\ngot:  %+v\nwant: %+v", i, got[i], want[i])
		}
	}
}

// TestConcurrentLinearizabilitySpotCheck runs concurrent readers and a
// single writer without any rebuild, verifying that every search result
// is explainable by a single committed revision (no torn state).
func TestConcurrentLinearizabilitySpotCheck(t *testing.T) {
	ix := New()
	const docs = 200
	for i := 0; i < docs; i++ {
		mustUpsert(t, ix, fmt.Sprintf("d%03d", i), []string{"t"}, chunks(0.9, 0.8))
	}
	stop := make(chan struct{})
	var wg sync.WaitGroup
	for w := 0; w < 4; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for {
				select {
				case <-stop:
					return
				default:
				}
				res, err := ix.Search(queryVec(), "t")
				if err != nil {
					t.Error(err)
					return
				}
				checkSearchInvariants(t, res)
			}
		}()
	}
	for round := 0; round < 500; round++ {
		id := fmt.Sprintf("d%03d", round%docs)
		mustUpsert(t, ix, id, []string{"t"}, chunks(0.7, 0.6, 0.5))
		if round%3 == 0 {
			if err := ix.Delete(id); err != nil {
				t.Fatal(err)
			}
		}
	}
	close(stop)
	wg.Wait()
}
