package vecindex

import (
	"math"
	"testing"
)

// queryVec is the fixed query used by most tests: the unit vector along
// dimension 0. A document chunk built with vecWithScore(s, k) then has
// cosine score exactly s against it.
func queryVec() []float32 {
	q := make([]float32, Dim)
	q[0] = 1
	return q
}

// vecWithScore returns a Dim-dim unit vector whose dot product with
// queryVec() equals score. varyDim (must be in [1, Dim)) carries the
// remaining energy so different chunks are not identical vectors.
func vecWithScore(score float64, varyDim int) []float32 {
	if varyDim < 1 || varyDim >= Dim {
		panic("varyDim out of range")
	}
	v := make([]float32, Dim)
	v[0] = float32(score)
	v[varyDim] = float32(math.Sqrt(1 - score*score))
	return v
}

// chunks builds a document vector payload with one chunk per score.
func chunks(scores ...float64) []float32 {
	var out []float32
	for i, s := range scores {
		out = append(out, vecWithScore(s, i%(Dim-1)+1)...)
	}
	return out
}

// resultKey is the identity of a hit, used for order-sensitive
// comparisons in tests.
type resultKey struct {
	docID string
	chunk int
	rev   uint64
}

func keysOf(res []Result) []resultKey {
	out := make([]resultKey, len(res))
	for i, r := range res {
		out[i] = resultKey{r.DocID, r.Chunk, r.Rev}
	}
	return out
}

// assertResults checks that got matches the expected (docID, chunk, rev)
// sequence exactly, and that scores are within epsilon of expScores.
func assertResults(t *testing.T, got []Result, exp []resultKey, expScores []float64) {
	t.Helper()
	if len(got) != len(exp) {
		t.Fatalf("result count: got %d (%v), want %d (%v)", len(got), keysOf(got), len(exp), exp)
	}
	for i := range got {
		k := resultKey{got[i].DocID, got[i].Chunk, got[i].Rev}
		if k != exp[i] {
			t.Fatalf("result %d: got %+v, want %+v (all got: %v)", i, k, exp[i], keysOf(got))
		}
		if math.Abs(float64(got[i].Score)-expScores[i]) > 1e-5 {
			t.Fatalf("result %d score: got %v, want %v", i, got[i].Score, expScores[i])
		}
	}
}

func mustUpsert(t *testing.T, ix *Index, docID string, tags []string, vec []float32) uint64 {
	t.Helper()
	rev, err := ix.Upsert(docID, tags, vec)
	if err != nil {
		t.Fatalf("Upsert(%s): %v", docID, err)
	}
	return rev
}

func mustSearch(t *testing.T, ix *Index, q []float32, tag string) []Result {
	t.Helper()
	res, err := ix.Search(q, tag)
	if err != nil {
		t.Fatalf("Search(%s): %v", tag, err)
	}
	return res
}
