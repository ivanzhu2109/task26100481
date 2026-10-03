package vecindex

import (
	"math"
	"testing"
)

// unitVec returns a 32-dim unit vector whose first component is s (its
// cosine score against the query [1,0,...]).
func unitVec(s float64) []float32 {
	v := make([]float32, Dim)
	v[0] = float32(s)
	v[1] = float32(math.Sqrt(1 - s*s))
	return v
}

func queryVec() []float32 {
	q := make([]float32, Dim)
	q[0] = 1
	return q
}

func chunks(scores ...float64) [][]float32 {
	out := make([][]float32, len(scores))
	for i, s := range scores {
		out[i] = unitVec(s)
	}
	return out
}

func TestUpsertSearchBasic(t *testing.T) {
	ix := New()
	rev1 := ix.Upsert("a", []string{"t"}, chunks(0.9, 0.8))
	rev2 := ix.Upsert("b", []string{"t"}, chunks(0.9))
	if rev2 <= rev1 {
		t.Fatalf("revisions not increasing: %d then %d", rev1, rev2)
	}
	res := ix.Search(queryVec(), "t")
	if len(res) != 3 {
		t.Fatalf("want 3 hits, got %d: %v", len(res), res)
	}
	// score desc; tie broken by docID asc
	want := []SearchResult{
		{DocID: "a", Chunk: 0, Rev: rev1, Score: 0.9},
		{DocID: "b", Chunk: 0, Rev: rev2, Score: 0.9},
		{DocID: "a", Chunk: 1, Rev: rev1, Score: 0.8},
	}
	for i, w := range want {
		got := res[i]
		if got.DocID != w.DocID || got.Chunk != w.Chunk || got.Rev != w.Rev {
			t.Fatalf("hit %d: want %+v got %+v", i, w, got)
		}
		if math.Abs(float64(got.Score-w.Score)) > 1e-5 {
			t.Fatalf("hit %d: want score %v got %v", i, w.Score, got.Score)
		}
	}
}

func TestTagFiltering(t *testing.T) {
	ix := New()
	ix.Upsert("a", []string{"x"}, chunks(1))
	ix.Upsert("b", []string{"y"}, chunks(1))
	if got := ix.Search(queryVec(), "x"); len(got) != 1 || got[0].DocID != "a" {
		t.Fatalf("tag x: %v", got)
	}
	if got := ix.Search(queryVec(), "z"); len(got) != 0 {
		t.Fatalf("tag z: %v", got)
	}
}

func TestThreshold(t *testing.T) {
	ix := New()
	ix.Upsert("a", []string{"t"}, chunks(0.34, 0.36))
	res := ix.Search(queryVec(), "t")
	if len(res) != 1 || res[0].Chunk != 1 {
		t.Fatalf("threshold: %v", res)
	}
}

func TestMaxResults(t *testing.T) {
	ix := New()
	var vecs [][]float32
	for i := 0; i < 25; i++ {
		vecs = append(vecs, unitVec(1))
	}
	ix.Upsert("a", []string{"t"}, vecs)
	res := ix.Search(queryVec(), "t")
	if len(res) != MaxResults {
		t.Fatalf("want %d hits, got %d", MaxResults, len(res))
	}
	for i, r := range res {
		if r.Chunk != i {
			t.Fatalf("hit %d: chunk %d", i, r.Chunk)
		}
	}
}

func TestUpsertReplacesAtomically(t *testing.T) {
	ix := New()
	ix.Upsert("a", []string{"old"}, chunks(1, 1, 1))
	rev := ix.Upsert("a", []string{"new"}, chunks(0.5))
	if got := ix.Search(queryVec(), "old"); len(got) != 0 {
		t.Fatalf("old tags still visible: %v", got)
	}
	got := ix.Search(queryVec(), "new")
	if len(got) != 1 || got[0].Rev != rev {
		t.Fatalf("new state: %v", got)
	}
}

func TestDeleteAndReinsert(t *testing.T) {
	ix := New()
	ix.Upsert("a", []string{"t"}, chunks(1, 1, 1))
	ix.Delete("a")
	if got := ix.Search(queryVec(), "t"); len(got) != 0 {
		t.Fatalf("deleted doc visible: %v", got)
	}
	// re-insert with fewer chunks: no old chunks inherited
	rev := ix.Upsert("a", []string{"t"}, chunks(1))
	got := ix.Search(queryVec(), "t")
	if len(got) != 1 || got[0].Rev != rev {
		t.Fatalf("reinsert: %v", got)
	}
}

func TestDeleteMissingIsNoop(t *testing.T) {
	ix := New()
	ix.Delete("ghost")
	rev := ix.Upsert("a", []string{"t"}, chunks(1))
	if rev != 1 {
		t.Fatalf("noop delete consumed a revision: rev=%d", rev)
	}
}

func TestRebuildEmptyAndIdempotent(t *testing.T) {
	ix := New()
	ix.Rebuild()
	ix.Upsert("a", []string{"t"}, chunks(1))
	before := ix.Search(queryVec(), "t")
	ix.Rebuild()
	after := ix.Search(queryVec(), "t")
	if len(before) != 1 || len(after) != 1 || before[0] != after[0] {
		t.Fatalf("rebuild changed results: %v vs %v", before, after)
	}
	if ix.CurrentGenerationID() != 3 {
		t.Fatalf("generation id: %d", ix.CurrentGenerationID())
	}
}
