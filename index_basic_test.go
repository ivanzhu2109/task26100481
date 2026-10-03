package vecindex

import (
	"fmt"
	"testing"
)

func TestUpsertReturnsIncreasingRevisions(t *testing.T) {
	ix := New()
	var prev uint64
	for i := 0; i < 10; i++ {
		rev := mustUpsert(t, ix, fmt.Sprintf("d%d", i), []string{"t"}, chunks(0.9))
		if rev <= prev {
			t.Fatalf("revision not strictly increasing: %d after %d", rev, prev)
		}
		prev = rev
	}
}

func TestSearchRankingFilteringAndRevisions(t *testing.T) {
	ix := New()
	ra := mustUpsert(t, ix, "a", []string{"x"}, chunks(0.9, 0.8))
	rb := mustUpsert(t, ix, "b", []string{"x"}, chunks(0.7))
	mustUpsert(t, ix, "c", []string{"y"}, chunks(0.95)) // wrong tag
	mustUpsert(t, ix, "d", []string{"x"}, chunks(0.34)) // below cutoff
	mustUpsert(t, ix, "e", []string{"x"}, chunks(0.36)) // above cutoff

	res := mustSearch(t, ix, queryVec(), "x")
	assertResults(t, res, []resultKey{
		{"a", 0, ra},
		{"a", 1, ra},
		{"b", 0, rb},
		{"e", 0, res[3].Rev},
	}, []float64{0.9, 0.8, 0.7, 0.36})
}

func TestUpsertAtomicallyReplacesTagsAndChunks(t *testing.T) {
	ix := New()
	mustUpsert(t, ix, "d", []string{"old"}, chunks(0.9, 0.85, 0.8))
	r2 := mustUpsert(t, ix, "d", []string{"new"}, chunks(0.88))

	if res := mustSearch(t, ix, queryVec(), "old"); len(res) != 0 {
		t.Fatalf("old tag still visible: %v", keysOf(res))
	}
	res := mustSearch(t, ix, queryVec(), "new")
	assertResults(t, res, []resultKey{{"d", 0, r2}}, []float64{0.88})
}

func TestDeleteThenReuseSameID(t *testing.T) {
	ix := New()
	mustUpsert(t, ix, "d", []string{"t"}, chunks(0.9, 0.8, 0.7))
	if err := ix.Delete("d"); err != nil {
		t.Fatal(err)
	}
	if res := mustSearch(t, ix, queryVec(), "t"); len(res) != 0 {
		t.Fatalf("deleted doc still visible: %v", keysOf(res))
	}
	// Reuse the same ID with a different chunk count.
	r2 := mustUpsert(t, ix, "d", []string{"t"}, chunks(0.6))
	res := mustSearch(t, ix, queryVec(), "t")
	assertResults(t, res, []resultKey{{"d", 0, r2}}, []float64{0.6})
}

func TestDeleteNonExistentIsNoOp(t *testing.T) {
	ix := New()
	if err := ix.Delete("nope"); err != nil {
		t.Fatal(err)
	}
}

func TestTop20LimitAndTieBreak(t *testing.T) {
	ix := New()
	for i := 0; i < 25; i++ {
		mustUpsert(t, ix, fmt.Sprintf("d%02d", i), []string{"t"}, chunks(0.9))
	}
	res := mustSearch(t, ix, queryVec(), "t")
	if len(res) != 20 {
		t.Fatalf("got %d results, want 20", len(res))
	}
	for i, r := range res {
		want := fmt.Sprintf("d%02d", i) // equal scores: docID asc
		if r.DocID != want || r.Chunk != 0 {
			t.Fatalf("result %d: got %s/%d, want %s/0", i, r.DocID, r.Chunk, want)
		}
	}
}

func TestSearchRejectsBadQuery(t *testing.T) {
	ix := New()
	if _, err := ix.Search(make([]float32, Dim+1), "t"); err != ErrBadVectorLen {
		t.Fatalf("got %v, want ErrBadVectorLen", err)
	}
}

func TestUpsertRejectsBadVectors(t *testing.T) {
	ix := New()
	if _, err := ix.Upsert("d", []string{"t"}, make([]float32, Dim+1)); err != ErrBadVectorLen {
		t.Fatalf("got %v, want ErrBadVectorLen", err)
	}
	if _, err := ix.Upsert("", []string{"t"}, chunks(0.9)); err != ErrEmptyDocID {
		t.Fatalf("got %v, want ErrEmptyDocID", err)
	}
}

func TestRebuildPreservesLogicalResults(t *testing.T) {
	ix := New()
	r1 := mustUpsert(t, ix, "a", []string{"t"}, chunks(0.9, 0.8))
	r2 := mustUpsert(t, ix, "b", []string{"t"}, chunks(0.7))
	mustUpsert(t, ix, "c", []string{"z"}, chunks(0.95))

	before := mustSearch(t, ix, queryVec(), "t")
	genBefore := ix.Stats().GenerationID
	if err := ix.Rebuild(); err != nil {
		t.Fatal(err)
	}
	genAfter := ix.Stats().GenerationID
	if genAfter == genBefore {
		t.Fatalf("generation id did not change: %d", genAfter)
	}
	after := mustSearch(t, ix, queryVec(), "t")
	if len(before) != len(after) {
		t.Fatalf("result count changed across rebuild: %d vs %d", len(before), len(after))
	}
	for i := range before {
		if before[i] != after[i] {
			t.Fatalf("result %d changed across rebuild: %+v vs %+v", i, before[i], after[i])
		}
	}
	assertResults(t, after, []resultKey{{"a", 0, r1}, {"a", 1, r1}, {"b", 0, r2}},
		[]float64{0.9, 0.8, 0.7})
}

func TestConcurrentRebuildsSerialize(t *testing.T) {
	ix := New()
	mustUpsert(t, ix, "d", []string{"t"}, chunks(0.9))
	start := ix.Stats().GenerationID
	done := make(chan struct{})
	for i := 0; i < 3; i++ {
		go func() {
			if err := ix.Rebuild(); err != nil {
				t.Error(err)
			}
			done <- struct{}{}
		}()
	}
	for i := 0; i < 3; i++ {
		<-done
	}
	if got := ix.Stats().GenerationID; got != start+3 {
		t.Fatalf("generation id: got %d, want %d", got, start+3)
	}
	if ix.Stats().Rebuilding {
		t.Fatal("Rebuilding still true after all rebuilds returned")
	}
}
