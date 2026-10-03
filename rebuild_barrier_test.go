package vecindex

import (
	"sync"
	"sync/atomic"
	"testing"
)

// TestRebuildBackfillBarrier pauses the rebuild backfill after it has
// read the old revision of a document but before it writes the new
// generation, then performs a replace, a delete, and an ID reuse with a
// different chunk count. After the backfill resumes and the swap
// completes, only the final revision may be visible.
func TestRebuildBackfillBarrier(t *testing.T) {
	backfillRead := make(chan struct{})
	releaseBackfill := make(chan struct{})
	var paused atomic.Bool

	hooks := Hooks{
		OnBackfillDoc: func(docID string, rev uint64) {
			if docID == "docX" && paused.CompareAndSwap(false, true) {
				close(backfillRead)
				<-releaseBackfill
			}
		},
	}
	ix := New(WithHooks(hooks))

	// docX starts with tag "old" and 2 chunks; filler docs exist too.
	mustUpsert(t, ix, "docX", []string{"old"}, chunks(0.9, 0.8))
	mustUpsert(t, ix, "filler", []string{"old", "new"}, chunks(0.7))
	// docY will be purely deleted during the paused backfill; its old
	// revision must not be resurrected by the stale backfill.
	mustUpsert(t, ix, "docY", []string{"new"}, chunks(0.85))

	genBefore := ix.Stats().GenerationID
	rebuildDone := make(chan error, 1)
	go func() { rebuildDone <- ix.Rebuild() }()

	// Backfill is now paused after reading docX's old revision, before
	// writing it into the new generation.
	<-backfillRead

	// All of these must complete while the backfill is paused.
	rMid := mustUpsert(t, ix, "docX", []string{"mid"}, chunks(0.6, 0.5, 0.4))
	if err := ix.Delete("docX"); err != nil {
		t.Fatal(err)
	}
	rFinal := mustUpsert(t, ix, "docX", []string{"new"}, chunks(0.95))
	_ = rMid
	// Pure delete: docY must stay gone after the swap.
	if err := ix.Delete("docY"); err != nil {
		t.Fatal(err)
	}
	// docZ is created during the rebuild; it is not in the backfill
	// snapshot and must survive the swap via the live dual-write.
	rZ := mustUpsert(t, ix, "docZ", []string{"new"}, chunks(0.65))
	// Searches must also complete during the paused rebuild, and must
	// already reflect the committed modifications.
	res := mustSearch(t, ix, queryVec(), "new")
	assertResults(t, res, []resultKey{
		{"docX", 0, rFinal},
		{"filler", 0, res[1].Rev},
		{"docZ", 0, rZ},
	}, []float64{0.95, 0.7, 0.65})

	close(releaseBackfill)
	if err := <-rebuildDone; err != nil {
		t.Fatal(err)
	}

	genAfter := ix.Stats().GenerationID
	if genAfter == genBefore {
		t.Fatal("generation id did not change after rebuild")
	}

	// Only the final revision may be visible on the new generation.
	res = mustSearch(t, ix, queryVec(), "new")
	assertResults(t, res, []resultKey{
		{"docX", 0, rFinal},
		{"filler", 0, res[1].Rev},
		{"docZ", 0, rZ},
	}, []float64{0.95, 0.7, 0.65})
	if res[0].Rev != rFinal {
		t.Fatalf("docX rev: got %d, want %d (stale backfill won)", res[0].Rev, rFinal)
	}
	for _, tag := range []string{"old", "mid"} {
		for _, r := range mustSearch(t, ix, queryVec(), tag) {
			if r.DocID == "docX" {
				t.Fatalf("docX visible under stale tag %q after rebuild: %+v", tag, r)
			}
		}
	}
	// docX has exactly one chunk now; chunk 1 must not exist.
	foundZ := false
	for _, r := range mustSearch(t, ix, queryVec(), "new") {
		if r.DocID == "docX" && r.Chunk != 0 {
			t.Fatalf("stale chunk survived: %+v", r)
		}
		if r.DocID == "docY" {
			t.Fatalf("deleted docY resurrected by backfill: %+v", r)
		}
		if r.DocID == "docZ" {
			foundZ = true
			if r.Rev != rZ {
				t.Fatalf("docZ rev: got %d, want %d", r.Rev, rZ)
			}
		}
	}
	if !foundZ {
		t.Fatal("docZ created during rebuild is missing after the swap")
	}
}

// TestSearchViewBarrier pauses a search after it has selected its view
// but before scoring. A document replace, a delete, and a full rebuild
// swap all complete meanwhile. The paused search must still return its
// selected view's complete results; a new search must see the latest
// state on the new generation.
func TestSearchViewBarrier(t *testing.T) {
	viewSelected := make(chan struct{})
	releaseSearch := make(chan struct{})
	var paused atomic.Bool

	var mu sync.Mutex
	var searchGens []uint64
	hooks := Hooks{
		OnSearchViewSelected: func(genID uint64) {
			mu.Lock()
			searchGens = append(searchGens, genID)
			mu.Unlock()
			if paused.CompareAndSwap(false, true) {
				close(viewSelected)
				<-releaseSearch
			}
		},
	}
	ix := New(WithHooks(hooks))

	rA := mustUpsert(t, ix, "docA", []string{"t"}, chunks(0.9, 0.8))
	rB := mustUpsert(t, ix, "docB", []string{"t"}, chunks(0.7))

	type searchOut struct {
		res []Result
		err error
	}
	oldCh := make(chan searchOut, 1)
	go func() {
		res, err := ix.Search(queryVec(), "t")
		oldCh <- searchOut{res, err}
	}()

	// The old search has pinned its view; everything below must still
	// be able to complete.
	<-viewSelected
	rA2 := mustUpsert(t, ix, "docA", []string{"t"}, chunks(0.6))
	if err := ix.Delete("docB"); err != nil {
		t.Fatal(err)
	}
	rC := mustUpsert(t, ix, "docC", []string{"t"}, chunks(0.95))
	if err := ix.Rebuild(); err != nil {
		t.Fatal(err)
	}
	genAfterSwap := ix.Stats().GenerationID

	close(releaseSearch)
	old := <-oldCh
	if old.err != nil {
		t.Fatal(old.err)
	}

	// The paused search returns its selected view: docA's original two
	// chunks and docB, nothing from later commits.
	assertResults(t, old.res, []resultKey{
		{"docA", 0, rA},
		{"docA", 1, rA},
		{"docB", 0, rB},
	}, []float64{0.9, 0.8, 0.7})

	// A new search sees the latest state on the new generation.
	fresh := mustSearch(t, ix, queryVec(), "t")
	assertResults(t, fresh, []resultKey{
		{"docC", 0, rC},
		{"docA", 0, rA2},
	}, []float64{0.95, 0.6})

	// Diagnostics: the old search ran on the generation that was active
	// before the swap, the new search on the swapped-in generation.
	mu.Lock()
	defer mu.Unlock()
	if len(searchGens) != 2 {
		t.Fatalf("expected 2 search view selections, got %v", searchGens)
	}
	if searchGens[0] == searchGens[1] {
		t.Fatalf("both searches used generation %d; want the fresh search on the new one", searchGens[0])
	}
	if searchGens[1] != genAfterSwap {
		t.Fatalf("fresh search used generation %d, Stats reports %d", searchGens[1], genAfterSwap)
	}
}
