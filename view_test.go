package vecindex

import (
	"sync"
	"testing"
)

// TestSearchPinnedView pauses a search after it selected its view but
// before scoring. A document replace, a delete, and a full rebuild switch
// all complete meanwhile. The paused search must still return the complete
// result of its selected view; a fresh search must see the latest state.
func TestSearchPinnedView(t *testing.T) {
	ix := New()
	revA1 := ix.Upsert("a", []string{"t"}, chunks(1, 0.9))
	revB1 := ix.Upsert("b", []string{"t"}, chunks(0.8))

	viewSelected := make(chan struct{})
	resume := make(chan struct{})
	var once sync.Once
	ix.hooks.onSearchView = func(genID uint64) {
		once.Do(func() {
			close(viewSelected)
			<-resume
		})
	}

	type searchOut struct {
		res []SearchResult
	}
	oldResultCh := make(chan searchOut, 1)
	go func() {
		oldResultCh <- searchOut{res: ix.Search(queryVec(), "t")}
	}()

	<-viewSelected // view pinned, scoring not started

	// All of these must complete while the old search is paused.
	ix.Upsert("a", []string{"t"}, chunks(-1)) // replaced: no longer matches
	ix.Delete("b")
	ix.Rebuild()
	genAfter := ix.CurrentGenerationID()

	close(resume)
	oldRes := (<-oldResultCh).res

	// The paused search returns its pinned view in full: a's two old chunks
	// at revA1 and b's chunk at revB1, even though b is deleted and the
	// serving generation has switched.
	if len(oldRes) != 3 {
		t.Fatalf("pinned view: want 3 hits, got %v", oldRes)
	}
	seen := map[string]SearchResult{}
	for _, r := range oldRes {
		seen[r.DocID] = r
	}
	if seen["a"].Rev != revA1 || seen["b"].Rev != revB1 {
		t.Fatalf("pinned view revisions: %v", oldRes)
	}

	// A fresh search sees the latest committed state on the new generation.
	newRes := ix.Search(queryVec(), "t")
	if len(newRes) != 0 {
		t.Fatalf("latest state: %v", newRes)
	}
	if ix.SearchesInGeneration(genAfter) != 1 {
		t.Fatalf("new search did not use generation %d", genAfter)
	}
}
