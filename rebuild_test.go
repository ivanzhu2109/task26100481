package vecindex

import (
	"sync"
	"testing"
)

// TestRebuildBackfillRace pauses the backfill right after it read the old
// revision of a document, then replaces tags+vectors, deletes, and
// re-inserts the same ID with a different chunk count. After the backfill
// resumes and the switch completes, only the final revision may be visible.
func TestRebuildBackfillRace(t *testing.T) {
	ix := New()
	rev1 := ix.Upsert("doc", []string{"tagOld"}, chunks(1, 1))
	ix.Upsert("other", []string{"tagOld"}, chunks(1))

	readOld := make(chan struct{})
	resume := make(chan struct{})
	var once sync.Once
	ix.hooks.onBackfillDoc = func(docID string, rev uint64) {
		if docID == "doc" && rev == rev1 {
			once.Do(func() {
				close(readOld)
				<-resume
			})
		}
	}

	genBefore := ix.CurrentGenerationID()
	rebuildDone := make(chan struct{})
	go func() {
		ix.Rebuild()
		close(rebuildDone)
	}()

	<-readOld // backfill holds the old revision, not yet installed

	// While the backfill is paused, writes and searches must still complete.
	rev2 := ix.Upsert("doc", []string{"tagMid"}, chunks(1, 1, 1)) // 3 chunks
	ix.Delete("doc")
	rev4 := ix.Upsert("doc", []string{"tagNew"}, chunks(1)) // 1 chunk
	if !(rev1 < rev2 && rev2 < rev4) {
		t.Fatalf("revisions: %d %d %d", rev1, rev2, rev4)
	}
	// The serving (old) generation is live: confirmed commits are visible
	// even while the backfill is paused.
	got := ix.Search(queryVec(), "tagNew")
	if len(got) != 1 || got[0].Rev != rev4 {
		t.Fatalf("pre-switch search: %v", got)
	}

	close(resume)
	<-rebuildDone

	if ix.CurrentGenerationID() == genBefore {
		t.Fatal("generation did not advance")
	}
	// Only the final revision is visible; old tags and old chunks are gone.
	got = ix.Search(queryVec(), "tagNew")
	if len(got) != 1 || got[0].DocID != "doc" || got[0].Chunk != 0 || got[0].Rev != rev4 {
		t.Fatalf("final state: %v", got)
	}
	for _, tag := range []string{"tagOld", "tagMid"} {
		// "other" still carries tagOld; "doc" must not appear.
		for _, r := range ix.Search(queryVec(), tag) {
			if r.DocID == "doc" {
				t.Fatalf("stale state visible under %s: %v", tag, r)
			}
		}
	}
	if n := len(ix.Search(queryVec(), "tagOld")); n != 1 {
		t.Fatalf("other doc lost: %d hits", n)
	}
}

// TestRebuildDiagnostics proves via the generation registry that searches
// before the switch ran on the old generation and searches after it run on
// the newly materialized one.
func TestRebuildDiagnostics(t *testing.T) {
	ix := New()
	ix.Upsert("a", []string{"t"}, chunks(1))
	gen1 := ix.CurrentGenerationID()
	ix.Search(queryVec(), "t")
	ix.Rebuild()
	gen2 := ix.CurrentGenerationID()
	if gen2 == gen1 {
		t.Fatal("no new generation")
	}
	ix.Search(queryVec(), "t")
	if n := ix.SearchesInGeneration(gen1); n != 1 {
		t.Fatalf("gen %d served %d searches, want 1", gen1, n)
	}
	if n := ix.SearchesInGeneration(gen2); n != 1 {
		t.Fatalf("gen %d served %d searches, want 1", gen2, n)
	}
}
