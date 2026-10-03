package vecindex

// backfillBatch is the number of records read from the authoritative store
// per lock acquisition during a rebuild.
const backfillBatch = 256

// Rebuild constructs an independent new retrieval generation from the
// authoritative document records and atomically switches to it, returning
// after the switch. At most one rebuild runs at a time; concurrent callers
// wait for the in-flight rebuild and return without starting another.
//
// The store lock is never held across the full traversal: records are read
// in small batches, and every commit during the rebuild is also applied to
// the new generation through the delta path, so no confirmed modification
// is lost at the switch.
func (ix *Index) Rebuild() {
	ix.mu.Lock()
	if ix.rebuilding {
		done := ix.rebuildDone
		ix.mu.Unlock()
		<-done
		return
	}
	ix.rebuilding = true
	ix.rebuildDone = make(chan struct{})
	ng := newGeneration(ix.serving.Load().id + 1)
	ix.building = ng
	ix.gens[ng.id] = ng
	cursor := ix.head
	ix.mu.Unlock()

	type item struct {
		id  string
		rec *record
	}
	for {
		ix.mu.Lock()
		batch := make([]item, 0, backfillBatch)
		for cursor != nil && len(batch) < backfillBatch {
			batch = append(batch, item{id: cursor.id, rec: cursor.rec})
			cursor = cursor.next
		}
		ix.mu.Unlock()
		if len(batch) == 0 {
			break
		}
		for _, it := range batch {
			if ix.hooks.onBackfillDoc != nil {
				ix.hooks.onBackfillDoc(it.id, it.rec.rev)
			}
			ng.backfill(it.id, it.rec)
		}
	}

	ix.mu.Lock()
	ix.serving.Store(ng)
	ix.building = nil
	ix.rebuilding = false
	close(ix.rebuildDone)
	ix.mu.Unlock()
}

// CurrentGenerationID reports which generation currently serves searches.
func (ix *Index) CurrentGenerationID() uint64 {
	return ix.serving.Load().id
}

// SearchesInGeneration reports how many searches selected the given
// generation as their view. Test diagnostic proving which materialized
// generation served a query.
func (ix *Index) SearchesInGeneration(id uint64) uint64 {
	ix.mu.Lock()
	g, ok := ix.gens[id]
	ix.mu.Unlock()
	if !ok {
		return 0
	}
	return g.searches.Load()
}
