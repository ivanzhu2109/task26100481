package vecindex

import (
	"hash/fnv"
	"sync"
	"sync/atomic"
)

const numShards = 64

// docEntry is the per-document head of the immutable version chain.
type docEntry struct {
	head atomic.Pointer[docVer]
}

type shard struct {
	mu sync.RWMutex
	m  map[string]*docEntry
}

// generation is one materialized retrieval generation. Documents are
// sharded; every committed state is an immutable docVer, so readers never
// hold locks while scoring and a paused search blocks nobody.
type generation struct {
	id       uint64
	shards   [numShards]shard
	searches atomic.Uint64
}

func newGeneration(id uint64) *generation {
	g := &generation{id: id}
	for i := range g.shards {
		g.shards[i].m = make(map[string]*docEntry)
	}
	return g
}

func (g *generation) shardOf(docID string) *shard {
	h := fnv.New64a()
	h.Write([]byte(docID))
	return &g.shards[h.Sum64()%numShards]
}

// applyDelta pushes a new committed version (upsert or delete tombstone)
// for docID. Called from the store commit path, serialized by ix.mu.
func (g *generation) applyDelta(docID string, v *docVer) {
	s := g.shardOf(docID)
	s.mu.Lock()
	e, ok := s.m[docID]
	if !ok {
		e = &docEntry{}
		s.m[docID] = e
	}
	v.prev = e.head.Load()
	e.head.Store(v)
	s.mu.Unlock()
}

// backfill installs a record read from the authoritative store, unless a
// newer or equal revision already reached this generation via the delta
// path. Returns true if installed.
func (g *generation) backfill(docID string, rec *record) bool {
	s := g.shardOf(docID)
	s.mu.Lock()
	e, ok := s.m[docID]
	if ok {
		if head := e.head.Load(); head != nil && head.rev >= rec.rev {
			s.mu.Unlock()
			return false
		}
	} else {
		e = &docEntry{}
		s.m[docID] = e
	}
	v := &docVer{seq: rec.rev, rev: rec.rev, tags: rec.tags, chunks: rec.chunks}
	v.prev = e.head.Load()
	e.head.Store(v)
	s.mu.Unlock()
	return true
}

// viewDoc resolves the document state visible at commit watermark wm.
func viewOf(head *docVer, wm uint64) *docVer {
	for v := head; v != nil; v = v.prev {
		if v.seq <= wm {
			return v
		}
	}
	return nil
}

// entrySnapshot is a lock-free handle to one document's version chain.
type entrySnapshot struct {
	docID string
	head  *docVer
}

// snapshotEntries copies (docID, head) pairs shard by shard. Each shard
// lock is held only for the pointer copy, never during scoring.
func (g *generation) snapshotEntries() []entrySnapshot {
	out := make([]entrySnapshot, 0, 1024)
	for i := range g.shards {
		s := &g.shards[i]
		s.mu.RLock()
		for id, e := range s.m {
			out = append(out, entrySnapshot{docID: id, head: e.head.Load()})
		}
		s.mu.RUnlock()
	}
	return out
}
