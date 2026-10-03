package vecindex

import "sync/atomic"

// verNode is one immutable committed version of a document. Nodes are
// linked newest-first; a node is never mutated after publication, so
// readers can walk chains without holding locks. Tombstones mark deletes.
type verNode struct {
	rev       uint64
	tags      tagSet    // nil for tombstone
	vec       []float32 // nil for tombstone; shared, immutable
	tombstone bool
	next      *verNode
}

// versionCell is the per-document head of a version chain.
type versionCell struct {
	head atomic.Pointer[verNode]
}

// prepend publishes n at the head of the chain. Callers must guarantee
// n.rev is greater than every rev already in the chain (true for live
// updates, which are serialized by Index.mu).
func (c *versionCell) prepend(n *verNode) {
	for {
		h := c.head.Load()
		n.next = h
		if c.head.CompareAndSwap(h, n) {
			return
		}
	}
}

// prependIfNewer publishes n only if the current head is older than n.
// Used by rebuild backfill, which races with newer live updates and must
// never overwrite them. Returns true if n was installed.
func (c *versionCell) prependIfNewer(n *verNode) bool {
	for {
		h := c.head.Load()
		if h != nil && h.rev >= n.rev {
			return false
		}
		n.next = h
		if c.head.CompareAndSwap(h, n) {
			return true
		}
	}
}

// asOf returns the newest node with rev <= r, or nil.
func (c *versionCell) asOf(r uint64) *verNode {
	for n := c.head.Load(); n != nil; n = n.next {
		if n.rev <= r {
			return n
		}
	}
	return nil
}

// prune drops every node older than the newest node with rev <= floor.
// Any reader with rev >= floor resolves to that node or a newer one, so
// the dropped nodes are unreachable. If no node has rev <= floor nothing
// is safe to drop. floor == 0 disables pruning. Callers must serialize
// pruning with prepends (Index.mu).
func (c *versionCell) prune(floor uint64) {
	if floor == 0 {
		return
	}
	for n := c.head.Load(); n != nil; n = n.next {
		if n.rev <= floor {
			n.next = nil
			return
		}
	}
}
