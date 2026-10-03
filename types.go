package vecindex

// Dim is the fixed vector dimensionality.
const Dim = 32

// ScoreThreshold is the minimum cosine score for a chunk to be returned.
const ScoreThreshold = 0.35

// MaxResults is the maximum number of chunks returned by one Search.
const MaxResults = 20

// SearchResult is one scored chunk hit.
type SearchResult struct {
	DocID string
	Chunk int
	Rev   uint64
	Score float32
}

// record is the authoritative committed state of one document.
// Records are immutable once committed; an upsert installs a fresh record.
type record struct {
	rev    uint64
	tags   map[string]struct{}
	chunks [][]float32
}

// docNode links a document into the insertion-ordered list used by
// Rebuild to walk the store in small batches. A node's next pointer is
// left intact when the node is unlinked, so an in-flight walk keeps a
// valid cursor.
type docNode struct {
	id         string
	rec        *record
	prev, next *docNode
}

// docVer is one immutable committed version of a document inside a
// retrieval generation. Versions form a chain through prev, ordered by
// decreasing commit sequence.
type docVer struct {
	seq     uint64
	rev     uint64
	tags    map[string]struct{}
	chunks  [][]float32
	deleted bool
	prev    *docVer
}
