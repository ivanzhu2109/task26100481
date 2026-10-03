package vecindex

// Dim is the fixed dimensionality of all vectors in the index.
const Dim = 32

// Result is a single search hit: one chunk of one document.
type Result struct {
	DocID string
	Chunk int
	Rev   uint64
	Score float32
}

// Stats exposes diagnostic information about the index.
type Stats struct {
	// GenerationID identifies the currently active search generation.
	GenerationID uint64
	// Rebuilding reports whether a Rebuild is currently in progress.
	Rebuilding bool
	// Docs is the number of documents in the authoritative store.
	Docs int
}

// Hooks is a test-only diagnostic surface. All callbacks are optional.
// The zero value means "no hooks".
//
// The hooks are invoked synchronously by the index; implementations must
// not call back into the index from the same goroutine.
type Hooks struct {
	// OnBackfillDoc is invoked during Rebuild after the backfill has read
	// the authoritative record for docID at revision rev, but before the
	// corresponding entry is written into the new generation.
	OnBackfillDoc func(docID string, rev uint64)

	// OnSearchViewSelected is invoked by Search after it has selected its
	// immutable generation view, but before scoring begins.
	OnSearchViewSelected func(generationID uint64)
}

// Option configures an Index.
type Option func(*Index)

// WithHooks installs diagnostic hooks (test-only).
func WithHooks(h Hooks) Option {
	return func(ix *Index) { ix.hooks = h }
}
