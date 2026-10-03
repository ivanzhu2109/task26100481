package vecindex

// Test hooks. Nil in production use.
type hooks struct {
	// onBackfillDoc is called after a record has been read from the
	// authoritative store, before it is installed into the new generation.
	onBackfillDoc func(docID string, rev uint64)
	// onSearchView is called after a search selected its view
	// (generation + watermark), before scoring begins.
	onSearchView func(genID uint64)
}
