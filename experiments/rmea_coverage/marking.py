"""Unchanged marked_words from pinned source5f7860e...; no dataset reader."""
def marked_words(row):
    """Use pretrained punctuation tokens; retain a map to original word indices.

    Every relation candidate receives the same words and ordered entity spans.
    Labels/types/metadata never enter this construction. Markers are excluded
    from evidence pooling; the contextualized original words retain their effect.
    """
    words, original = [], []
    starts = {row['subject'][0]: '@', row['object'][0]: '#'}
    ends = {row['subject'][1]: '@', row['object'][1]: '#'}
    for index, token in enumerate(row['tokens']):
        if index in ends:
            words.append(ends[index]); original.append(None)
        if index in starts:
            words.append(starts[index]); original.append(None)
        words.append(token); original.append(index)
    if len(row['tokens']) in ends:
        words.append(ends[len(row['tokens'])]); original.append(None)
    return words, original
