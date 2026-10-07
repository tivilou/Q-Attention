"""Explicit source-adapter dispatch; unavailable cohort entries never fall back."""
def build(row, source, checkpoint, *, device, checkpointing=False):
    if row["source"] == "improved":
        from .improved import Improved
        return Improved(source, checkpoint, row["input_format"], device=device, checkpointing=checkpointing)
    if row["id"] == "ptr":
        from .ptr import PTR
        return PTR(source, checkpoint, device=device, checkpointing=checkpointing)
    raise ValueError("no implemented adapter: " + row["id"])


def metadata_identity(row):
    if row["id"] == "ptr":
        from .ptr_resources import FILES, REVISION
        return {"revision": REVISION, "files": FILES}
    return {}
