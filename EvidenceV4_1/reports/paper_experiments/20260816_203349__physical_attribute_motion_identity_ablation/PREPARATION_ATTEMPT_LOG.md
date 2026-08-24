# A1 preparation attempt log

This directory is append-only. No accepted artifact or formal overlay was modified.

- overlay_a1 was pre-created empty by directory setup and is nonauthoritative.
- overlay_a1_frozen was a copied formal overlay left unmodified when the CPU constructor rejected a whitespace-sensitive visibility-template match. It is nonauthoritative.
- overlay_a1_frozen_r2 was likewise copied and left unmodified when that template check rejected the same formatting mismatch. It is nonauthoritative.
- overlay_a1_frozen_r3 is the only authoritative A1 ablation overlay. Its constructor completed successfully and its trajectory SHA256 is 7d7755f14a544d6acc10bfe32efa95060fb2544347e9f1276fcb3c8379dea63b.

All new SC/SS inputs, manifests, launch scripts, and reports bind exclusively to overlay_a1_frozen_r3 where an A1 overlay is needed. SC/SS themselves bind to the byte-identical accepted formal overlay.
