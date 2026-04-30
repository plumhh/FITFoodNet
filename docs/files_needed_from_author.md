# Release Checklist

The core code has been organized into the open-source layout. Before publishing
to GitHub, confirm the following items.

## Must Not Commit

- Raw FoodX-251 or VireoFood172 images.
- DINOv3 pretrained weights.
- Local absolute paths from the training machine.
- Private logs containing local paths.
- Large `.pth`, `.pt`, or `.ckpt` files unless intentionally released through
  GitHub Releases or another model-hosting service.

## Optional Release Assets

- Example FoodX-251 `train_labels.csv` / `val_labels.csv` format with a few
  dummy rows.
- Example VireoFood172 `train_list.txt` / `val_list.txt` format with a few
  dummy rows.
- Final command lines used for the reported FoodX-251 and VireoFood172 results.
- Full fine-tuning learning-rate candidates tested for Table 2.
- Public checkpoint links if you decide to release trained weights separately.

## Paper-Aligned Assumptions

- Main training uses total physical batch size 64 with three-GPU DataParallel.
- `batch_size` in released scripts/configs means total batch size, not per-GPU
  batch size.
- Reported FoodX-251 and VireoFood172 metrics are validation/evaluation split
  metrics, not hidden test-set metrics.
