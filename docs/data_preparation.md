# Data Preparation

## FoodX-251

Expected files:

```text
FoodX251/
|-- train/
|-- val/
|-- test_set/
|-- train_labels.csv
|-- val_labels.csv
`-- class_list.txt
```

The CSV files should contain at least:

```csv
img_name,label
example.jpg,0
```

The split is controlled by `train_labels.csv` and `val_labels.csv`. The
`test_set` directory is included only as an additional image lookup directory
when resolving file names; it does not define the training/evaluation split.

## VireoFood172

Expected ImageFolder layout:

```text
VireoFood172/
|-- train/
|   |-- class_000/
|   `-- ...
`-- val/
    |-- class_000/
    `-- ...
```

Alternatively, TXT list files can be used:

```text
train_list.txt
val_list.txt
```

Each line should contain:

```text
relative_image_path label
```
