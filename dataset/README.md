# Dataset folder

Place the six recording sessions here:

```
dataset/
├── lms_subject1_LA_downstairs_Nopeople_1/
├── lms_subject1_LA_upstairs_Nopeople_1/
├── lms_subject1_LB_upstairs_Nopeople_2/
├── lms_subject1_RA_downstairs_Nopeople_1/
├── lms_subject1_RA_upstairs_Nopeople_1/
├── lms_subject1_RB_downstairs_Nopeople_1/
└── splits/
```

Each session contains `sharpen_rgb/PNG` (model input) and `raw_depth/PNG`
(ground truth), plus `colored_depth/PNG`. Most sessions also provide
`smooth_depth/PNG` and `transformed_depth/PNG`, but
`lms_subject1_RB_downstairs_Nopeople_1` does not; in that session the three
folders it does have also contain an `image.gif` preview next to `PNG/`.

`splits/` holds one file per split (`all.txt`, `staircase.txt`,
`corridor.txt`), with one RGB–depth pair per line, relative to this folder.
