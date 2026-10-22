# Dataset folder

Place the six recording sessions here:

```
dataset/
├── lms_kamal_LA_downstairs_Nopeople_1/
├── lms_kamal_LA_upstairs_Nopeople_1/
├── lms_kamal_LB_upstairs_Nopeople_2/
├── lms_kamal_RA_downstairs_Nopeople_1/
├── lms_kamal_RA_upstairs_Nopeople_1/
├── lms_kamal_RB_downstairs_Nopeople_1/
└── splits/
```

Each session contains `sharpen_rgb/PNG`, `raw_depth/PNG`, `colored_depth/PNG`,
`smooth_depth/PNG` and `transformed_depth/PNG`.

`splits/` holds one file per split (`all.txt`, `staircase.txt`,
`corridor.txt`), with one RGB–depth pair per line, relative to this folder.
