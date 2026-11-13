The full suite of data used to make these figures is too large to store directly on GitHub. It is stored on the Hugging Face dataset [zaneblood1/cmb_lensing_data](https://huggingface.co/datasets/zaneblood1/cmb_lensing_data), in the `map_joint_comparison/` folder (~13 GB). To download it:

```bash
hf download zaneblood1/cmb_lensing_data --repo-type dataset \
    --include "map_joint_comparison/*" --local-dir cmb_lensing_data
```

The dataset card describes the layout of the files.
