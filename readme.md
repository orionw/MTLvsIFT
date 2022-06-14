![Python application](https://github.com/orionw/TransferPrediction/workflows/transferprediction/badge.svg?branch=master)

[When to Use Multi-Task Learning vs Intermediate Fine-Tuning for Pre-Trained Encoder Transfer Learning](https://aclanthology.org/2022.acl-short.30/)
========
A repository for understanding when to use different transfer learning method for NLP

To reproduce the results, follow these instructions. Note that this repository was started in 2020 and contains many out of date versions of NLP packages, such as a very old version of Huggingface Transformers. Future work should likely move to more up-to-date transfer learning frameworks in order to include newer models and for ease of use.


Setup
-----
0. Create the version of python you will use (e.g. `python3 -m venv env` and `source env/bin/activate`)
1. Run the install: `./bin/install`
2. Verify your installation with `./bin/verify`
3. Gather the GLUE data by running `python ./bin/get_glue.py --data_dir <PATH>`.  This will download (most of) the GLUE data, due to legal requirements.


Get Results on Data
-------------------
0. Generate the models and scores: 
- Use `./bin/glue/single_glue` to generate the single task GLUE models and scores (used for MTL dynamic sampling and for STILTs)
- Run `./bin/batch_glue_pairs_by_seed.sh` to generate the scores for MTL, replacing the path for the single task scores.
- Use `./bin/batch_glue_pairs_by_seed.sh _all` for MTL-ALL.
- Use `./bin/glue/transfer_glue` after adding the single-task model checkpoints to train on the target task for STILTs.
1. Gather the scores needed after running the models
- For single-task scores use `./bin/create_matrices --dir_path <PATH_TO_SINGLE> --output_dir <OUTPUT_DIR> --single --use_seed`
- For MTL and STILTs, use `./bin/create_matrices --dir_path <PATH_TO_SCORES> --output_dir <OUTPUT_PATH> --use_seed --single_data_path <PATH_TO_SINGLE>` to generate the matrices for these over the different random seeds
- For MTL-ALL use `python3 ./bin/summarize_mtl_all_results.py --data_dir <DATA_DIR> --output_dir <OUTPUT_DIR>`
2. Compare the models to make Figure 1 in the paper: `./bin/compare_matrices --mtl_matrix MTL_score_matrix.csv --inter_matrix STILTS_score_matrix.csv --output_dir <OUTPUT_DIR>`

For the validation size experiment, see files in `./bin/glue/*validation*` which follow the same pattern as above, but with "validation" in the name.


Limitations
-----------
Since this code base was written in 2020, it is possible that these packages and dependencies may have problems in the future. Unfortunately, this repository will not be updated to current versions. However, please reach out or file an issue if you have any problems with reproducing this work using the given dependencies.


Citation
---------
If you found this paper or code helpful, please consider citing:
```
@inproceedings{weller-etal-2022-use,
    title = "When to Use Multi-Task Learning vs Intermediate Fine-Tuning for Pre-Trained Encoder Transfer Learning",
    author = "Weller, Orion  and
      Seppi, Kevin  and
      Gardner, Matt",
    booktitle = "Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers)",
    month = may,
    year = "2022",
    address = "Dublin, Ireland",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.acl-short.30",
    doi = "10.18653/v1/2022.acl-short.30",
    pages = "272--282",
}
```

