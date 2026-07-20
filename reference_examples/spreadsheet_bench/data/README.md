# SpreadsheetBench data

The spreadsheet payload is **not** checked into git (the global `**/data/` rule ignores
it; only `loader.py`, `fetch_data.sh`, this README, and `__init__.py` are tracked).

Fetch it with:

```bash
bash data/fetch_data.sh                       # sample_data_200 (bring-up default)
bash data/fetch_data.sh spreadsheetbench_912_v0.1
bash data/fetch_data.sh spreadsheetbench_verified_400
```

Provenance: https://github.com/RUCKBReasoning/SpreadsheetBench (Apache-2.0), data blobs
stored via git-LFS under the upstream `data/` directory. `fetch_data.sh` pulls the real
blob from the `media.githubusercontent.com` LFS endpoint and extracts it to
`data/<source>/`.

Expected layout after extraction (loader tolerates one extra nesting level and a
`*.jsonl` index in place of `dataset.json`):

```
data/<source>/dataset.json
data/<source>/spreadsheet/<id>/{1,2,3}_<id>_input.xlsx
data/<source>/spreadsheet/<id>/{1,2,3}_<id>_answer.xlsx
```

Each instruction: `id`, `instruction`, `instruction_type`
(`Cell-Level Manipulation` | `Sheet-Level Manipulation`), `answer_position`
(comma-separated, optionally sheet-qualified ranges, e.g. `Sheet1!A1:B10`).

Release differences the loader handles automatically:
- `sample_data_200` / `spreadsheetbench_912_v0.1`: files `{n}_{id}_input.xlsx` +
  `{n}_{id}_answer.xlsx`, ~3 test cases each.
- `spreadsheetbench_verified_400`: files `{n}_{id}_init.xlsx` + `{n}_{id}_golden.xlsx`
  (1 test case each), plus an extra `answer_sheet` field (used as the default sheet for
  positions without a `Sheet!` qualifier). A few records have malformed `answer_position`
  ranges; the comparator treats those as unparseable (scored as a miss) rather than crashing.
  Use `sample_data_200` / the 912 set for the cleanest standard format.
