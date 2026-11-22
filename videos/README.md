Put the generated videos of each system in its own folder, one file per script named after its key in
`data/benchmark.json`:

    videos/<model_name>/<sample_id>.mp4

The benchmark has 300 scripts: 125 of 4 s, 125 of 8 s (the base set) and 50 of 15 s (the extension). Systems that cannot
generate 15 s videos can be evaluated on the base set only; `msbench aggregate` reports each duration separately.

`videos/example/` holds the two smoke-test videos (`bash scripts/smoke_test.sh`).
