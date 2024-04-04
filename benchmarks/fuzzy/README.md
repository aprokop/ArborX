# Input

The example controls its input through command-line options:
- `--binary`
  Indicator whether the data provided through `--filename` option is text or
  binary. If the data is binary, it is expected that number of points is an
  4-byte integer, and each coordinate is a 4-byte floating point number.
- `--core-min-size`
  `minPts` parameter of the DBSCAN algorithm
- `--eps`
  `eps` parameter of the DBSCAN algorithm
- `--filename`
  The data is expected to be provided as an argument to the `--filename`
  option.
- `--impl`
  Switch between two algorithms described in [2]: `fdbscan` (FDBSCAN) and
  `fdbscan-densebox` (FDBSCAN-DenseBox).

## Data file format

 For an `d`-dimensional data of size `n`, the structure of the file is `[n, d,
 p_{1,1}, ..., p_{1,d}, p_{2,1}, ..., p_{2,d}, ...]`, where `p_i = (p_{i,1},
 ..., p_{i,d})` is the `i`-th point in the dataset. In the binary format, all
 fields are 4 bytes, with size and dimension being `int`, and coordinates being
 `float`.

# Output

The example produces cluster labels.

# Running the example with the HACC data

```shell
./ArborX_Benchmark_Fuzzy.exe --eps 1 --filename input.txt --core-min-size 2
```
