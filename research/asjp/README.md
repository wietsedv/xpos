# ASJP LDND distances

Data and instructions from

https://asjp.clld.org/help

## Usage

You can directly use the `ldnd.txt` output file. To reproduce this file, you can run the following commands:

```bash
python3 filter.py  # creates `lists2.txt` based on `lists.txt` (check whitelist filter in `filter.py`)
# gfortran asjp62.f -o asjp62  # macOS
./asjp62 < lists2.txt > ldnd.txt
python3 most_similar.py en
```
