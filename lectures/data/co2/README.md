# Mauna Loa monthly mean CO2

`co2_mm_mlo.csv`: monthly mean atmospheric CO2 (ppm) at Mauna Loa Observatory, March 1958 to October 2026,
downloaded on 2026-11-23 from NOAA Global Monitoring Laboratory:
https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_mm_mlo.csv

Saved here so the week-11 lecture and exercises work offline and do not change when NOAA updates the file.
The file's own header (lines starting with `#`) describes the data and NOAA's request for credit; keep it.

Columns: `year, month, decimal date, average, deseasonalized, ndays, sdev, unc`.
Load with `pandas.read_csv(path, comment="#")`.
