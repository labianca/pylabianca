# pylabianca
Python tools for spike and LFP analysis.

## installation
`pylabianca` can be installed using `pip`:
```
pip install pylabianca
```
To get most up-to-date version you can also install directly from github:
```
pip install git+https://github.com/labianca/pylabianca
```

## docs
Online docs are currently under construction.  

You can get the example human data [here](https://www.dropbox.com/scl/fo/wevgovmxv8qrl52w12b6z/h?rlkey=1je64v2h1h6zyqhzmhiykpqqu&dl=0).  
The preprocessed FieldTrip data are available [here](https://www.dropbox.com/scl/fo/i6q4e0ix805dds92jibmw/h?rlkey=cfdm1730qubqwb64zj1j02tvt&dl=0).

Below you can find jupyter notebook examples showcasing `pylabianca` features.
* [introductory notebook](doc/intro_overview.ipynb) - a general overview using human intracranial spike data (sorted with Osort).  
* [FiedTrip data example notebook](doc/fieldtrip_example.ipynb) - another broad overview using fieldtrip sample spike data from non-human primates.
* [decoding example](doc/decoding_example.ipynb) - overview of decoding with pylabianca

To better understand the data formats read natively by pylabianca (and how to read other formats) see [data formats page](doc/data_formats.md).
