## Running a python pylabianca script

This document assumes that you've followed the "Setting up python environment" instructions from pylabianca repository.

To run the reference unit selection script you need to:
1. know where the pylabianca package was downloaded to (by default it will be in src subdirectory in the folder where the `environment.yml` file was located). If you don't remember the folder - take a look at section `I` to see how to find it using python.
2. know the directory where you have your exported units (`.mat` files with spike info after curation)

First you need to open the script. It is located in pylabianca package directory:<br />
`pylabianca\pylabianca\scripts`<br />
See next section to see how to find pylabianca package directory. You can also download the script from our pylabianca github repository (in the same way you downloaded the environment file).

### I. How to find pylabianca package location?
If you don't know where pylabianca package is located, you can follow these steps:
1. open Spyder (look for "Spyder (pylabianca)" in Windows start menu).
2. In the spyder console (lower left part of the Spyder GUI) type:
   `import pylabianca as pln`
3. The import should not raise any errors. Then you can write:
   `pln.__file__`
   This will display the location of the main pylabianca file and thus - the package directory.<br />
<img src="img/02_find_pylabianca_location.PNG" style="max-width:750px;width:75%">

### II. How to open and run the script
There are many ways to open/run a python script, but I explain below how to do it from Spyder, as it should be the easiest:
1. Open the script using `File -> Open` menu.<br />
   <img src="img/02_open_file.PNG" style="max-width:200px;width:20%">
3. Click on the green arrow (or `F5`) to run the script.<br />
   <img src="img/03_run_file.PNG" style="max-width:300px;width:40%">

Of course, before running the script you would like to change the paths and sometimes other parameters, see next section.

### III. `spike_postproc_choose_ref.py` - parameters
From line 25 to line 62 you should find the settings section of the script. All the settings are described
in the script. You have to change `data_dir` and `save_fig_dir` variables.<br />
You can wrap these variables into multiple lines as currently in the script, or just copy respective paths and paste into one line, for example:
```python
save_fig_dir = r'C:\Users\mmagnuski\Dropbox\PROJ\Labianka\sorting\ref_tests\sub-W02_test01'
```
Note that you need the `r` before the string (like above) so that backslashes are treated literally (so that `'\n'` does not mean new line for example).

### IV. Investigate script outputs
Once you've run the script you should see similar output produced on the screen successively:
<img src="img/04_refsel_script_output.PNG" style="max-width:700px;width:70%">

Currently `spike_postproc_choose_ref.py` does not save modified data, only generates figures to check if the reference selection it performs makes sense. Go to the `save_fig_dir` to see the generated figures. The contents of the directory should look similar to this:
<img src="img/05_figures_folder.PNG" style="max-width:700px;width:70%">

The first folder "ignored_clusters" contains coincidence clusters that were ignored as they did not reach `min_ref_channels` threshold. Then each processed cluster is shown with three images:
* `cluster_xx_01_coincid.png` (where `xx` is the cluster number) - presents coincidence matrix between units in the cluster.
* `cluster_xx_02_score_percentiles.png` shows which unit would have been picked using percentile method.
* `cluster_xx_03_score_within_cluster_ranks.png` shows which unit would have been picked using within-cluster ranks method.

Both methods should give similar results, but they may point to different units sometimes.<br />
Currently high weight is attributed to waveform density - a measue that should prefer waveforms with nice concentration of many spikes - so in many cases units with many spikes will be preferred. Take a look if there are any tricky cases where the script chooses units that are not good. 
