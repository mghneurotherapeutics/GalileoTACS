# Galileo TACS


## Layout & Instructions

## Manuscript Information
Paper submitted to Frontiers on ..... The initial commit represents the code in the state used to generate the figures for the initial submission.

It was returned for revisions on ..... The commit XX represents the commit that has the code at the status when revisions were submitted.

## Computing Environment & Dependencies

### Python

Most of the analysis was done with Python. I have included an environment.yml file that one can use to recreate the Python environment with anaconda. Instructions for installing anaconda can be found <a href="https://conda.io/docs/user-guide/install/download.html">here</a>. Once installed you can run the following to create a conda environment and switch to it:

	conda env create -f environment.yml
	source activate galileo

For more information on conda environments see <a href="https://conda.io/docs/user-guide/tasks/manage-environments.html">here</a>. Within the galileo environment you can then run all of the Python code.

Additionally, I highly recommend setting up jupyter notebook extensions to allow a table of contents for navigating the juptyer analysis notebook. This can be done by following steps 2 & 3 <a href="https://github.com/ipython-contrib/jupyter_contrib_nbextensions">here</a> (step 1 has already been done and is packaged in the environment). The extension is called Table of Contents (2). As a result, you will be able to click to different sections in the analysis notebook without having to scroll through.

### MNE-C Tools

Instructions for installing and sourcing the MNE-C tools can be found <a href="https://mne-tools.github.io/stable/install_mne_c.html">here</a>. This code only uses the mne_browse_raw
function which can be evoked at the command line. Only compatible with Linux and MacOS.

mne_browse_raw was used to inspect the raw data and mark stimulation onsets and offsets. The MNE-C tools are not needed beyond this step.

### Blackrock & MATLAB

The original raw data comes in a Blackrock recording array specific format described <a href="http://support.blackrockmicro.com/KB/View/166838-file-specifications-packet-details-headers-etc">here</a>.

This code used version 4.4.0.0 of the NPMK toolkit with MATLAB R2015b to extract the raw data into .mat files. The NPMK toolkit can be downloaded <a href="https://github.com/BlackrockMicrosystems/NPMK/releases">here</a>.

MATLAB & the NPMK toolkit were only used to extract the data from the blackrock format and are not needed beyond this step.

### Computing Specs

All analysis was run on a Dell PowerEdge T330 Tower Server running CentOS7 with 64 GB of RAM.

All of the data, including raw and processed formats, requires 122 GB of storage space. This is due to inefficient repetition of the raw data being saved out in different formats as part of the extraction process.
Working with only the processed data starting from epochs or the final version of the raw data results in 30 or 45 GB of space needed respectively.

Due to the large file sizes, the code is quite memory intensive, especially if using multiple threads.


