# Exotracker
A python script to track exocitosis events in yeast cells

Installation
------------

1) Install latest miniconda: https://docs.anaconda.com/miniconda/miniconda-install/ (use default path)
2) Clone the repository or download it as a zip file and uzip it to an empty folder
3) Copy requirements.txt to your USER_HOME (e.g. C:/Users/sebas in Windows)
4) Launch Anaconda Prompt (miniconda3) and type:
	* conda create -n exotracker python=3.9
	* conda activate exotracker
	* pip install -r requirements.txt
   
<ins>Note</ins>:

The time-lapses should be pre-processed by slight 2D+T Gaussian blur and bleach correction.<br />
This can be performed with the ImageJ macro Yeast_Mosaicer.ijm provided which additionally montages all cell clusters from a set of time-lapses to a mosaic.
