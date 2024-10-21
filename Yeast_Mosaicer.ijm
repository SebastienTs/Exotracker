/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Name: 		Yeast_Mosaicer
// Version:		1.0
// Author:		Sébastien Tosi (sebastien.tosi@gmail.com)
// Aim:			- Crop out yeast cells / clusters and montage them (fixed size windows)
//				- Preprocess the montage (2D+T blur, bleach correction)
//				- Apply the same operations to an extra _C2 channel (same crop windows)
//
// Usage:		- Run the macro (drag and drop to Fiji + Run)
//				- Set the folder paths, root filename and configuration
//
/////////////////////////////////////////////////////////////////////////////////////////////////

// Dialog box
Dialog.create("Yeast_Mosaicer");
Dialog.addDirectory("Experiment folder", "D:/Projects/UPF/Widefield_TIRF/30C/3_Preprocessed/");
Dialog.addString("Files root name (_C1)", "OGY1494_Su_Sec9_30_prepro_C1_", 50);
Dialog.addDirectory("Montages export folder", "D:/Projects/UPF/Widefield_TIRF/30C/5_Montage/");
Dialog.addNumber("Number of files to process", 10);	
Dialog.addNumber("Crop window size (pixels)", 160);	
Dialog.addCheckbox("Apply preprocessing", true);
Dialog.addCheckbox("Process extra channel (_C2)", true);
Dialog.show();
experimentfolder = Dialog.getString();
firstchanfile = Dialog.getString();
outputfolder = Dialog.getString();
nprocess = Dialog.getNumber();
wndsz = Dialog.getNumber();
preprocess = Dialog.getCheckbox();
processch2 = Dialog.getCheckbox();
basename = experimentfolder+firstchanfile;
basename2 = replace(basename, "_C1", "_C2");

// Cells segmentation parameters
gaussrad = 9;		// yeast cells segmentation: Gaussian blur radius (pix)
bckrad = 50;		// yeast cells segmentation: background subtraction radius (pix)
minarea = 1000;		// yeast cells segmentation: minimum object area (pix)
expand = 16;		// dilation around segmented objects (pix)

// Postprocessing
xblur = 0.75;		// preprocessing: Gaussian blur (pix)
tblur = 1;			// preprocessing: time blurring (time step)
bcklvl = 12;		// preprocessing: bleach correction background level (intensity level)

run("Close All");
run("ROI Manager...");
setBatchMode(true);
for(loop=0;loop<nprocess;loop++)
{
	// Channel 1
	print("loop: "+d2s(loop,0));
	open(basename+d2s(loop,0)+".tif");
	originalid = getImageID();
	XSize = getWidth();
	YSize = getHeight();
	ZSize = nSlices;
	run("Clear Results");
	run("Select None");
	run("Duplicate...", "title=Mask");
	run("Gaussian Blur...", "sigma="+d2s(gaussrad,0)+" stack");
	run("Subtract Background...", "rolling="+d2s(bckrad,0));
	setAutoThreshold("Default dark");
	run("Analyze Particles...", "size="+d2s(minarea,0)+"-Infinity exclude clear include add");
	run("Set Measurements...", "area bounding redirect=None decimal=2");
	roiManager("Measure");
	selectImage("Mask");
	close();
	for(i=0;i<nResults;i++)
	{
		X0 = getResult("BX", i);
		Y0 = getResult("BY", i);
		DX = getResult("Width", i);
		DY = getResult("Height", i);
		selectImage(originalid);
		makeRectangle(X0-expand, Y0-expand, DX+2*expand, DY+2*expand);
		run("Duplicate...", "title=Crop duplicate");
		if((getWidth()>wndsz)||(getHeight()>wndsz))print("Cropped!");
		run("Canvas Size...", "width="+d2s(wndsz,0)+" height="+d2s(wndsz,0)+" position=Center zero");
		if(isOpen("Row"))run("Combine...", "stack1=Row stack2=Crop");
		rename("Row");
	}
	resetMinAndMax();
	selectImage(originalid);
	close();
	if(isOpen("Montage"))run("Combine...", "stack1=Montage stack2=Row combine");
	rename("Montage");
	
	// Channel 2
	if(processch2)
	{
		open(basename2+d2s(loop,0)+".tif");	
		originalid = getImageID();
		for(i=0;i<nResults;i++)
		{
			X0 = getResult("BX", i);
			Y0 = getResult("BY", i);
			DX = getResult("Width", i);
			DY = getResult("Height", i);
			selectImage(originalid);
			makeRectangle(X0-expand, Y0-expand, DX+2*expand, DY+2*expand);
			run("Duplicate...", "title=Crop duplicate");
			if((getWidth()>wndsz)||(getHeight()>wndsz))print("Cropped!");
			run("Canvas Size...", "width="+d2s(wndsz,0)+" height="+d2s(wndsz,0)+" position=Center zero");
			if(isOpen("Row2"))run("Combine...", "stack1=Row2 stack2=Crop");
			rename("Row2");
		}
		resetMinAndMax();
		selectImage(originalid);
		close();
		if(isOpen("Montage2"))run("Combine...", "stack1=Montage2 stack2=Row2 combine");
		rename("Montage2");
	}
}
setBatchMode("exit & display");

if(preprocess)
{
	selectImage("Montage");
	run("Gaussian Blur 3D...", "x="+d2s(xblur,2)+" y="+d2s(xblur,2)+" z="+d2s(tblur,2));
	run("Bleach Correction", "correction=[Simple Ratio] background="+d2s(bcklvl,0));
	save(outputfolder+File.separator+"Montage_preprocessed_C1.tif");
	close();
	selectImage("Montage");
	save(outputfolder+File.separator+"Montage_C1.tif");
	close();
	if(isOpen("Montage2"))
	{
		selectImage("Montage2");
		run("Gaussian Blur 3D...", "x="+d2s(xblur,2)+" y="+d2s(xblur,2)+" z="+d2s(tblur,2));
		run("Bleach Correction", "correction=[Simple Ratio] background="+d2s(bcklvl,0));
		save(outputfolder+File.separator+"Montage_preprocessed_C2.tif");
		close();
		selectImage("Montage2");
		save(outputfolder+File.separator+"Montage_C2.tif");
		close();
	}
}
