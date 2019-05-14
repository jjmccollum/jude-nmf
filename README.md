# jude-nmf
Non-negative matrix factorization applied to Wasserman's collation of the epistle of Jude.

## About the Project

### Introduction
The text-critical practice of grouping manuscripts (MSS) into families or texttypes often faces two obstacles: the methodological question of how exactly to isolate the groups, given the chicken-and-egg relationship between "good" group readings and "good" group manuscripts, and contamination in the manuscript tradition. We introduce _non-negative matrix factorization_ (NMF) as a simple, automated, and efficient solution to both problems. Within minutes, NMF can cluster hundreds of manuscripts and readings simultaneously, producing an output that details potential contamination according to an easy-to-interpret mixture model.

This repository houses the dataset we used for this project, as well as Python code for a rudimentary user interface through which one can consume input collation data, apply NMF to it, and output the results to user-readable Excel files.

### What Is Non-negative Matrix Factorization?
Suppose we have a collation consisting of _n_ MSS and _m_ readings (that is, total readings across however many variation units there are). We can easily encode this collation as an _m_ × _n_ matrix _X_ with 0-1 values, with a 1 occurring wherever a MS represented in one column attests to a reading represented by the corresponding row. As its name suggests, NMF attempts to find an _m_ × _k_ matrix _W_, called the _basis matrix_, and a _k_ × _n_ matrix _H_, called the _mixture matrix_, where both _W_ and _H_ have only non-negative values and their matrix product _WH_ approximates the original data matrix _X_ as closely as possible.

In the language of MSS and readings, the coefficients in the basis matrix _W_ tell us which readings are the most characteristic of each textual group. The columns of the basis matrix are best described as _weighted reading profiles_, similar to the slightly simpler profiles employed in the Claremont Profile Method (Frederik Wisse, _The Profile Method for the Classification and Evaluation of Manuscript Evidence, as Applied to the Continuous Greek Text of the Gospel of Luke_, SD 44 \[Grand Rapids, MI: Wm. B. Eerdmans Publishing, 1982\]). Naturally, isolated readings may not have strong basis coefficients for any group, as they are not informative regarding group membership, while other readings may be characteristic of more than one group (reflecting either polygenesis or shared ancestry of the groups involved). The assignment of lower numerical weights to these types readings helps to reduce their influence on the classification of MSS.

Meanwhile, the coefficients in the mixture matrix _H_ tell us the degree to which each group's reading profile contributes to the pattern of readings found in any given MS. If a MS is well-approximated by one group profile with a high coefficient, then it can be considered a pure representative of that group. Meanwhile, MSS with smaller coefficients divided over multiple groups can be viewed as the result of contamination.

### Application
The Python code maintained in this repository is intended to allow an interested user to apply NMF to any set of collation data, provided the data is encoded properly, but for the project that gave rise to this code, we used Tommy Wasserman's extensive collation of 560 MSS of the New Testament epistle of Jude (_The Epistle of Jude: Its Text and Transmission_, ConBNT 43 \[Stockholm: Almqvist & Wiksell International, 2006\]; for the digital dataset, see Tommy Wasserman, "Transcription of the Manuscripts Containing the New Testament Letter of Jude," 2012, http://dx.doi.org/10.17026/dans-xcz-cqbr). We note that our dataset for Wasserman's collation hides all details of textual content, as variant readings are described only by their codes from the book; for full details, the reader is encouraged to consult Wasserman's study. In experimenting with this data, we found that the resulting clusters inferred by NMF corresponded closely to human-identified textual families. Our research is detailed in Joey McCollum, "Biclustering Readings and Manuscripts via Non-negative Matrix Factorization, with Application to the Text of Jude" (forthcoming).

## Technical Details

### Software and Hardware Specs
For ease of use, we stored all collation datasets as Microsoft Excel spreadsheets. For all computational work, we used release 3.5 of the Python programming language as part of the Anaconda Distribution (https://www.anaconda.com/distribution/). To read and write data from and to Excel spreadsheets, we used the Python PANDAS package (pandas.pydata.org). For factorization and rank estimation routines, we used NIMFA, a Python library dedicated to NMF (Marinka Žitnik and Blaž Zupan, "NIMFA: A Python Library for Nonnegative Matrix Factorization," _Journal of Machine Learning Research_ 13 \[2012\], 849–853; this library is open-source and can be downloaded at https://github.com/marinkaz/nimfa). For solving non-negative least-squares systems of equations, we used the SciPy stack of open-source Python modules for scientific computing. (https://scipy.org/).

Our implementation of NMF was run on a platform with an Intel i7-4770 quad-core processor and 16GB of memory.

### Data Encoding and Preprocessing
The Python code provided in this repository expects collation data to be formatted in a specific way. Collation data input files are expected to be Excel spreadsheets with a header row containing MS IDs and a header column containing reading IDs. All unambiguous readings, including omissions, are represented by rows in the collation spreadsheet, and all MSS are represented by columns. While extremely fragmentary witnesses like highly lacunose MSS, correctors' hands, alternate readings, and commentary readings can be included as witnesses, they are best left out until the postprocessing stage. The Python code allows the user to specify a _minimum extant reading_ threshold for MSS to ensure that lacunose sources are set aside for classification until post-processing, to prevent them from negatively influencing NMF's inference of group reading profiles. The user can also specify a _minimum attestation threshold_ for readings; for example, a threshold of 2 would exclude singular readings from consideration.

To encourage NMF to construct group reading profiles that rely more of group-exclusive readings and therefore result in better-separated clusters, the user can also specify a transformation of the 0-1 input data to a values distributed according to the _term frequency-inverse document frequency_ (TF-IDF) weighting scheme (Karen Spärk Jones, "A Statistical Interpretation of Term Specificity and Its Application in Retrieval," _Journal of Documentation_ 28.1 \[1972\], 11–21). As its name might suggest, this scheme places more weight on readings not found in many MSS, while very common readings that are likely less informative are assigned lower weights.

### Factorization Method
Presently, our code only uses the the LSNMF method, an alternating least-squares formulation of NMF (see Chih-Jen Lin, "Projected Gradient Methods for Nonnegative Matrix Factorization," _Neural Computation_ 19.10 \[2007\], 2756–2779), with the initial entries of _W_ and _H_ set using NNDSVD seeding (Christos Boutsidis and Efstratios Gallopoulos, "SVD Based Initialization: A Head Start for Nonnegative Matrix Factorization," _Pattern Recognition_ 41.4 \[2008\], 1350–1362).

### Postprocessing
Once NMF has produced a basis matrix _W_ and a mixture matrix _H_ for the extant collation data, the final step is to furnish tentative group classifications for the remaining fragmentary collation data. This is done by solving a series of non-negative least-squares equations to find a combination of established reading profiles (in the basis matrix _W_) that best approximates the reading pattern of each fragmentary witness. This can be done quickly using SciPy's `optimize.nnls` routine.

## How to Use

### Getting Set Up

The `nmf_classifier` module runs in Python 3.5+ and requires the following packages:

* `pandas`
* `sklearn`
* `scipy`
* `nimfa`

First, download the .zip archive of this repository or clone it using the command

`git clone https://github.com/jjmccollum/jude-nmf.git`

Then navigate to the py folder and enter the command

`python setup.py install`

Once the package is set up, you can run it from the interactive Python shell of your choice. If you are new to using Python, iPython (https://ipython.org/) and Jupyter Notebook (https://jupyter.org/) are particularly accessible options. Open the shell of your choice in the py directory of this package and then enter the command

`run nmf_classifier.py`

### Basic Usage

To initialize the NMF classifier on the Python command line, enter the command `nc = NmfClassifier()`. The initializer accepts optional parameters related to preprocessing collation data. For instance, the command `nc = NmfClassifier(min_extant=300, min_support=2, use_tfidf=True)` will initialize a classifier that will set aside MSS with fewer than 300 extant readings as fragmentary, exclude readings attested by fewer than two MSS (i.e., singular readings), and use TF-IDF weighting rather than uniform (0-1) weights for readings. The default settings are `min_extant=1`, `min_support=1`, and `use_tfidf=False`.

Once the NMF classifier is initialized, the first step is to read in the input collation data file. If the name of the file is `jude_collation.xlsx` and it is located in the data directory parallel to the directory containing this script, then the command to enter will look like `nc.read('../data/jude_collation.xlsx')`. The classifier will read in the collation data from the file, removing rows for readings with too little MS support and splitting the collation matrix into one matrix for MSS with enough extant readings and another for "fragmentary" MSS without enough extant readings.

Once the classifier has read in the input data, you can check the first ten rows of the collation matrix with the command `nc.extant_collation_df[0:10]`.

To find the best number of clusters to aim for, you'll need to perform rank estimation. To check every possible choice between 2 and 30 clusters, enter the command `nc.get_rank_ests(2, 30)`. Be warned that this process will take some time. Once it's finished, if you'd like to print out metrics relevant to rank estimation, enter the command `nc.print_rank_ests()`. If you do not supply your own output filename, a default name will be supplied. The output will be an Excel spreadsheet.

To cluster the collation data into ten groups, enter the command `nc.get_nmf_results(10)`. After this completes, you can check the output matrices by calling `nc.W_df` (for the reading-profile matrix) or `nc.H_df` (for the profile-MS matrix). To print the contents of these matrices, along with summary statistics for the NMF run, to an Excel workbook, enter the command `nc.print_nmf_results()`. Optionally, you can supply an output filename; otherwise, a default name will be chosen.

Once you have a basis matrix _W_ from an NMF run, you can use it to classify the fragmentary MSS as a post-processing step. To do this, enter the command `nc.get_fragmentary_nmf_results()`. You can check the mixture matrix _H_ for the fragmentary MSS by calling `nc.fragmentary_H_df`. To print the contents of this matrix to an Excel spreadsheet, enter the command `nc.print_fragmentary_nmf_results()`. Optionally, you can supply an output filename; otherwise, a default name will be chosen.
