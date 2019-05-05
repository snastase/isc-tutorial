# Intersubject correlation tutorial
This repo accompanies the manuscript "Measuring shared responses across subjects using intersubject correlation" by Nastase, Gazzola, Hasson, and Keysers ([2019](https://doi.org/10.1101/600114)). Here, you'll find a Jupyter Notebook tutorial ([`isc_tutorial.ipynb`](https://github.com/snastase/isc-tutorial/blob/master/isc_tutorial.ipynb)) introducing basic intersubject correlation (ISC) analyses and statistical tests as implemented in Python using the Brain Imaging Analysis Kit ([BrainIAK](http://brainiak.org/)). The notebook uses both simulated data and a publicly available fMRI dataset. Using Google Colaboratory, you can run the analyses interactively in the tutorial notebook entirely in the cloud. To navigate directly to the notebook on Google Colab, click here: [**Tutorial on Google Colab**](https://colab.research.google.com/drive/1EHI9buw-nvj5UDNg7MWUiQ1ITVJSswtH).

This notebook is geared toward early-career cognitive neuroscientists (e.g., graduate students) or researchers unfamiliar with ISC analysis. We assume some basic familiarity with Python. The tutorial provides an introductory treatment of the following topics:
* Computing ISCs
* Computing ISFCs
* Statistical tests for ISCs
* Correcting for multiple tests
* Loading and visualizing fMRI data

In addition to the tutorial, this repo contains several other tools: 1. a "standalone" suite of ISC functions for mimicking BrainIAK's ISC functionality without a BrainIAK installation ([`isc_standalone.py`](https://github.com/snastase/isc-tutorial/blob/master/isc_standalone.py); may be slower than the BrainIAK implementation); 2. MATLAB code for computing leave-one-out ISCs ([`isc_loo.m`](https://github.com/snastase/isc-tutorial/blob/master/isc_loo.m)); 3. a Python-based command-line interface for computing leave-one-out ISCs ([`isc_cli.py`](https://github.com/snastase/isc-tutorial/blob/master/isc_cli.py); requires a Python 3 installation with NumPy/SciPy and NiBabel on Linux or Mac); 4. several files for unit testing and generating figures in the manuscript. Given two or more NIfTI images containing BOLD time series (one file per subject), the command-line tool computes leave-one-out ISCs with a variety of options for supplying a mask image, z-scoring inputs, and Fisher z-transforming or summarizing the output (see `python3 isc_cli.py --help` for more detailed documentation). Usage example:
```python
# Three subjects in working directory with mask and no summarization
python3 isc_cli.py --input s1.nii.gz s2.nii.gz s3.nii.gz --output isc.nii.gz \
                   --mask mask.nii.gz --zscore

# Wildcard subject IDs with full paths and Fisher z-transformation
pyhon3 isc_cli.py --input /input/path/s*.nii.gz --output /output/path/isc.nii.gz \
                  --mask /input/path/mask.nii.gz --zscore --fisherz

# Compute ISCs and return mean across left-out subjects
python3 isc_cli.py --input s*.nii.gz --output mean_isc.nii.gz \
                   --mask mask.nii.gz --zscore --summarize mean
```

### What is ISC analysis?
ISC analyses measure stimulus-evoked responses that are shared across individuals. For example, in a conventional ISC analysis, we compute the correlation between response time series for a given brain area across individuals while they watch a movie or listen to a story ([Hasson et al., 2004](https://doi.org/10.1126/science.1089506), [2010](https://doi.org/10.1016/j.tics.2009.10.011)). This type of analysis reveals brain areas that are reliably engaged by the stimulus, ranging from low-level sensory structures to brain areas processing high-level narrative qualities of the stimulus ([Hasson et al., 2008](https://doi.org/10.1523/jneuroosci.5487-07.2008); [Lerner et al., 2011](https://doi.org/10.1523/jneurosci.3684-10.2011)). This is method is particularly useful for studying social communication ([Hasson et al., 2014](https://doi.org/10.1016/j.tics.2011.12.007)), because we can measure brain-to-brain coupling between speakers and listeners ([Stephens et al., 2010](https://doi.org/10.1073/pnas.1008662107); [Silbert et al., 2014](https://doi.org/10.1073/pnas.1323812111)). Rather than computing ISCs between corresponding brain across subjects, we can compute ISCs between brain regions, an approach called intersubject functional correlation (ISFC) analysis ([Simony et al., 2016](https://doi.org/10.1038/ncomms12141)). This approach measures reliable, stimulus-evoked functional integration, or "connectivity", across brain areas. As shown in the schematic below, ISFC analysis is a generalization of ISC analysis: the diagonal elements of the ISFC matrix represent the ISCs for each voxel, while the off-diagonal elements capture connectivity between voxels.

![Alt text](./figure_3.png?raw=true&s=100 "ISC and ISFC analysis schematic")

#### References
* Hasson, U., Ghazanfar, A. A., Galantucci, B., Garrod, S., & Keysers, C. (2012). Brain-to-brain coupling: a mechanism for creating and sharing a social world. *Trends in Cognitive Sciences*, *16*(2), 114–121. https://doi.org/10.1016/j.tics.2011.12.007

* Hasson, U., Malach, R., & Heeger, D. J. (2010). Reliability of cortical activity during natural stimulation. *Trends in Cognitive Sciences*, *14*(1), 40–48. https://doi.org/10.1016/j.tics.2009.10.011

* Hasson, U., Nir, Y., Levy, I., Fuhrmann, G., & Malach, R. (2004). Intersubject synchronization of cortical activity during natural vision. *Science*, *303*(5664), 1634–1640. https://doi.org/10.1126/science.1089506

* Hasson, U., Yang, E., Vallines, I., Heeger, D. J., & Rubin, N. (2008). A hierarchy of temporal receptive windows in human cortex. *Journal of Neuroscience*, *28*(10), 2539–2550. https://doi.org/10.1523/jneuroosci.5487-07.2008

* Lerner, Y., Honey, C. J., Silbert, L. J., & Hasson, U. (2011). Topographic mapping of a hierarchy of temporal receptive windows using a narrated story. *Journal of Neuroscience*, *31*(8), 2906–2915. https://doi.org/10.1523/jneurosci.3684-10.2011

* Nastase, S. A., Gazzola, V., Hasson, U., & Keysers, C. (2019). Measuring shared responses across subjects using intersubject correlation. *bioRxiv*, 600114. https://doi.org/10.1101/600114

* Silbert, L. J., Honey, C. J., Simony, E., Poeppel, D., & Hasson, U. (2014). Coupled neural systems underlie the production and comprehension of naturalistic narrative speech. *Proceedings of the National Academy of Sciences of the United States of America*, *111*(43), E4687–E4696. https://doi.org/10.1073/pnas.1323812111

* Simony, E., Honey, C. J., Chen, J., Lositsky, O., Yeshurun, Y., Wiesel, A., & Hasson, U. (2016). Dynamic reconfiguration of the default mode network during narrative comprehension. *Nature Communications*, *7*, 12141. https://doi.org/10.1038/ncomms12141

* Stephens, G. J., Silbert, L. J., & Hasson, U. (2010). Speaker–listener neural coupling underlies successful communication. *Proceedings of the National Academy of Sciences of the United States of America*, *107*(32), 14425–14430. https://doi.org/10.1073/pnas.1008662107
