### Final Machine Learning Project 2017
# Predicting High Priority Violators of Clean Air Act
#### Group Members
### Andrew Yaspan, Regina Widjaya, Shambhavi Mohan, Sun-joo Lee

#### Motivation
Air pollution is a significant global issue for several immediate and long-term reasons. People who live in areas in the vicinity of facilities that contribute to air pollution are exposed to unsafe levels of hazardous materials that can have serious health consequences, such as increased rates of cancer, developmental and reproductive effects among others.
EPA can conduct only limited number of inspections per year. Currently, the EPA has just two major ways of selecting facilities for inspection: 1) facilities with high residential population or 2) facilities with a history of violations or have not filled out the RMP Inspection report.
We wanted to create a more robust data driven selection criteria

#### Dataset
The EPA’s Enforcement and Compliance History Online (ECHO) website provides publicly available data regarding facility-specific compliance and enforcement information.
Used the following dataset:
Air Full Compliance Evaluations (FCEs) and Partial Compliance Evaluations (PCEs); Case Files HPVs and FRVs; Air Title V Certifications; Air Stack Tests; ICIS-Air Formal Actions; ICIS-Air Informal Actions

#### Our Files:
* ICIS_explore2.py: Data Exploration with graphs to understand the distribution of our data
* ICIS-Output&Features.ipynb / ICIS-Output_Features.py:  Ran a loop with mulitple machine learning algorithms with different parameter sets that allows us to view the output and evaluation metrics of the various machine learning models over different time intervals between January 1, 2000 and January 1, 2017. We experimented with various models to pick a model to best address our selected policy problem- helping the EPA improve their Clean Air Act inspection schedule in an effor to prevent further violations.
* In order to run the functions in this file, you must visit this link: https://echo.epa.gov/tools/data-downloads and download the ICIS-AIR_downloads zip file, then place that folder inside this directory.

#### Results
* Ensemble tree (gradient boosted and extra tree) methods outperform the other methods which is great because of limited data and hence chances of overfitting.
* To measure performance we considered recall and the F1 score as the most important performance metric.

#### Limitations
* Segmented dataset with limited common values.
* No standardized tests were performed in all facilities
