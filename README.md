## Machine Learning for Materials Science  

### Objective
The purpose of this project is to use machine learning and data visualization tools to predict the properties of the materials. My current project is dedicated to perovskite materials with the general chemical formula of **ABO3**.   
The main source of information in my journey into the application of ML models in materials science comes from the amazing international project and its repositories, ***materials project***. (https://materialsproject.org/)  

### Dataset
The dataset is the result of the nice work done by Antoine Emery, entitled *High-throughput DFT calculations of formation energy, stability, and oxygen vacancy formation energy of ABO3 perovskites* and was published with DOI (https://doi.org/10.6084/m9.figshare.5334142.v1).  
The features in the dataset have different kinds of correlation together, some more linear relations, some non-linear, cluster-based relations etc. I assume this is typical for many classes of materials, as chemical and physical properties usually do not show a linear correlation in the real world.  
To handle the multiclass problem with the different relationships among features, I think the use of a Stacking Classifier might be the safer approach to get the benefit of the potential of different models like regression, tree-based etc.
