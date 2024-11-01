# Multilayer Graph Attention Networks for COVID-19 Prediction with the Fusion Method and Explainable AI
This is the source code for Multilayer Graph Attention Networks for COVID-19 Prediction with the Fusion Method and Explainable AI. The flowchart and visualization of the multilayer network figures are shown below:

## Flowchart
![Flowchart](https://github.com/Fujimoto-lab-UTHealth-HHD/GAT-COVID19/blob/main/Flowchart.png)


## Visualization of the multilayer network
<img src="https://github.com/Fujimoto-lab-UTHealth-HHD/GAT-COVID19/blob/main/Visualization_of_the_multilayer_network.png" width="500" height = "600">

## Usage
-----
### Requirements
For FGAT and GNNExplainer:

* Python 3.10+, torch 2.5.0+, torch_geometric 2.6.1+ and corresponding versions of scikit-learn, pandas, and numpy

### Train
To train the FGAT model, run  (in Jupyter Notebook):
```
FGAT.ipynb
```


### Explainer
To train the GNNExplainer, run (in Jupyter Notebook):
```
Explainer.ipynb
```