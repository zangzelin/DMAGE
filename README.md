# Deep Manifold Transformation for Dimension Reduction  (DMT)


This is a PyTorch implementation of the DMT



The code includes the following modules:
* Datasets (Swiss Roll, S-Curve, MNIST, SpheresA, SpheresB, Fashion MNIST, Coil20, Coil100)
* Training for DMT-Enc and DMT-AE (DMT-Enc + DMT-Dec)
* Test for manifold learning (DMT-Enc) 
* Test for manifold generation (DMT-Dec) 
* Visualization
* Evaluation metrics 
* The compared methods include: UMAP, [t-SNE](https://github.com/scikit-learn/scikit-learn), Topological AutoEncoder (TopoAE)</a>, [Modified Locally Linear Embedding (MLLE)](https://github.com/scikit-learn/scikit-learn), [ISOMAP](https://github.com/scikit-learn/scikit-learn), . (Note: We modified the original TopoAE source code to make it able to run the Swiss roll dataset by adding a swiss roll dataset generation function and modifying the network structure for fair comparison.)

## Requirements

* pytorch == 1.6.0
* scipy == 1.4.1
* numpy == 1.18.5
* scikit-learn == 0.21.3
* csv == 1.0
* matplotlib == 3.1.1
* imageio == 2.6.0

## Description

* main.py  
  * Train() -- Train a new model (encoder and/or decoder)
  * Test() -- Train a new model (encoder and/or decoder)
  * InlinePlot() -- Inline plot intermediate results during training
* dataset.py  
  * LoadData() -- Load data of selected dataset
* model.py  
  * Encoder() -- For latent feature extraction
  * Decoder() -- For generating new data on the learned manifold 
  * MLDL_Loss() -- Calculate six losses
* indicator.py -- Calculate performance metrics from results, each being the average of 10 seeds
* tool.py  
  * GIFPloter() -- Auxiliary tool for online plot
  * CompPerformMetrics() -- Auxiliary tool for evaluating metric 
  * Sampling() -- Sampling in the latent space for generating new data on the learned manifold 

## Running the code

1. Clone this repository

  ```
  git clone https://github.com/Westlake-AI/DMT.git
  ```

2. Install the required dependency packages

3. To get the results of DMT and other baselines, run

  ```
python main.py # for DMT
python baseline.py # for other baseline
  ```

the results are available in ./log/

4. To get the metrics for DMT and other baseline

  ```
  python indicator.py
  ```
The evaluation metrics are available in `./result/v2/lisind.csv`