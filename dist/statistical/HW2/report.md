# Question

What are latent variables, and what methods are used for dimensional reduction of multidimensional data? provide examples.

## Tasks  

1. Define latent variable
2. Describe dimensionality reduction methods (Provide example)

---
---

### 1)

Usage area:\
Latent variable are on essential concept in statics, machine learning and various scientific major, In particular way that is used in area involving complex data analysis and modelling.

Difference between observable and latent variables\
observable variables can be directly measured or observed, but the latent variables represent underlying factor or constructs that are not directly observable but are inferred from other measurable variable.

In other hand, a latent variable is a hidden or unobservable that influences observed data. These variable that influences observed data and not directly measurable but are inferred through mathematical models that relate them to observable data.

Types of latent variables

* Continuous: Variables takes a continuous value. In machine learning, continuous latent variables often appear in probabilistic model like Gaussian Mixture Model (GMM) and Factor Analysis.
* Categorical: Variables that represent discrete categories or group. The Categorical variables are often used in models like Latent Class Analysis (LCA) and Hidden Markov Models (HMMs).

""Factor Analysis: A technique used to describe variability among observed variables in terms of fewer unobserved (latent) variables called factors.""\
**Benefits of Latent Variables**

1. Simplification of Complex Models: Latent variables allow researchers and data scientists to capture underlying structures that explain complex relationships in the data.
2. Increased Model Accuracy: By accounting for hidden factors, models that include latent variables often result in more accurate predictions or estimates.
3. Theoretical Insight: Latent variables offer insights into abstract constructs and theories that cannot be directly measured, providing a deeper understanding of human behavior, societal trends, or hidden patterns in data.

**Challenges of Latent Variables**

1. Model Identification: Estimating latent variables can be difficult since they are not directly observable. This requires complex mathematical models and assumptions, which may not always hold true.
2. Interpretation: In some cases, latent variables can be difficult to interpret or explain, especially in high-dimensional models where the meaning of the latent variable may not be clear.
3. Overfitting: Introducing too many latent variables in a model can lead to overfitting, where the model becomes too tailored to the training data and performs poorly on new, unseen data.

---
---

### 2)

High dimensionality can lead to challenges like increased computational cost, overfitting, etc. Dimension reduction techniques is a fundamental process in data analysis and machine learning that transforms high-dimensional data into a lower-dimensional representation while preserving its essential structure.

Methods:

* Principal Component Analysis (PCA): A linear method that captures global variance
* t-distributed Stochastic Neighbor Embedding (t-SNE): A nonlinear method focusing on preserving local neighborhoods  
* Uniform Manifold Approximation and Projection (UMAP): A nonlinear technique that balances local and global structure while being scalable

#### Principal Component Analysis (PCA)

PCA is a statistical technique introduced by mathematician Karl Pearson in 1901. It works by transforming high-dimensional data into a lower-dimensional space while maximizing the variance (or spread) of the data in the new space. This helps preserve the most important patterns and relationships in the data.

Standardizing our dataset to ensures that each variable has a mean of $0$ and a standard deviation of $1$.

$$Z = \dfrac{(X−μ)}{\sigma}
​$$
* $\mu$  is the mean of independent features {$\mu = \mu_1, \mu_2, ..., \mu_m$}
* $\sigma$ is the standard of independent feature {$\sigma = \sigma_1, \sigma_2, ..., \sigma_m$}

Dataset $X$ (of shape $n \times d$, where $n$ is the number of samples and $d$ is the number of features):

$$X \in \mathbb{R}^{m \times n}$$

The main goal of PCA is to find a new set of axes (principal components) that maximize variance in the data. Compute the mean-centered data ($X_{\text{centered}} = X - \mu$) Then, Compute the covariance matrix $$C = \frac{1}{n-1} X_{\text{centered}}^T X_{\text{centered}}$$

Compute eigenvalues and eigenvectors of $C$:
   $$
   C v_i = \lambda_i v_i
   $$
where $\lambda_i$ are eigenvalues and $v_i$ are eigenvectors. Then, Sort eigenvectors by eigenvalues in descending order **(largest first)**. Select the top $k$ eigenvectors and form a matrix $W$ of shape ($d \times k$). Project the data onto the new space:
$$
X_{\text{reduced}} = X_{\text{centered}} W
$$
 ---
#### t-Distributed Stochastic Neighbor Embedding (t-SNE)

t-SNE is a nonlinear method that converts high-dimensional pairwise distances into probabilities to preserve local structures and advanced nonlinear dimensionality reduction technique primarily utilized for visualizing high-dimensional datasets. Unlike Principal Component Analysis (PCA), which prioritizes the preservation of global structures, t-SNE emphasizes maintaining local data relationships. This makes it particularly effective for cluster analysis, pattern recognition, and exploratory data visualization in fields such as bioinformatics, natural language processing.

The methodology can be summarized as follows:

First, compute pairwise similarities in high-dimensional Space, the probability of similarity between two points is measured using a conditional Gaussian distribution. The similarity between points $x_i $ and $ x_j $ is defined as:
     $$p_{j|i} = \frac{\exp(-||x_i - x_j||^2 / 2\sigma_i^2)}{\sum_{k \neq i} \exp(-||x_i - x_k||^2 / 2\sigma_i^2)} $$
The bandwidth parameter $ \sigma_i $ is chosen to ensure a fixed perplexity (like smoothed k-nearest neighbors approach).

Second, map data to a low-dimensional space, points are randomly initialized in a lower-dimensional space. The similarity between points in the reduced space is modeled using a **Student’s t-distribution** with a single degree of freedom:
     $$q_{j|i} = \frac{(1 + ||y_i - y_j||^2)^{-1}}{\sum_{k \neq i} (1 + ||y_i - y_k||^2)^{-1}} $$

The heavy-tailed nature of the (t-distribution) mitigates the tendency of distant points collapsing together.

**Optimize Mapping via Gradient Descent**\
The algorithm iteratively minimizes the Kullback-Leibler (KL) divergence between the probability distributions of high- and low-dimensional representations:
   $$ KL(P || Q) = \sum_i \sum_j p_{ij} \log \frac{p_{ij}}{q_{ij}} $$
The optimization is typically performed using gradient descent with adaptive learning rates to improve convergence.

#### Uniform Manifold Approximation and Projection (UMAP)

UMAP is based on principles from topology and Riemannian geometry. The core methodology can be described in the following steps:

1. **Graph Construction (Fuzzy Topology Representation)**:
   - UMAP constructs a weighted k-nearest neighbors (KNN) graph, encoding local relationships in the high-dimensional space.
   - The probability of a point $x_j$ being connected to $x_i$ is defined using a fuzzy set approach:
     $$p_{j|i} = \exp \left( \frac{-||x_i - x_j||^2}{\rho_i} \right)$$
   - The parameter $\rho_i$ controls the local connectivity, ensuring density-aware neighborhood estimation.

2. **Optimization via Laplacian Eigenmaps**:
   - UMAP seeks a low-dimensional representation that preserves the high-dimensional manifold structure.
   - A cross-entropy loss function is minimized between the high-dimensional and low-dimensional representations, ensuring both global and local structural integrity.

#### Comparison of Techniques

| **Aspect**                | **PCA**                                           | **t-SNE**                                                  | **UMAP**                                                   |
|---------------------------|---------------------------------------------------|------------------------------------------------------------|------------------------------------------------------------|
| **Nature**                | Linear, deterministic                             | Non-linear, probabilistic                                  | Non-linear, uses topological & geometric assumptions       |
| **What It Preserves**     | Global variance (overall data structure)        | Local neighborhood relationships                           | Both local neighborhoods and some global structure         |
| **Speed & Scalability**   | Very fast and scalable                            | Computationally intensive $O(n^2)$                  | Fast and scalable to large datasets                        |

Other methods:

**Locally Linear Embedding (LLE)**

Locally Linear Embedding (LLE) is a nonlinear manifold learning algorithm used for dimensionality reduction. It aims to preserve neighborhood relationships in a lower-dimensional space, making it useful for nonlinear data structures where traditional methods like PCA fail. LLE is based on the assumption that each data point and its nearest neighbors lie on or near a locally linear patch of a high-dimensional manifold.

Compute k-nearest neighbors (KNN)\[Identify the $k$ nearest neighbors of each data point],
reconstruct each point as a weighted sum of its neighbors\[The reconstruction weights $w_{ij}$], are found by minimizing the reconstruction error:
$$\sum_i || x_i - \sum_j w_{ij} x_j ||^2 $$
Subject to the constraint that the sum of weights for each point equals $1$:
   $$ \sum_j w_{ij} = 1 $$

Compute the low-dimensional embedding by find the lower-dimensional representation $y_i$ by minimizing the cost function:
   $$\sum_i || y_i - \sum_j w_{ij} y_j ||^2 $$

This ensures that the local geometric structure is preserved.


**Autoencoders (Neural Networks)**

Autoencoders are a type of artificial neural network used for unsupervised dimensionality reduction. They learn a compressed representation of input data by encoding it into a lower-dimensional space and then reconstructing the original input.

An autoencoder consists of two main parts:
1. Encoder: Maps high-dimensional input $x$ to a lower-dimensional latent representation $z$.
2. Decoder: Reconstructs the original input $x$ from the latent representation $z$.

The goal is to minimize the reconstruction loss:
$$L = ||x - \hat{x}||^2 $$
where $\hat{x}$ is the reconstructed output.

---
### reference

1. *[Latent Variable](https://www.geeksforgeeks.org/what-is-latent-variable/)*
2. Jolliffe, I. T. (2002). *Principal Component Analysis*. Springer.
3. https://www.geeksforgeeks.org/ml-t-distributed-stochastic-neighbor-embedding-t-sne-algorithm/?ref=next_article_top
4. https://www.geeksforgeeks.org/principal-component-analysis-pca/
5. van der Maaten, L., & Hinton, G. (2008). *Visualizing Data Using t-SNE*. JMLR.
6. McInnes, L., Healy, J., & Melville, J. (2018). *UMAP: Uniform Manifold Approximation and Projection*. arXiv.
8. The two methods Autoencoders (Neural Networks), Locally Linear Embedding (LLE) used the AI for understanding Algorithm.

**Amir Hosssein Movahedi**