# Question

Consider the linear system $( Ax = b )$. Instead of solving it directly, we formulate it as an optimization problem by minimizing the norm:  
$$
L(x) = \min_x ||Ax - b||
$$  

## Tasks  

1. Implement a Gradient Descent (GD) algorithm to solve this optimization problem.  
2. If the coefficient matrix $A$ is not full-rank, discuss five alternative algorithms that can be used to solve the system effectively.  
3. Choose one of these five methods, implement the optimization, and provide the corresponding code.  

---
---

### 1)

  We aim to solve the optimization problem with Gradient Descent for Least Squares Optimization:
  $$
  \min_x ||Ax - b||^2
  $$

  where:
  - $A$ is an $ m \times n $ matrix.
  - $x$ is an $n \times 1 $ vector (our unknown variable).
  - $b$ is an $m \times 1 $ vector.

  We want to find an $x$ that minimizes the squared Euclidean distance between $Ax$ and $b$.

  **Gradient Computation**\
  The loss function is:
  $$
  L(x) = ||Ax - b||^2 = (Ax - b)^T (Ax - b)
  $$
  Taking its gradient with respect to $x$:

  $$\nabla_x L(x) = 2A^T (Ax - b)$$

  **Gradient Descent Update Rule**\
  Gradient Descent (GD) updates $x$ iteratively as:

  $$
  x^{(k+1)} = x^{(k)} - \alpha \cdot \nabla_x L(x^{(k)})
  $$

  where:

  - $\alpha$ is the learning rate.
  - $x^{(k)}$ is the value of $( x )$ at iteration $k$.
  - $\nabla_x L(x) = 2A^T (Ax - b) $ is the gradient.

  **Choosing the Learning Rate**

  A proper learning rate $\eta$ is crucial:

  - If $\alpha$ is too large, GD may diverge.
  - If $\alpha$ is too small, convergence is slow.

  A good choice is:

  $$
  0 < \eta < \frac{1}{\lambda_{\max}(A^T A)}
  $$
  where $\lambda_{\max}(A^T A)$ is the largest eigenvalue of $( A^T A )$. This ensures convergence.

---
---

### 2)

When $A $ is rank-deficient (not full rank), the linear system $Ax = b$ does not have a unique solution. In such cases, standard solution methods might fail or yield unstable results. In such cases, using numerically stable methods is crucial. Below are five common methods to solve such systems along with detailed explanations and the relevant mathematical formulas.

1. **Gaussian Elimination with Partial Pivoting** \
  Transform the matrix $A$ into an upper triangular form through forward elimination, then solve the system using back substitution.

    **Algorithm**

    For each column $i$ (from 1 to $n$), find the row $k$ with the maximum absolute
    value in that column
    $
    k = \operatorname{arg\,max}_{j \in \{i, \dots, n\}} |a_{ji}|
    $, swap row $i$ with row $k$ (this operation can be represented using a permutation matrix $P$). For rows $j = i+1, \dots, n$, compute the multiplier
    $
    (\text{factor} = \frac{a_{ji}}{a_{ii}})
    $
    then update the remaining entries:
    $$
    a_{jk} \leftarrow a_{jk} - \text{factor} \cdot a_{ik} \quad \text{for } k = i, \dots, n,
    $$
    and update the right-hand side:
    $$
    b_j \leftarrow b_j - \text{factor} \cdot b_i.
    $$
    Once the matrix is in an upper triangular form, solve for \(x\) starting from the last row:
    $$
    x_i = \frac{b_i - \sum_{j=i+1}^{n} a_{ij}\,x_j}{a_{ii}}.
    $$
    Enhances numerical stability by selecting the largest pivot element at each step. It is straightforward and suitable for moderate-sized systems.

2. **Singular Value Decomposition (SVD)**

    **Algorithm**

    Decompose the matrix $A$ into three matrices:
    $
    A = U \Sigma V^T$,
    where:

    - $U$ is an $m \times m$ orthogonal matrix,
    - $\Sigma$ is an $m \times n$ diagonal matrix containing the singular values $\sigma_1, \sigma_2, \dots, \sigma_r$ (with $r$ being the rank of $A$)
    - $V$ is an $n \times n$ orthogonal matrix.

    To solve the system, use the Moore-Penrose pseudoinverse:
    $
    x = V\,\Sigma^{-1}\,U^T b,
    $ where $(\Sigma^{-1})$ is constructed by taking the reciprocal of each nonzero singular value. If any singular values are extremely small, regularization techniques can be applied to reduce their impact.\
    Decomposes the matrix into singular values and vectors, allowing the detection and proper handling of very small singular values. This method is especially effective for severely ill-conditioned systems.

3. **QR Factorization**

    **Algorithm**

    Factorize $A$ into an orthogonal matrix $Q$ and an upper triangular matrix $R$,
    $A = QR$ ,where:

    - $Q$ is an orthogonal matrix $Q^T Q = I$,
    - $R$ is an upper triangular matrix.

    Then, using the Classical Gram-Schmidt Process:

    - Step 1:  
      For the first column:
      $
      \mathbf{q}_1 = \frac{\mathbf{a}_1}{\|\mathbf{a}_1\|}.
      $

    - Step 2:  
      For $(i = 2, \dots, n)$:
      - Compute the intermediate vector:
        $$
        \mathbf{u}_i = \mathbf{a}_i - \sum_{j=1}^{i-1} (\mathbf{q}_j^T \mathbf{a}_i)\,\mathbf{q}_j,
        $$
      - Normalize to obtain:
        $$
        \mathbf{q}_i = \frac{\mathbf{u}_i}{\|\mathbf{u}_i\|}.
        $$

    - **Constructing $(R)$:**  
      The entries of $R$ are given by:
      $$
      r_{ji} = \mathbf{q}_j^T \mathbf{a}_i \quad \text{for } j \le i.
      $$

    **Solving the System**

      With $A = QR$, the system becomes:
      $
      QRx = b.
      $
      Multiplying both sides by $Q^T$ gives:\
      $
      Rx = Q^T b,
      $ which is then solved via back substitution.

      Provides high numerical stability due to the orthogonality of $Q$ and is ideal for sensitive systems.

4. **LU Decomposition with Pivoting**

    **Algorithm**

    Decompose the matrix $A$ into the product of a lower triangular matrix $L$ and an upper triangular matrix $U$ with a permutation matrix $P$:
    $
    PA = LU,
    $
    where:

    - $P$ is a permutation matrix (accounting for row exchanges),
    - $L$ is a lower triangular matrix with ones on the diagonal ($l_{ii} = 1$),
    - $U$ is an upper triangular matrix.

    **Equations**
    - **For elements of $U$:**
      $$
      u_{kj} = a_{kj} - \sum_{s=1}^{k-1} l_{ks}\,u_{sj}, \quad \text{for } k=1,\dots,n \text{ and } j=k,\dots,n.
      $$
    - **For elements of $L$:**
      $$
      l_{ik} = \frac{1}{u_{kk}} \left(a_{ik} - \sum_{s=1}^{k-1} l_{is}\,u_{sk}\right), \quad \text{for } i=k+1,\dots,n.
      $$

    Solution Process

    1. Solve $(Ly = Pb)$:
      Use forward substitution to obtain the intermediate vector $y$.
    2. Solve $(Ux = y)$: 
      Use back substitution to compute the final solution $x$.

    Decomposes $A$ into $L$ and $U$ while using pivoting to improve stability, making it a widely used method.

5. **Regularization Methods – Tikhonov Regularization**

    **Algorithm**
    
    Improve the stability of the solution by reducing the system’s sensitivity to noise and round-off errors by adding a regularization term.

    **Optimization Problem**\
    Formulate the problem as:
    $$
    \min_{x} \left\{ \|Ax - b\|^2 + \lambda \|x\|^2 \right\},
    $$
    where $\lambda$ is the regularization parameter.

    **Derivation of the Solution**

    1. Set Up the Gradient: 
      Compute the gradient with respect to $(x)$ and set it to zero:
      $$
      \nabla_x \left(\|Ax - b\|^2 + \lambda \|x\|^2\right) = 2A^T(Ax - b) + 2\lambda x = 0.
      $$
    2. Simplify:  
      Dividing through by 2:
      $$
      A^T A\, x - A^T b + \lambda x = 0.
      $$
    3. Rearrange:  
      $$
      (A^T A + \lambda I)x = A^T b.
      $$
    4. Solution:  
      $$
      x = (A^T A + \lambda I)^{-1} A^T b.
      $$
      Adds a regularization term to stabilize the solution by mitigating the effects of noise and round-off errors. Each method has its advantages and is chosen based on the characteristics of the matrix $A$, the required accuracy, and the system's sensitivity to numerical errors.

---
---

### reference

For all algorithm searches, refer to [Wikipedia](https://en.wikipedia.org/wiki/Main_Page). To check the text, use ChatGPT, except for Gaussian Elimination with Partial Pivoting, for which you should use [this source](https://www-personal.umd.umich.edu/~fmassey/math473/Notes/c1/1.5.1%20LU%20decompositions%20with%20partial%20pivoting.pdf). For Regularization Methods – Tikhonov Regularization, refer to [this document](https://www.mit.edu/~9.520/scribe-notes/cl7.pdf).


**Amir Hosssein Movahedi**
