#!/usr/bin/env python
# coding: utf-8

# # Iterative Methods for Linear Systems

# ## A general framework of Iterative Methods
# 
# Goal: Solve a large linear system
# 
# $$
# A\vec{x}=\vec{f},
# $$
# 
# where $A\in M_{n\times n}(\mathbb C)$ is a large matrix, $\vec f\in \mathbb C^n$ is a vecor.
# 
# The general idea is of Iterative Method is to develop an iterative scheme, that is we want to find a sequenence of vector $\vec{x}^0,\vec{x}^1,\cdots$ such that
# 
# $$
# \vec{x_k}\to \vec{x}^*\qquad as \qquad k\to \infty,
# $$
# 
# where $\vec{x}^*$ is the analytic solution of $A\vec{x}=\vec{f}$.
# 
# To invent a general iterative scheme, we choose a matrix $N$ and **split** matrix $A$ into
# 
# $$
# A=N-(N-A)=N-P.
# $$
# 
# The choice of $N$ can vary under different situation. There are some classical choices of $N$, and we will discuss them later. Now that we have this splitting equation, we have
# 
# $$
# A\vec{x}=\vec{f} \iff (N-P)\vec{x}=\vec{f} \iff N\vec{x}=P\vec{x}+\vec{f}.
# $$
# 
# From the above equation, we can easily develop an iterative scheme as follows:
# 
# $$
# N\vec{x}^{k+1}=P\vec{x}^k+\vec{f}.
# $$
# 
# We always choose an invertible $N$, therefore
# 
# $$
# \vec{x}^{k+1}=N^{-1}P\vec{x}^k+N^{-1}\vec{f}.
# $$(eqn:1)
# 
# Obviously, if the sequence $\{\vec x^k\}_{k=0}^{\infty}$ converges, it converges to $\vec{x}^*$, the analytic solution of $A\vec{x}=\vec{f}$.

# To ensure the convergence, we study the error sequence $\{\vec e^k\}_{k=0}^{\infty}$ where
# 
# $$
# \vec e^k=\vec x^k-\vec x^*.
# $$
# 
# Denote $M=N^{-1}P$. From $(1)$ we know that 
# 
# $$
# \begin{align*}
# \vec{x}^{k+1}&=M\vec{x}^k+N^{-1}\vec{f}\tag{2}\\
# \vec{x}^*&=M\vec{x}^*+N^{-1}\vec{f}\tag{3}\\
# \end{align*}
# $$
# 
# Using $(3)-(2)$, we get
# 
# $$
# \vec{e}^{k+1}=M\vec{e}^k=\cdots=M^k\vec{e}^0\tag{4}.
# $$
# 
# Since $\vec{e}^0$ is an arbitrary number depending on the intial value of the iterative scheme, the convergence of $(4)$ is determined by $M^k$. If $M^k$ converges to $0$ as $k$ goes to infinity, then the error sequence will converge to $0$ as well. 
# 

# ## Convergence of $M^k$
# 
# **Question**: Will $M^k$ converge to $0$?
# 
# Firstly, we consider a simple case when $M$ is diagonalizable. That is, $M$ has $n$ linearly independent eigenvector $\{\vec u_1,\vec u_2,\cdots,\vec u_n\}\in \mathbb C^n$, with $n$ corresponding eigenvalues $\lambda_1,\lambda_2,\cdots,\lambda_n$ s.t.
# 
# $$
# D=Q^{-1}MQ,
# $$
# 
# where
# 
# $$
# D=diag(\lambda_1,\lambda_2,\cdots,\lambda_n)\qquad \text{and}\qquad
# Q = 
# \left[
#   \begin{array}{cccc}
#     \mid & \mid & & \mid\\
#     \vec u_{1} & \vec u_{2} & \ldots & \vec u_{n} \\
#     \mid & \mid & & \mid 
#   \end{array}
# \right].
# $$
# 
# WLOG, we assume $\mid \lambda_1\mid \geq \mid \lambda_2\mid \geq \cdots \geq \mid \lambda_n\mid$. Subsequently, we have
# 
# $$
# M\vec u_i=\lambda_i \vec u_i\qquad\forall i=1,\cdots,n.
# $$
# 
# Let the initial error to be $\vec e^0=\sum_{i=1}^na_i\vec u_i$. We can always find such $a_i,\cdots,a_n$ because $\{\vec u_1,\vec u_2,\cdots,\vec u_n\}$ forms a basis in $\mathbb C^n$.
# 
# Then,
# 
# $$
# \vec e^k=M^k\vec e^0=\sum_{i=1}^n a_i M^k\vec u_i=\sum_{i=1}^n a_i \lambda_i^k\vec u_i\leq \lambda_1^k(a_1\vec u_1+\sum_{i=2}^n a_i (\frac{\lambda_i}{\lambda_1})^k\vec u_i).\tag{5}
# $$
# 
# Since $\mid \lambda_1\mid$ is the largest eigenvalue in magnitude, $(\frac{\lambda_i}{\lambda_1})^k$ will tend to $0$ when $k$ goes to infinity. Therefore, by $(5)$, $\mid\mid\vec e^k\mid\mid$ converges to zero if and only if $\mid \lambda_1\mid <1$.
# 
# Denote $\rho(M):=\mid \lambda_1\mid=max\{\mid\lambda_k\mid: \lambda_k \text{ is the eigenvalue of }M\}$. We call $\rho(M)$ the **spectral radius** of $M$. The necessary and sufficient condition for $M^k$ to converge is  $\rho(M)<1$.
# 

# ## Estimating spectral radius $\rho(M)$
# 
# We know that the spectral radius $\rho(M)$ is so important, but how can we determine $\rho(M)<1$ or not? One can always compute all the eigenvalues by brute force and pick the largest one. One can also use more sophiscated method (e.g.: Power method, QR method) to find the largest eigenvalue in maginitude -- we will discuss it in the next chapter. Alternatively, we do have other way to determine the range of $\rho(M)=\mid\lambda_1\mid$. By the following theorem, we cannot identify the exact value of $\rho(M)$, but we can find an upper bound of it, which will also be helpful to our convergence analysis in some cases.

# ### Theorem 1 (Gershgorin Circle Theorem)
# 
# Given a matrix $A=(a_{ij})_{1\leq i,j\leq n}\in M_{n\times n}(\mathbb C)$, we consider an eigenvector $\vec e=(e_1,\cdots,e_n)^T$ with eigenvalue $\lambda$. Let $l$ be the index such that $e_l$ is the largest in magnitude of $\vec e$, i.e. $\mid e_l\mid \geq \mid e_j\mid$ for all $j$. Then
# 
# $$
# \lambda\in \overline{B_{a_{ll}}(\sum_{j=1,j\neq l}^n \mid a_{lj}\mid)}.
# $$
# 
# 
# 
# Then $A\vec e=\lambda \vec e$.
# 
# For each $1\leq i\leq n$, the $i$-th entry of $A\vec e=\lambda \vec e$ is
# 
# $$
# \begin{align*}
# \sum_{j=1}^n a_{ij}e_j&=\lambda e_i\\
# \iff a_{ii}e_i+\sum_{j=1,j\neq i}^n a_{ij}e_j&=\lambda e_i\\
# \iff \mid(a_{ii}-\lambda)e_i\mid&=\mid -\sum_{j=1,j\neq i}^n a_{ij}e_j\mid\\
# \iff \mid a_{ii}-\lambda\mid \cdot \mid e_i \mid &\leq \sum_{j=1,j\neq i}^n \mid a_{ij}\mid \cdot \mid e_j\mid 
# \end{align*}
# $$
# 
# Let $l$ be the index such that $e_l$ is the largest in magnitude of $\vec e$, i.e. $\mid e_l\mid \geq \mid e_j\mid$ for all $j$, then
# 
# $$
# \begin{align*}
# &\mid a_{ll}-\lambda\mid \cdot \mid e_l \mid \leq \sum_{j=1,j\neq l}^n \mid a_{lj}\mid \cdot \mid e_j\mid \leq \sum_{j=1,j\neq l}^n \mid a_{lj}\mid \cdot \mid e_l\mid\\
# &⇒ \mid a_{ll}-\lambda\mid \leq \sum_{j=1,j\neq l}^n \mid a_{lj}\mid\\
# &⇒ \lambda\in \overline{B_{a_{ll}}(\sum_{j=1,j\neq l}^n \mid a_{lj}\mid)}
# \end{align*}
# $$
# 
# 
# From above we know that $\lambda$ lies in a ball of center $a_{ll}$, radius $\sum_{j=1,j\neq l}^n \mid a_{lj}\mid$. It seems that we are done in finding the bound of $\lambda$, but we are not. Note that if we have already known $l$, we can determine the ball by looking into the $l$-th row of $A$, then the bound of $\lambda$ is found. However, the problem is that we do NOT know $l$. To find $l$, we should
# 
# * find eigenvector $\vec e$, and
# * find the largest component $e_l$ in magnitude,
# 
# but $\vec e$ is exactly what we are seeking for. It's a tautology.
# 
# Fortunately, we can play a trick by releasing the requirement in equation $(6)$: we take the **union** of all balls $B_{a_{ii}},\ i=1,\cdots,n$, then
# 
# $$
# \lambda\in  \bigcup_{i=1}^n \overline{B_{a_{ii}}(\sum_{j=1,j\neq i}^n \mid a_{ij}\mid)} \tag{7}
# $$

# Theorem 1 leads to an important property of matrix.
# 
# ### Strictly Dominant Diagonal (SDD) Matrix
# 
# ````{prf:definition} SDD Matrix
# :label: def_SDD
# 
# A matrix $A=(a_{ij})_{1\leq i,j\leq n}\in M_{n\times n}(\mathbb C)$ is strictly dominant diagonal (SDD) if
# $$
# \mid a_{ii}\mid  > \sum_{j=1,j\neq i}^n \mid a_{ij}\mid ,\qquad \forall i=1,\cdots,n.
# $$
# 
# ````
# 
# 
# ````{prf:theorem}
# :label: thm_SDD
# Obviously, an SDD matrix $A$ must be non-singular. A quick proof is as follows.
# ````
# 
# ````{prf:proof}
# :label: pf_SDD
# By Theorem 1, all eigenvalues of $A$ satisfy
# 
# $$
# \lambda\in  \bigcup_{i=1}^n \overline{B_{a_{ii}}(\sum_{j=1,j\neq i}^n \mid a_{ij}\mid)} \subset \bigcup_{i=1}^n \overline{B_{a_{ii}}(\mid a_{ii} \mid)}.
# $$
# 
# Therefore, every ball $\overline{B_{a_{ii}}(\sum_{j=1,j\neq i}^n \mid a_{ij}\mid)}$ must NOT touch 0, which implies that NO eigenvalue is 0. So $A$ is non-singular.
# 
# ````
# 
# SDD helps us to study the convergence of iterative method in some situation. We will discuss it later. Now let's introduce some concrete algorithms which utilize iterative methods.

# # Different splitting methods
# 
# Look back to our splitting method again:
# 
# $$
# A=N-(N-A)=N-P.
# $$
# 
# Of course, the splitting method itself is not random, so we need our matrix $N$ has some desirable properties:
# 
# 1. $N$ should be related to $A$, otherwise, it won't reduce the computation cost;
# 2. $N$ should be simple;
# 3. $N$ should have an inverse and $N^{-1}$ is easy to compute;
# 4. The spectral radint of $M=N^{-1}P$ should be small to ensure convergence.
# 
# There are many possible choices of $N$. Here I will introduce three most popular choices, making up three common iterative methods to solve large linear system: Jacobi mathod, Gauss-Seidal method, and SOR method.

# ## Jacobi Method
# 
# Jacobi Method is the simplest and the most intuitive. We choose 
# $$
# N=D,
# $$
# where matrix $D$ is a diagonal component of $A$.
# 
# Therefore, we split $A=N-P=D-(D-A)$, and get an iterative scheme as
# 
# $$
# D\vec{x}^{n+1}=(D-A)\vec{x}^n+\vec{f},\qquad\text{or}\qquad\vec{x}^{n+1}=D^{-1}(D-A)\vec{x}^n+D^{-1}\vec{f}.
# $$
# 

# ### Code demo

# In[1]:


from pprint import pprint
from numpy import array, zeros, diag, diagflat, dot
from numpy import linalg

def jacobi(A, b, tol=10**(-7), N=25, x=None):
    """Solves the equation Ax=b via the Jacobi iterative method."""
    # Create an initial guess if needed                                                                                                                                                            
    if x is None:
        x = zeros(len(A[0]))

    # Create a vector of the diagonal elements of A                                                                                                                                                
    # and subtract them from A                                                                                                                                                                     
    D = diag(A)
    R = A - diagflat(D)

    # Iterate for N times                                                                                                                                                                          
    for i in range(N):
        x_iter = (b - dot(R,x)) / D
        tol_iter = linalg.norm(x_iter-x)
        if(tol_iter < tol): 
          x = x_iter
          break
        x = x_iter
    return x, i, tol_iter


# Consider the linear system
# 
# $$
# \begin{pmatrix}
# 5 & -2 & 3\\
# -3 & 9 & 1\\
# 2 & -1 & -7\\
# \end{pmatrix}
# \begin{pmatrix}
# x_1 \\
# x_2 \\
# x_3 
# \end{pmatrix}=
# \begin{pmatrix}
# -1 \\
# 2 \\
# 3 
# \end{pmatrix}
# $$
# 
# Jacobi method starts from $\vec x^0=(0,0,0)^T$.

# In[2]:


A = array([[5,-2,3],[-3,9,1],[2,-1,-7]])
pprint(A)
b = array([-1,2,3])
pprint(b)
guess = array([0,0,0])


# In[3]:


sol_jacobi = jacobi(A,b,N=25,x=guess)
pprint(sol_jacobi)


# From the code above, after 13 iterations, Jacobi converges under the tolerence threshold $10^{-7}$, giving result $\vec x^{13}=(0.186,0.331,-0.423)^T$.

# ### Convergence of Jacobi
# 
# 
# 
# To analyse the convergence of Jacobi method, we should study the spectral radius of 
# 
# $$
# M=N^{-1}P=D^{-1}(D-A).
# $$
# 
# For the above example,
# 
# $$
# A=\begin{pmatrix}
# 5 & -2 & 3\\
# -3 & 9 & 1\\
# 2 & -1 & -7\\
# \end{pmatrix}
# =\begin{pmatrix}
# 5 & 0 & 0\\
# 0 & 9 & 0\\
# 0 & 0 & -7\\
# \end{pmatrix}+
# \begin{pmatrix}
# 0 & 2 & -3\\
# 3 & 0 & -1\\
# -2 & 1 & 0\\
# \end{pmatrix}
# =N-P,
# $$
# 
# then we have 
# $$
# M=N^{-1}P=
# \begin{pmatrix}
# 0 & 2/5 & -3/5\\
# 1/3 & 0 & -1/9\\
# 2/7 & -1/7 & 0\\
# \end{pmatrix},
# $$
# 
# where $\rho(M)=0.267<1$, so Jacobi must converge.
# 
# Instead of studying $\rho(M)$, we could derive the convergence directly by special matrix property of $A$ (Theorem 2).
# 

# 
# 
# #### Theorem 2
# 
# > If A is SDD, then Jacobi method to solve $A\vec x=\vec f$ converges.
# 
# *Proof*. Let $A=(a_{ij})_{1\leq i,j\leq n}$.
# 
# Jacobi method: 
# 
# $$
# \begin{align*}
# D\vec{x}^{n+1}&=(D-A)\vec{x}^n+\vec{f}\\
# ⇒ a_{ii}x_i^{n+1}&=-\sum_{j=1,j\neq i}^n a_{ij} x_j^n + f_i,\qquad i=1,\cdots,n\\
# ⇒ x_i^{n+1}&=-\frac{1}{a_{ii}}\sum_{j=1,j\neq i}^n a_{ij} x_j^n + \frac{1}{a_{ii}} f_i,\qquad i=1,\cdots,n.\tag{8}
# \end{align*}
# $$
# 
# Let $x^*$ be the solution of $A\vec x=\vec f$ (this $x^*$ always exists because A is SDD hence nonsingular), then
# 
# 
# $$
# x_i^{*}=-\frac{1}{a_{ii}}\sum_{j=1,j\neq i}^n a_{ij} x_j^* + \frac{1}{a_{ii}} f_i,\qquad i=1,\cdots,n.\tag{9}
# $$
# 
# By $(8)-(9)$, we have
# $$
# \begin{align*}
# e_i^{n+1}&=-\frac{1}{a_{ii}}\sum_{j=1,j\neq i}^n a_{ij} e_j^n\\
# ⇒\mid e_i^{n+1} \mid&<\sum_{j=1,j\neq i}^n \frac{\mid a_{ij}\mid}{\mid a_{ii}\mid} \mid e_j^n\mid \leq \sum_{j=1,j\neq i}^n \frac{\mid a_{ij}\mid}{\mid a_{ii}\mid} \|\vec e^n\|_{∞},
# \end{align*}
# $$
# 
# where $ \|\vec e^n\|_{∞}=\max_j \{\mid e_j^n\mid\}$.
# 
# Note that because $A$ is SDD, 
# 
# $$
# r=\max_i\{\sum_{j=1,j\neq i}^n \frac{\mid a_{ij}\mid}{\mid a_{ii}\mid}\}<1.
# $$
# 
# Therefore,
# 
# $$
# \mid e_i^{n+1} \mid < r\|\vec e^n\|_{∞},\qquad i=1,\cdots,n.\tag{9}
# $$
# 
# As $(9)$ is true for all $i$, so
# 
# $$
# \|\vec e^n\|_{∞}<r\|\vec e^n\|_{∞}.\tag{10}
# $$
# 
# By $(10)$, and $r<1$, we know that $\vec e^n\to 0$ as $n\to ∞$. Jacobi method converges.

# ## Gauss-Seidal Method
# 
# Gauss-Seidal Method uses another splitting rule. We choose 
# $$
# N=L+D,
# $$
# where matrix $D$ is a diagonal component of $A$, and $L$ is a lower triangular component of $A$.
# 
# Remark: any matrix $A$ can be splitted into 
# 
# $$
# A=L+D+U=\text{lower triangular}+\text{diagonal}+\text{upper triangular}.
# $$
# 
# Therefore, we split $A=N-P=(L+D)-(-U)$, and get an iterative scheme as
# 
# $$
# (L+D)\vec{x}^{n+1}=-U\vec{x}^n+\vec{f},\qquad\text{or}\qquad\vec{x}^{n+1}=-(L+D)^{-1}U\vec{x}^n+(L+D)^{-1}\vec{f}.
# $$

# ### Code demo

# In[4]:


from pprint import pprint
from numpy import array, zeros, diag, diagflat, dot, triu, tril
from numpy import linalg

def gaussSeidal(A, b, tol=10**(-7), N=25, x=None):
    """Solves the equation Ax=b via the Gauss-Seidal iterative method."""
    # Create an initial guess if needed                                                                                                                                                            
    if x is None:
        x = zeros(len(A[0]))                                                                                                                                                                   

    U = triu(A, 1)
    R = tril(A) # L+D

    # Iterate for N times                                                                                                                                                                          
    for i in range(N):
        x_iter = linalg.inv(R) @ (b - U@x) 
        tol_iter = linalg.norm(x_iter-x)
        x = x_iter
        if(tol_iter < tol): 
          break
    return x, i, tol_iter


# Consider the linear system (same as the example in Jacobi)
# 
# $$
# \begin{pmatrix}
# 5 & -2 & 3\\
# -3 & 9 & 1\\
# 2 & -1 & -7\\
# \end{pmatrix}
# \begin{pmatrix}
# x_1 \\
# x_2 \\
# x_3 
# \end{pmatrix}=
# \begin{pmatrix}
# -1 \\
# 2 \\
# 3 
# \end{pmatrix}
# $$
# 
# Gauss-Seidal method starts from $\vec x^0=(0,0,0)^T$.

# In[5]:


A = array([[5,-2,3],[-3,9,1],[2,-1,-7]])
pprint(A)
b = array([-1,2,3])
pprint(b)
guess = array([0,0,0])


# In[6]:


sol_gaussSeidal = gaussSeidal(A,b,N=25,x=guess)
pprint(sol_gaussSeidal)


# From the code above, after 8 iterations, Gauss-Seidal converges under the tolerence threshold $10^{-7}$, giving result $\vec x^{13}=(0.186,0.331,-0.423)^T$.

# ### Convergence of Gauss-Seidal
# 
# To analyse the convergence of Gauss-Seidal method, we should study the spectral radius of 
# 
# $$
# M=N^{-1}P=(L+D)^{-1}(-U).
# $$
# 
# 
# For the above example,
# 
# $$
# A=\begin{pmatrix}
# 5 & -2 & 3\\
# -3 & 9 & 1\\
# 2 & -1 & -7\\
# \end{pmatrix}
# =\begin{pmatrix}
# 5 & 0 & 0\\
# -3 & 9 & 0\\
# 2 & -1 & -7\\
# \end{pmatrix}+
# \begin{pmatrix}
# 0 & 2 & -3\\
# 0 & 0 & -1\\
# 0 & 0 & 0\\
# \end{pmatrix}
# =N-P,
# $$
# 
# then we have 
# $$
# M=N^{-1}P=
# \begin{pmatrix}
# 0 & 2/5 & -3/5\\
# 0 & 2/15 & -14/45\\
# -1/21 & 1/63 & 1/7\\
# \end{pmatrix},
# $$
# 
# where $\rho(M)=0.205<1$, so Gauss-Seidal must converge.
# 
# 
# 
# Similar to Theorem 2 in Jacobi method, instead of studying $\rho(M)$, we could derive the convergence directly by special matrix property of $A$ (Theorem 3).

# 
# 
# #### Theorem 3
# 
# > If A is SDD, then Gauss-Seidal method to solve $A\vec x=\vec f$ converges.
# 
# *Proof*. Let $A=(a_{ij})_{1\leq i,j\leq n}$.
# 
# Gauss-Seidal:
# 
# $$
# \begin{align*}
# (L+D)\vec{x}^{n+1}&=-U\vec{x}^n+\vec{f}\\
# ⇒ a_{ii}x_i^{n+1}&=-\sum_{j=1}^{i-1} a_{ij} x_j^{n+1} - \sum_{j=i+1}^{n} a_{ij} x_j^n + f_i,\qquad i=1,\cdots,n \tag{11}
# \end{align*}
# $$
# 
# 
# Let $x^*$ be the solution of $A\vec x=\vec f$ (this $x^*$ always exists because A is SDD hence nonsingular), then
# 
# 
# $$
# ⇒ a_{ii}x_i^*=-\sum_{j=1}^{i-1} a_{ij} x_j^* - \sum_{j=i+1}^{n} a_{ij} x_j^* + f_i,\qquad i=1,\cdots,n \tag{12}
# $$
# 
# By $(11)-(12)$, we have
# 
# $$
# \begin{align*}
# a_{ii}e_i^{n+1}&=-\sum_{j=1}^{i-1} a_{ij} e_j^{n+1} - \sum_{j=i+1}^{n} a_{ij}e_j^n\\
# e_i^{n+1}&=-\sum_{j=1}^{i-1} \frac{a_{ij}}{a_{ii}} e_j^{n+1} - \sum_{j=i+1}^{n} \frac{a_{ij}}{a_{ii}}e_j^n\tag{13}
# \end{align*}
# $$
# 
# Now, we want to prove: 
# $$
# \|\vec e^{n+1}\|_{∞}\leq r \|\vec e^{n}\|_{∞}\qquad \text{for}\qquad r=\max_i\{\sum_{j=1,j\neq i}^n \frac{\mid a_{ij}\mid}{\mid a_{ii}\mid}\}<1,
# $$
# which is equivalent to prove that
# 
# $$
# \mid e_i^{n+1}\mid\leq r \|\vec e^{n}\|_{∞},\qquad i=1,\cdots,n.\tag{14}
# $$
# 
# We prove $(14)$ by induction.
# 
# * For $i=1$:
# 
# $$
# \begin{align*}
# e_1^{n+1}&= \sum_{j=2}^{n} \frac{a_{1j}}{a_{11}}e_j^n\\
# ⇒ \mid e_1^{n+1}\mid &\leq - \sum_{j=2}^{n} \frac{|a_{1j}|}{|a_{11}|}|e_j^n|\leq \|\vec e^{n}\|_{∞}\cdot \sum_{j=2}^{n} \frac{|a_{1j}|}{|a_{11}|}\leq r \|\vec e^{n}\|_{∞}\qquad \text{for}\qquad i=1.
# \end{align*}
# $$
# 
# * Assume $\mid e_j^{n+1}\mid\leq r \|\vec e^{n}\|_{∞}$ for $j=1,\cdots,i-1$, then
# 
# $$
# \begin{align*}
# |e_i^{n+1}|&=\sum_{j=1}^{i-1} \frac{|a_{ij}|}{|a_{ii}|} |e_j^{n+1}| + \sum_{j=i+1}^{n} \frac{|a_{ij}|}{|a_{ii}|}|e_j^n|\\
# |e_i^{n+1}|&\leq r \|\vec e^{n}\|_{∞}\sum_{j=1}^{i-1} \frac{|a_{ij}|}{|a_{ii}|}+\|\vec e^{n}\|_{∞} \sum_{j=i+1}^{n} \frac{|a_{ij}|}{|a_{ii}|}\\
# &\leq \|\vec e^{n}\|_{∞}\{\sum_{j=1,j\neq i}^{n} \frac{|a_{ij}|}{|a_{ii}|}\}\\
# &\leq r \|\vec e^{n}\|_{∞}
# \end{align*}
# $$
# 
# Proof for $(14)$ is completed. Hence, 
# 
# $$
# \|\vec e^{n+1}\|_{∞}\leq r \|\vec e^{n}\|_{∞} \leq r^{n+1} \|\vec e^{0}\|_{∞}\to 0 \qquad \text{as}\qquad n\to ∞.
# $$
# 
# So Gauss-Seidal converges.
# 
# 
# 

# ## Successive over-relaxation Method
# 
# Recall in G-S method, $A=L+D+U$, so
# 
# $$
# L\vec x^{k+1}+ D\vec x^{k+1}+ U\vec x^{k} =\vec f.
# $$
# 
# SOR method makes a samll modification when updating $x^{k+1}$:
# $$
# \begin{align*}
# &L\vec x^{k+1}+ D\vec y^{k+1}+ U\vec x^{k} =\vec f\\
# &\vec x^{k+1} =\vec x^{k} + \omega (\vec y^{k+1}-\vec x^{k})
# \end{align*}
# $$
# 
# Then $\vec y^{k+1}=\frac{1}{\omega}(\vec x^{k+1}+(\omega-1)\vec x^{k})$, and 
# $$
# \begin{align*}
# &L\vec x^{k+1}+ \frac{1}{\omega}D\vec (\vec x^{k+1}+(\omega-1)\vec x^{k})+ U\vec x^{k} =\vec f\\
# &\implies (L+\frac{1}{\omega}D) \vec x^{k+1} = [\frac{1}{\omega}D-(D+U)] \vec x^{k} +\vec f.
# \end{align*}
# $$
# 
# Actually, SOR Method is similar to G-S method. We choose 
# $$
# N=L+\frac{1}{\omega}D.
# $$
# When $\omega=1$, SOR degenerates to G-S method.
# 
# Remark: any matrix $A$ can be splitted into 
# 
# $$
# A=L+D+U=\text{lower triangular}+\text{diagonal}+\text{upper triangular}.
# $$
# 
# Therefore, we split $A=N-P=(L+\frac{1}{\omega}D)-(\frac{1}{\omega}D-(D+U))$, and get an iterative scheme as
# 
# $$
# N\vec{x}^{n+1}=P\vec{x}^n+\vec{f},
# $$
# 
# where $N=(L+\frac{1}{\omega}D)$, $P=\frac{1}{\omega}D-(D+U)$.

# ### Code demo

# In[7]:


from pprint import pprint
from numpy import array, zeros, diag, diagflat, dot, triu, tril
from numpy import linalg

def SOR(A, b, w, tol=10**(-7), num=25, x=None):
    """Solves the equation Ax=b via the SOR iterative method."""
    # Create an initial guess if needed                                                                                                                                                            
    if x is None:
        x = zeros(len(A[0])) 

    D = diag(diag(A))
    N = tril(A,-1)+D/w # L+D/w
    P = D/w-(D+triu(A,1))

    # Iterate for num times                                                                                                                                                                          
    for i in range(num):
        x_iter = linalg.inv(N) @ (b + P@x) 
        tol_iter = linalg.norm(x_iter-x)
        x = x_iter
        if(tol_iter < tol): 
          break
    return x, i, tol_iter


# Consider the linear system (same as the example in Jacobi and Gauss-Seidal)
# 
# $$
# \begin{pmatrix}
# 5 & -2 & 3\\
# -3 & 9 & 1\\
# 2 & -1 & -7\\
# \end{pmatrix}
# \begin{pmatrix}
# x_1 \\
# x_2 \\
# x_3 
# \end{pmatrix}=
# \begin{pmatrix}
# -1 \\
# 2 \\
# 3 
# \end{pmatrix}
# $$
# 
# SOR method starts from $\vec x^0=(0,0,0)^T$.

# In[8]:


A = array([[5,-2,3],[-3,9,1],[2,-1,-7]])
pprint(A)
b = array([-1,2,3])
pprint(b)
guess = array([0,0,0])


# If we set $\omega=1$, then SOR behaves exactly the same as Gauss-Seidal:

# In[9]:


sol_SOR = SOR(A,b,w=1,num=25,x=guess)
pprint(sol_SOR)


# If we set $\omega=1.1$, $\omega=0.5$, ... then SOR also converges.

# In[10]:


sol_SOR = SOR(A,b,w=1.1,num=25,x=guess)
pprint(sol_SOR)


# In[11]:


sol_SOR = SOR(A,b,w=0.5,num=25,x=guess)
pprint(sol_SOR)


# However, if we choose $\omega=1.9$, SOR won't converge.

# In[12]:


sol_SOR = SOR(A,b,w=1.9,num=25,x=guess)
pprint(sol_SOR)


# The choice of $\omega$ will affect the convergence of SOR. We will discuss it now.

# ### Convergence of SOR
# 
# To analyse the convergence of SOR method, we should study the spectral radius of 
# 
# $$
# \begin{align*}
# M_{\omega}&=N^{-1}P=(L+\frac{1}{\omega}D)^{-1}(\frac{1}{\omega}D-(D+U))\\
# &=(\frac{1}{\omega}(\omega L+D))^{-1}(\frac{1}{\omega}(D-\omega(D+U)))\\
# &=(D+\omega L)^{-1}((1-\omega)D-\omega U)
# \end{align*}.
# $$
# 
# Note that $M_{\omega}$ depends on $\omega$. So we can adjust the value of $\omega$ to make $\rho(M_{\omega})$ as small as possible.
# 
# Firstly, there is a necessary condition for the convergence of SOR method.
# 

# 
# #### Theorem 4
# 
# > If SOR method to solve $A\vec x=\vec f$ converges, then $0<\omega<2$. (The converse is not true.)
# 
# *Proof*. Suppose SOR converges. Then $\rho(M_{\omega})=\max_i\{|\lambda_i|:\lambda_i\text{ is eigenvalue of }M_{\omega}\}<1$.
# 
# So $|det(M_{\omega})|=Π_{i=1}^n |\lambda_i|<[\rho(M_{\omega})]^n<1$. Now we look at $det(M_{\omega})$.
# 
# By direct calculation,
# 
# $$
# \begin{align*}
# det(M_{\omega})&=det[(D+\omega L)^{-1}((1-\omega)D-\omega U))]\\
# &=[det(D+\omega L)]^{-1}\cdot det[(1-\omega)D-\omega U]\\
# &=det(D^{-1})\cdot det[(1-\omega)D]\\
# &=det[(1-\omega)^nI_n]\\
# &=(1-\omega)^n.
# \end{align*}
# $$
# 
# Then $\rho(M_{\omega})<1⇒|det(M_{\omega})|<1 \iff |(1-\omega)|^n<1 \iff 0<\omega<2$.  Proof completed.
# 
# Note that $|det(M_{\omega})|<1\not⇒\rho(M_{\omega})<1$, so the converse is not true.
# 
# * Remark: To find the optimal $\omega_{opt}$ with smallest $\rho(M_{\omega})$, we solve the optimization problem
# 
# $$
# \min_{\omega} \lambda_i \qquad\text{s.t.} \qquad 0<\omega<2.
# $$

# Now we look at the sufficient condition for the convergence of SOR method.
# 
# #### Theorem 5
# 
# > If $A$ is SDD, and $0<\omega\leq1$, then SOR method to solve $A\vec x=\vec f$ converges.
# 
# *Proof.* Suppose  $A$ is SDD and $0<\omega\leq 1$. Need to show: $\rho(M_{\omega})<1$.
# 
# Proof by contradiction, we assume that $\exists \lambda$ such that $|\lambda|\geq 1$, where $\lambda$ is eigenvalue of $M_{\omega}$. 
# 
# Then
# 
# $$
# \begin{align*}
# &det(\lambda I-M_{\omega})=0\\
# &det[\lambda I-(D+\omega L)^{-1}((1-\omega)D-\omega U)]=0\\
# &det\{\lambda(D+\omega L)^{-1} [(D+\omega L)-\frac{1}{\lambda}((1-\omega)D-\omega U)]\}=0
# \end{align*}
# $$
# 
# Note that $\lambda\neq 0$, $det(D+\omega L)\neq 0$, so
# $$
# det(C)=det[(D+\omega L)-\frac{1}{\lambda}((1-\omega)D-\omega U)]=0.
# $$
# 
# Since $\omega(1-\frac{1}{|\lambda|})\leq 1-\frac{1}{|\lambda|}$, we have $1-\frac{1}{|\lambda|}(1-\omega)\geq \omega \tag{15}$
# 
# We write $C=(c_{ij})$ and $A=(a_{ij})$, then for each $1\leq i\leq n$, the diagonal part
# 
# $$
# \begin{align*}
# |c_{ii}|&=|1-\frac{1}{\lambda}(1-\omega)a_{ii}|\\
# &\geq [1-\frac{1}{|\lambda|}(1-\omega)] |a_{ii}|\\
# &\geq \omega |a_{ii}| \tag{by (15)}\\
# &>\omega\sum_{j=1,j\neq i}^n|a_{ij}| \tag{A is SDD}\\
# &=\omega\sum_{j=1}^{i-1}|a_{ij}| + \omega\sum_{j=i+1}^{n}|a_{ij}|\\
# &\geq \omega\sum_{j=1}^{i-1}|a_{ij}| + \frac{\omega}{|\lambda|}\sum_{j=i+1}^{n}|a_{ij}|\\
# &=\sum_{j=1,j\neq i}^n|c_{ij}|.
# \end{align*}
# $$
# 
# The last equation holds because the non-diagonal part of $C$ is $\omega L+\frac{\omega}{\lambda}U$.
# 
# Therefore, $C$ is SDD, so $C$ is non-singular, $det(C)\neq 0$, contradiction arises.
# 
# Hence $\forall |\lambda|<1$, that is $\rho(M_{\omega})<1$. Proof completed.
# 

# ### Optimal parameter selection
# 
# In SOR method, the spectral radius $\rho(M_{\omega})$ depends on $\omega$, now we introduce a theorem to find the optimal $\omega$ giving a smallest $\rho(M_{\omega})$.

# #### Consistently ordered Matrix
# > Let $A=L+D+U$. If the eigenvalues of $\alpha D^{-1}L+\frac{1}{\alpha}D^{-1}U\ (\alpha\neq 0)$ are independent of $\alpha$, then matrix $A$ is said to be consistently ordered. 
# 
# Examples of consistently ordered matrix:
# 
# * $$A=
# \begin{pmatrix}
# 10 & 1\\
# 1 & 10\\
# \end{pmatrix}\implies \alpha D^{-1}L+\frac{1}{\alpha}D^{-1}U=
# \begin{pmatrix}
# 0 & \frac{1}{10\alpha}\\
# -\frac{\alpha}{10} & 0\\
# \end{pmatrix}
# $$
# Char. poly: $\lambda^2-\frac{1}{100}=0\implies \lambda=\pm \frac{1}{10}$ are independent of $\alpha$.
# 
# * Tridiagonal matrix 
# \begin{pmatrix}
# \lambda_1 & * &  &  & \\
# * & \lambda_2 & * &  & \\
# & * & \lambda_3 & \ddots &\\
#  &  & * & \ddots & *\\
#  &  & & * & \lambda_n\\
# \end{pmatrix} is consistently ordered. 

# #### Theorem 6 (D. Young)
# Consider a linear system $A\vec x=\vec f$.
# 
# Assume that
# 1. $0<\omega<2$;
# 2. $M_J=N_J^{-1}P_J$ for Jacobi method has only real eigenvalues;
# 3. $\beta=\rho(M_J)<1$;
# 4. $A$ is consistently ordered.
# 
# Then:
# 1. $\rho(M_{SOR})<1$ (SOR converges).
# 2. Optimal parameter $\omega_{opt}$ for fastest convergence is $\omega_{opt}=\frac{2}{1+\sqrt{1-\beta^2}}$, and $\rho(M_{SOR},\omega_{opt})=\omega_{opt}-1$.

# Example.
# 
# Consider 
# $$
# A\vec x=
# \begin{pmatrix}
# 10 & 1\\
# 1 & 10\\
# \end{pmatrix}
# \begin{pmatrix}
# x_1 \\
# x_2 \\
# \end{pmatrix}=
# \begin{pmatrix}
# 12 \\
# 21 \\
# \end{pmatrix}
# =\vec f
# $$
# 
# We know that
# *  $M_J=N_J^{-1}P_J$ for Jacobi method has only real eigenvalues $\lambda=\pm \frac{1}{10}$;
# * $\beta=\rho(M_J)=\frac{1}{10}<1$;
# * $A$ is consistently ordered.
# 
# By Theorem 6, 
# 
# * $\rho(M_{SOR})<1$ (SOR converges);
# * Optimal parameter $\omega_{opt} =\frac{2}{1+\sqrt{1-\beta^2}}=\frac{2}{1+\sqrt{1-\frac{1}{100}}}=1.0025125$;
# * $\rho(M_{SOR},\omega_{opt})=\omega_{opt}-1=0.0025125$.
# 

# # Convergence of Iterative Method
# 
# Here we study the convergence of Iterative Method again from a more general perspective.

# ## Theorem 7 (Householder-John)
# 
# > Suppose $A$ and $(N^*+N-A)$ are self-adjoint and positive definite matrices.
# Then the iterative scheme $N\vec x^{(k+1)}=P\vec x^{(k)}+\vec b$ converges.
# 
# Remark (definitions review):
# 
# * $B$ is self-adjoint if $B^*:=\overline {B}^T=B$.
# * $B$ is positive definite if $\vec x^*B\vec x>0$ for $\forall \vec x\neq \vec 0, \vec x\in \mathbb{C}^n$.
# 
# 
# *Proof*.
# 
# Suppose $A$ and $(N^*+N-A)$ are self-adjoint and positive definite.
# 
# Condiser $M=N^{-1}P=N^{-1}(N-A)=I-N^{-1}A$. Need to show: all eigenvalues $\lambda$ of $M$ satisfy $|\lambda|<1$.
# 
# Let $\lambda$ be an eigenvalues of $M$. Then $M\vec x=\lambda \vec x,\ \vec x\neq \vec 0$, i.e.
# 
# $$
# \begin{align*}
# (I-N^{-1}A)\vec x&=\lambda \vec x\\
# (N-A)\vec x&=\lambda N\vec x\\
# (1-\lambda)N\vec x&=A\vec x\tag {16}
# \end{align*}
# $$
# 
# Note that $\lambda\neq 0$. (Otherwise, $A\vec x=0$, then $\vec x^*A\vec x=0$, contradicting to the fact that A is positive definite.)
# 
# From $(16)$, multiply $\vec x^*$ on both sides:
# $$
# \begin{align*}
# (1-\lambda)\vec x^*N\vec x&=\vec x^*A\vec x \tag{17}\\
# \vec x^*N\vec x&=\frac{1}{1-\lambda}\vec x^*A\vec x \tag{18}
# \end{align*}
# $$
# 
# Take conjugate transpose on both sides of $(17)$:
# 
# $$
# \begin{align*}
# [(1-\lambda)\vec x^*N\vec x]^*&=[\vec x^*A\vec x]^* \\
# (1-\overline{\lambda})\vec x^* N^* \vec x^{**}&= \vec x^*A^*\vec x^{**}\\
# (1-\overline{\lambda})\vec x^* N^* \vec x&= \vec x^*A\vec x\\
# \vec x^* N^* \vec x&=\frac{1}{1-\overline{\lambda}}\vec x^*A\vec x \tag{19}
# \end{align*}
# $$
# 
# Take $(18)+(19)-\vec x^* A \vec x$:
# 
# $$
# \vec x^* (N^*+N-A) \vec x = \{\frac{1}{1-\lambda}+\frac{1}{1-\overline{\lambda}}-1\} \vec x^*A\vec x
# $$
# 
# Since $\vec x \neq \vec 0$ is eigenvector, and both $(N^*+N-A)$, $A$ are positive definite, so $\vec x^* (N^*+N-A) \vec x>0, \vec x^*A\vec x>0$, which implies
# 
# $$
# \begin{align*}
# &\frac{1}{1-\lambda}+\frac{1}{1-\overline{\lambda}}-1=\frac{1-|\lambda|^2}{|1-\lambda|^2}>0\\
# &\implies 1-|\lambda|^2 > 0\\
# &\implies |\lambda|<1.
# \end{align*}
# $$
# 
# So the iterative scheme converges. Proof completed.
# 

# ## Examples for Householder-John

# ### Example 1
# 
# Consider a real, symmetric, positive definite (tri-diagonal) matrix
# 
# $$
# A=\begin{pmatrix}
# \alpha_1 & \beta_1 &  &  & \\
# \beta_1 & \alpha_2 & \beta_2 &  & \\
# & \beta_2 & \alpha_3 & \ddots &\\
#  &  & \ddots & \ddots & \beta_{n-1}\\
#  &  & & \beta_{n-1} & \alpha_n\\
# \end{pmatrix}.
# $$
# 
# Prove that Gauss-Seidal to solve $A\vec x=\vec b$ converges.
# 
# *Proof*. First, $A$ is self-adjoint and positive definite.
# 
# For Gauss-Seidal, 
# 
# $$
# N=L+D=\begin{pmatrix}
# \alpha_1 & 0 &  &  & \\
# \beta_1 & \alpha_2 & 0 &  & \\
# & \beta_2 & \alpha_3 & \ddots &\\
#  &  & \ddots & \ddots & 0\\
#  &  & & \beta_{n-1} & \alpha_n\\
# \end{pmatrix},
# $$
# 
# so
# 
# $$
# N^*+N-A=\begin{pmatrix}
# \alpha_1 & 0 &  &  & \\
# 0 & \alpha_2 & 0 &  & \\
# & 0 & \alpha_3 & \ddots &\\
#  &  & \ddots & \ddots & 0\\
#  &  & & 0 & \alpha_n\\
# \end{pmatrix}.
# $$
# 
# Obviously, $N^*+N-A$ is self-adjoint.
# 
# Now we need to prove that $N^*+N-A$ is positive definite.
# 
# $$
# \begin{align*}
# &N^*+N-A \text{ is positive definite}\\
# &\iff\alpha_1 x_1^2+\cdots+\alpha_n x_n^2>0,\qquad\forall \vec x=(x_1,\cdots,x_n)^T\\
# &\iff \alpha_1,\cdots,\alpha_n >0 \tag{20}
# \end{align*}
# $$
# 
# Let $\vec e_i=(0,0,\cdots,1,\cdots,0)^T$ where the $i$-th entry is 1, $1\leq i\leq n$.
# 
# Since $A$ is positive definite, $\vec e_i^*A\vec e_i=\alpha_i>0$ for all $1\leq i\leq n$, so $(20)$ is true. Therefore, $N^*+N-A$ is positive definite.
# 
# By Householder-John Theorem, Gauss-Seidal converges.
# 
# 

# ### Example 2
# 
# > Suppose $A$ is real symmetric positive-definite (SPD) matrix.
# Prove that SOR method to solve $A\vec x=\vec b$ converges iff $0<\omega<2$.
# 
# 
# *Proof*. Firstly, $A$ is self-adjoint and positive-definite. Now we consider (N^*+N-A).
# 
# In SOR method,
# 
# $N=\frac{1}{\omega}D+L$, where $L=U^*=U^T$ as $A$ is symmetric. Therefore,
# 
# $$
# N^*+N-A=[\frac{1}{\omega}D+U]+[\frac{1}{\omega}D+L]-A=(\frac{2}{\omega}-1)D.
# $$
# 
# Easy to observe that
# 
# * $N^*+N-A$ is self-adjoint.
# * $N^*+N-A$ is positive-definite iff $(\frac{2}{\omega}-1)>0$ iff $0<\omega<2$.
# 
# 
# By Householder-John Theorem, SOR converges iff $0<\omega<2$.
# 
