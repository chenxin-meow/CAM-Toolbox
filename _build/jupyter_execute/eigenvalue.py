#!/usr/bin/env python
# coding: utf-8

# # Eigenvalue Problems

# Given a linear transformation $f(x)=Ax$ mapping from $\mathbb C^n$ to $\mathbb C^m$, we are always interested in studying the eigenvalues and eigenvectors of $A$. This helps us to understand the characteristics of linear transformation. Moreover, the convergence analysis of iterative methods for solving linear system depends on the spectral radius of specific matrices, which also requires us to study the eigenvalues and eigenvectors. However, the analytical scheme to find eigenvalues needs to solve an $n$-th order polynomial characteristic equation. The computation is difficult when $A$ is a really large matrix. Therefore, applied mathematicians invent more sophisticated methods to solve the problem numerically. 
# 
# In this chapter, we will discuss two methods to solve the eigenvalue problem: Power method and QR method. Power method finds the largest dominant eigenvalue and its corresponding eigenvector, whereas QR method can find all the eigenvalues at the same time. These two methods, although look in huge difference, are the same from a more general perspective. We will discuss the relationship between Power method and QR method as well.

# ## Power Method
# 
# Consider a matrix $A\in M_{n\times n}(\mathbb C)$ with $n$ eigenvalues $\lambda_1, \lambda_2, \cdots, \lambda_n$ and corresponding eigenvectors $\vec x_1, \vec x_2, \cdots, \vec x_n$. Suppose that $A$ is diagonalizable, that is, $\vec x_1, \vec x_2, \cdots, \vec x_n$ form a basis in $\mathbb C^n$. Further assume that $|\lambda_1|> |\lambda_2|\geq  \cdots\geq |\lambda_n|$ (i.e. $\lambda_1$ strictly dominates all other eigenvalues). 
# 
# 
# 
# Given an initial guess  $\vec x^0=a_1\vec x_1+a_2\vec x_2+\cdots+a_n\vec x_n,\ a_1\neq 0$,  we consider an iterative scheme:
# 
# $$
# \vec x^{(k+1)}=\frac{A\vec x^{(k)}}{\|A\vec x^{(k)}\|_{\infty}}.
# $$
# 
# It follows that
# 
# $$
# \vec x^{(k)}=\frac{A\vec x^{(k-1)}}{\|A\vec x^{(k-1)}\|_{\infty}}=\frac{A\frac{A\vec x^{(k-2)}}{\|A\vec x^{(k-2)}\|_{\infty}}}{\|A\frac{A\vec x^{(k-2)}}{\|A\vec x^{(k-2)}\|_{\infty}}\|_{\infty}}=\frac{A^2\vec x^{(k-2)}}{\|A^2\vec x^{(k-2)}\|_{\infty}}=\cdots=\frac{A^k\vec x^{(0)}}{\|A^k\vec x^{(0)}\|_{\infty}}.
# $$
# 
# Since
# 
# $$
# \begin{align*}
# A^k\vec x^{(0)}&=a_1\lambda_1^k\vec x_1+a_2\lambda_2^k\vec x_2+\cdots+a_n\lambda_n^k\vec x_n\\
# &=a_1\lambda_1^k(\vec x_1+\sum_{j=2}^n\frac{a_j}{a_1}(\frac{\lambda_j}{\lambda_1})^k\vec x_j),
# \end{align*}
# $$
# 
# we have
# 
# $$
# \begin{align*}
# \vec x^{(k)}&=\frac{A^k\vec x^{(0)}}{\|A^k\vec x^{(0)}\|_{\infty}}\\
# &=\frac{a_1\lambda_1^k(\vec x_1+\sum_{j=2}^n\frac{a_j}{a_1}(\frac{\lambda_j}{\lambda_1})^k\vec x_j)}{\|a_1\lambda_1^k(\vec x_1+\sum_{j=2}^n\frac{a_j}{a_1}(\frac{\lambda_j}{\lambda_1})^k\vec x_j)\|_{\infty}}\\
# &\to \frac{a_1\lambda_1^k\vec x_1}{\|a_1\lambda_1^k\vec x_1\|_{\infty}}\qquad\qquad\text{as }k\to \infty\\
# &=\frac{a_1\lambda_1^k}{|a_1||\lambda_1^k|\|\vec x_1\|_{\infty}}\vec x_1.
# \end{align*}
# $$
# 
# Note that $\vec x_1$ is the eigenvector corresponding with the largest eigenvalue, and $\vec x^{(k)}$ converges to the direction of $\vec x_1$.
# 
# Moreover, observe that
# 
# $$
# \begin{align*}
# \|A\vec x^{(k)}\|_{\infty} &\to \| \frac{a_1\lambda_1^k}{|a_1||\lambda_1^k|\|\vec x_1\|_{\infty}}\cdot A\vec x_1\|_{\infty}\\
# &=\| \frac{1}{\|\vec x_1\|_{\infty}}\cdot \lambda_1\vec x_1\|_{\infty}\\
# &=|\lambda_1|\qquad\qquad\text{as }k\to \infty\\
# \end{align*}
# $$
# 
# that is, $\|A\vec x^{(k)}\|_{\infty}$ converges to the largest eigenvalue $|\lambda_1|$.
# 
# Now we can formulate the Power method as follows.

# ### Algorithm 1 (Power Method)
# Goal: find the largest eigenvalue in magnitude of $A$.
# 
# 1. Initialization: $\vec x^{(0)}=a_1\vec x_1+a_2\vec x_2+\cdots+a_n\vec x_n,\ a_1\neq 0$.
# 
# 2. For $k=1,2,\cdots$, repeat until convergence:
#    * Compute $\vec x^{(k)}=\frac{A\vec x^{(k-1)}}{\|A\vec x^{(k-1)}\|_{\infty}}.$
#    * Compute $\rho_k=\|A\vec x^{(k)}\|_{\infty}$.
# 
# Then $\rho_k\to |\lambda_1|$ as $k\to \infty$.
# 
# Remark: In fact, if $\lambda_1$ has multiplicity greater than 1, i.e.  $|\lambda_1|=\cdots=|\lambda_i|>|\lambda_{i+1}|\geq  \cdots\geq |\lambda_n|$, Power method also works, and the proof is similar to the above.

# ### Code demo

# In[1]:


import numpy as np
import random

def power(A,tol=10**(-7),N=100,x=None,seed=10):
  """Finding the dominated eigenvalue of A via the Power method."""

  random.seed(seed)

  # initilize x
  n=A.shape[1]
  x=np.random.rand(n,1)
  x=x/max(abs(x))
  rho=max(abs(A@x))

  # iterate for N times 
  for i in range(N):
    # power iteration
    x_iter = A@x/max(abs(A@x))
    rho_iter=max(abs(A@x_iter))
    x = x_iter

    # check for convergence
    tol_iter = abs(rho_iter-rho)
    if(tol_iter < tol): 
      rho = rho_iter
      break
    rho = rho_iter

  x = np.transpose(x)

  return x, rho, i, tol_iter


# Consider the matrix $A=\begin{pmatrix}
# 5 & -2 & 3\\
# -3 & 9 & 1\\
# 2 & -1 & -7\\
# \end{pmatrix}$.

# In[2]:


A = np.array([[5,-2,3],[-3,9,1],[2,-1,-7]])
sol_power=power(A)
print('The dominated eigenvalue = %0.4f' %(sol_power[1]))
print('The corresponding eigenvector = ', sol_power[0])
print('Iteration = ', sol_power[2])
print('Tolerence = ', sol_power[3])


# From the code above, the dominated eigenvalue is $|\lambda_1|=10.2271$.
# 
# 

# ### Power method in matrix form
# 
# We can also analyze the Power method in matrix form. This will help us to extend Power method to the case that $A$ is non-diagonalizable. But before that, let's re-visit the Power method when $A$ is diagonalizable.
# 
# Suppose $A\in M_{n\times n}(\mathbb C)$ has $n$ eigenvalues $\lambda_1, \lambda_2, \cdots, \lambda_n$ and corresponding eigenvectors $\vec x_1, \vec x_2, \cdots, \vec x_n$. Assume that $|\lambda_1|> |\lambda_2|\geq  \cdots\geq |\lambda_n|$.
# 
# Write $Q=(\vec x_1, \vec x_2, \cdots, \vec x_n)\in M_{n\times n}(\mathbb C)$, then $A$ can be diagonalized into
# 
# $$
# A=Q
# \begin{pmatrix}
# \lambda_1 & &\\
# &\ddots &\\
# & & \lambda_n
# \end{pmatrix}
# Q^{-1}.
# $$
# 
# The initial guess
# 
# $$
# \vec x^0=a_1\vec x_1+\cdots+a_n\vec x_n=Q\begin{pmatrix}a_1\\ \vdots\\ a_n\end{pmatrix},\ a_1\neq 0.
# $$

# We consider the iterative scheme in the Power method.
# 
# $$
# \begin{align*}
# A^k\vec x^{(0)}&=Q\begin{pmatrix}
# \lambda_1 & &\\
# &\ddots &\\
# & & \lambda_n
# \end{pmatrix}Q^{-1} Q\begin{pmatrix}a_1\\ \vdots\\ a_n\end{pmatrix}\\
# &=Q\begin{pmatrix}a_1\lambda_1^k\\ a_2\lambda_2^k\\ \vdots\\ a_n\lambda_n^k\end{pmatrix}=a_1\lambda_1^kQ \begin{pmatrix} 1\\ \frac{a_2}{a_1}(\frac{\lambda_2}{\lambda_1})^k\\ \vdots\\ \frac{a_n}{a_1}(\frac{\lambda_n}{\lambda_1})^k\end{pmatrix}\\
# &\to a_1\lambda_1^kQ \begin{pmatrix}1\\ 0\\ \vdots\\ 0\end{pmatrix} \qquad\text{as }k\to \infty\\
# &=a_1\lambda_1^k \vec x_1.
# \end{align*}
# $$
# 
# Then, 
# 
# $$
# \begin{align*}
# \vec x^{(k)}=\frac{A^k\vec x^{(0)}}{\|A^k\vec x^{(0)}\|_{\infty}}\to\frac{a_1\lambda_1^k}{|a_1||\lambda_1^k|\|\vec x_1\|_{\infty}}\vec x_1,
# \end{align*}
# $$
# 
# $$
# \begin{align*}
# \|A\vec x^{(k)}\|_{\infty} &\to \| \frac{a_1\lambda_1^k}{|a_1||\lambda_1^k|\|\vec x_1\|_{\infty}}\cdot A\vec x_1\|_{\infty}\\
# &=\| \frac{1}{\|\vec x_1\|_{\infty}}\cdot \lambda_1\vec x_1\|_{\infty}\\
# &=|\lambda_1|\qquad\text{as }k\to \infty.\\
# \end{align*}
# $$

# We have discussed the case that $A$ is diagonalizable. When $A$ is non-diagonalizable, Power method also works well, but we need a different formulation of $A$'s diagonalization.
# 
# In fact, whether $A$ is diagonalizable or not, we can always "diagonalize" $A$ into
# 
# $$
# A=QJQ^{-1},
# $$
# 
# where $Q=(\vec x_1, \vec x_2, \cdots, \vec x_n)$. Here $\vec x_1, \cdots, \vec x_n$ are the basis of the generalized eigenvecor with eigenvalues $\lambda_1,\cdots,\lambda_n$ ; and $J$ is the **Jordan normal form** (block diagonal matrix)
# 
# $$
# J=\begin{pmatrix}
# \lambda_1 & & &\\
# &[J(\lambda_{i_1})]& & &\\
# & & \ddots &\\
# & & &[J(\lambda_{i_l})]
# \end{pmatrix},
# $$
# 
# and the Jordan block can be written as
# 
# $$
# J(\lambda_{i_j})=\begin{pmatrix}
# \lambda_{i_j} & 1& & &\\
# &\lambda_{i_j}& 1& & &\\
# & & \ddots & \ddots\\
# & & & \ddots&1\\
# & & & &\lambda_{i_j}
# \end{pmatrix}.
# $$
# 
# 
# 
# The initial guess
# 
# $$
# \vec x^0=a_1\vec x_1+\cdots+a_n\vec x_n=Q\begin{pmatrix}a_1\\ \vdots\\ a_n\end{pmatrix},\ a_1\neq 0.
# $$
# 
# We consider the iterative scheme 
# 
# $$
# \vec x^{(k)}=\frac{A^k\vec x^{(0)}}{\|A^k\vec x^{(0)}\|_{\infty}}
# $$
# 
# in the Power method. Then
# 
# 
# 
# 

# $$
# \begin{align*}
# A^k\vec x^{(0)}&=QJ^kQ^{-1}Q\begin{pmatrix}a_1\\ \vdots\\ a_n\end{pmatrix}=QJ^k\begin{pmatrix}a_1\\ \vdots\\ a_n\end{pmatrix}\\
# &=Q\begin{pmatrix}
# \lambda_1^k & & &\\
# &[J(\lambda_{i_1})]^k& & &\\
# & & \ddots &\\
# & & &[J(\lambda_{i_l})]^k
# \end{pmatrix}
# \begin{pmatrix}a_1\\ a_2 \\ \vdots\\ a_n\end{pmatrix}\\
# &=Q\begin{pmatrix}
# a_1\lambda_1^k \\
# \begin{pmatrix}
# [J(\lambda_{i_1})]^k & &\\
# &\ddots&\\
# & & [J(\lambda_{i_l})]^k
# \end{pmatrix}
# \begin{pmatrix} a_2 \\ \vdots\\ a_n\end{pmatrix}
# \end{pmatrix}\\
# &=a_1\lambda_1^kQ
# \begin{pmatrix}
# 1 \\
# \frac{1}{a_1}\begin{pmatrix}
# [\frac{J(\lambda_{i_1})}{\lambda_1}]^k & &\\
# &\ddots&\\
# & & \frac{J(\lambda_{i_l})}{\lambda_1}]^k
# \end{pmatrix}
# \begin{pmatrix} a_2 \\ \vdots\\ a_n\end{pmatrix}
# \end{pmatrix}\\
# &\to a_1\lambda_1^kQ \begin{pmatrix}1\\ 0\\\vdots\\ 0\end{pmatrix} \qquad\text{as }k\to \infty\\
# &=a_1\lambda_1^k \vec x_1.
# \end{align*}
# $$

# Therefore, 
# 
# $$
# \begin{align*}
# \vec x^{(k)}=\frac{A^k\vec x^{(0)}}{\|A^k\vec x^{(0)}\|_{\infty}}\to \frac{a_1\lambda_1^k}{|a_1||\lambda_1^k|\|\vec x_1\|_{\infty}}\vec x_1,
# \end{align*}
# $$
# 
# Hence $\vec x^{(k)}$ converges to the direction of $\vec x_1$, the eigenvector corresponding with the largest eigenvalue, and 
# 
# $$
# \begin{align*}
# \|A\vec x^{(k)}\|_{\infty} &\to \| \frac{a_1\lambda_1^k}{|a_1||\lambda_1^k|\|\vec x_1\|_{\infty}}\cdot A\vec x_1\|_{\infty}\\
# &=\| \frac{1}{\|\vec x_1\|_{\infty}}\cdot \lambda_1\vec x_1\|_{\infty}\\
# &=|\lambda_1|\qquad\qquad\text{as }k\to \infty,\\
# \end{align*}
# $$
# 
# that is, $\|A\vec x^{(k)}\|_{\infty}$ converges to the largest eigenvalue $|\lambda_1|$.

# ### Inverse Power Method
# 
# Power method is powerful. Besides finding the largest eigenvalue, it can also be extended to get the smallest eigenvalue. 
# 
# 
# Consider a matrix $A$ with eigenvalues $|\lambda_1|\geq  |\lambda_2|\geq  \cdots\geq |\lambda_{n-1}|> |\lambda_n|$. Then its inverse $A^{-1}$ has eigenvalues $|\frac{1}{\lambda_n}|>|\frac{1}{\lambda_{n-1}}|\geq \cdots |\frac{1}{\lambda_1}|$. Intuitively, we can apply the Power method on $A^{-1}$ to get the largest eigenvalue $|\frac{1}{\lambda_n}|$ of $A^{-1}$, then we find the smallest eigenvalue $|\lambda_n|$ of$A$.
# 
# However, directly computing $A^{-1}$ is hard. Instead of computing $A^{-1}$ explicitly, in each iteration, we solve $A\vec y=\vec x^{(k-1)}$ to determine $A^{-1} x^{(k-1)}$.

# #### Algorithm 2 (Inverse Power Method)
# Goal: find the smallest eigenvalue in magnitude of $A$.
# 
# 1. Initialization: Pick $\vec x^{(0)}$ (usually with $\|\vec x^{(0)}\|_{\infty}=1$).
# 
# 2. For $k=1,2,\cdots$, repeat until convergence:
#    * Solve $A\vec y=\vec x^{(k-1)}$.
#    * Compute $\vec x^{(k)}=\frac{\vec y}{\|\vec y\|_{\infty}}.$
#    * Compute $\rho_k=\|A\vec x^{(k)}\|_{\infty}$.
# 
# Then $\rho_k\to |\lambda_n|$ as $k\to \infty$.

# #### Code demo

# In[3]:


def inverse_power(A,tol=10**(-7),N=100,x=None,seed=10):
  """Finding the smallest eigenvalue of A via the Inverse Power method."""
  
  random.seed(seed)

  # initilize x
  n=A.shape[1]
  x=np.random.rand(n,1)
  x=x/max(abs(x))
  rho=max(abs(A@x))

  # iterate for N times 
  for i in range(N):
    # inverse power iteration
    y=np.linalg.solve(A,x)
    x_iter = y/max(abs(y))
    rho_iter=max(abs(A@x_iter))
    x = x_iter

    # check for convergence
    tol_iter = abs(rho_iter-rho)
    if(tol_iter < tol): 
      rho = rho_iter
      break
    rho = rho_iter

  x = np.transpose(x)

  return x, rho, i, tol_iter


# Consider the matrix $A=\begin{pmatrix}
# 5 & -2 & 3\\
# -3 & 9 & 1\\
# 2 & -1 & -7\\
# \end{pmatrix}$.

# In[4]:


sol_inverse_power=inverse_power(A)
print('The smallest eigenvalue = %0.4f' %(sol_inverse_power[1]))
print('The corresponding eigenvector = ', sol_inverse_power[0])
print('Iteration = ', sol_inverse_power[2])
print('Tolerence = ', sol_inverse_power[3])


# From the code above, the smallest eigenvalue is $|\lambda_n|=4.1830$.

# 
# ### Inverse Power Method with Shift
# 
# Other than the smallest/largest eigenvalues, sometimes we are also interested in eigenvalues in between. Given a constant $\mu$, can we find the eigenvalue of $A$ that closet to $\mu$? The answer is very simple. 
# 
# Simply consider $B=A-\mu I$, then $B$ has eigenvalues $\lambda_1-\mu,\lambda_2-\mu,\cdots,\lambda_n-\mu$. We can apply the Inverse Power Method on $B$ to find the smallest $|\lambda_j-\mu|$. This method is called the Inverse Power Method with Shift.
# 

# 
# #### Algorithm 3 (Inverse Power Method with Shift)
# Goal: Take $\mu \in \mathbb R$, find the eigenvalue of $A$ that closet to $\mu$.
# 
# 1. Initialization: Pick $\vec x^{(0)}$ (usually with $\|\vec x^{(0)}\|_{\infty}=1$).
# 
# 2. For $k=1,2,\cdots$, repeat until convergence:
#    * Solve $(A-\mu I)\vec y=\vec x^{(k-1)}$.
#    * Compute $\vec x^{(k)}=\frac{\vec y}{\|\vec y\|_{\infty}}.$
#    * Compute $\rho_k=\|A\vec x^{(k)}\|_{\infty}$.
# 
# Then $\rho_k\to |\lambda_n|$ as $k\to \infty$.

# #### Code demo

# In[5]:


def inverse_power_shift(A,mu,tol=10**(-7),N=100,x=None,seed=10):
  """Finding the eigenvalue of A that closet to mu via the Inverse Power method with Shift."""
  
  random.seed(seed)

  # initilize x
  n=A.shape[1]
  x=np.random.rand(n,1)
  x=x/max(abs(x))
  rho=max(abs(A@x))

  # iterate for N times 
  for i in range(N):
    # inverse power iteration
    y=np.linalg.solve(A-mu*np.identity(n),x)
    x_iter = y/max(abs(y))
    rho_iter=max(abs(A@x_iter))
    x = x_iter

    # check for convergence
    tol_iter = abs(rho_iter-rho)
    if(tol_iter < tol): 
      rho = rho_iter
      break
    rho = rho_iter

  x = np.transpose(x)

  return x, rho, i, tol_iter


# Consider the matrix $A=\begin{pmatrix}
# 5 & -2 & 3\\
# -3 & 9 & 1\\
# 2 & -1 & -7\\
# \end{pmatrix}$. Suppose we want to find eigenvalue of $A$ that closet to $\mu=6$.

# In[6]:


sol_inverse_power_shift=inverse_power_shift(A,mu=6)
print('The eigenvalue that closet to mu = %0.4f' %(sol_inverse_power_shift[1]))
print('The corresponding eigenvector = ', sol_inverse_power_shift[0])
print('Iteration = ', sol_inverse_power_shift[2])
print('Tolerence = ', sol_inverse_power_shift[3])


# From the code above, the eigenvalue that closet to $\mu=6$ is $|\lambda|=4.1830$.
# 
# 

# 
# ### Convergence Analysis of Power Method 
# 
# In Power Method, when updating $\vec x^{(k)}$, we have
# 
# $$
# \vec x^{(k)}=
# \frac{a_1\lambda_1^k(\vec x_1+\sum_{j=2}^n\frac{a_j}{a_1}(\frac{\lambda_j}{\lambda_1})^k\vec x_j)}{\|a_1\lambda_1^k(\vec x_1+\sum_{j=2}^n\frac{a_j}{a_1}(\frac{\lambda_j}{\lambda_1})^k\vec x_j)\|_{\infty}}\to \frac{a_1\lambda_1^k\vec x_1}{\|a_1\lambda_1^k\vec x_1\|_{\infty}}\qquad\text{as }k\to \infty\\
# $$
# 
# Note that 
# 
# $$
# 1>|\frac{\lambda_2}{\lambda_1}|>|\frac{\lambda_3}{\lambda_1}|>\cdots>|\frac{\lambda_n}{\lambda_1}|,
# $$
# 
# The convergence rate depends on $\eta:=|\frac{\lambda_2}{\lambda_1}|$.
# 
# Now let's look into three Power Methods mentioned above. Suppose $A$ has eigenvalues $|\lambda_1|>|\lambda_2|>\cdots |\lambda_n|$ with corresponding eigenvectors $\vec v_1, \vec v_2, \cdots, \vec v_n$.
# 
# 1. Power Method
#    * Converges if $\eta:=|\frac{\lambda_2}{\lambda_1}|<1$ and $\langle \vec v_1,\vec x^{(0)}\rangle \neq 0$ ($\vec v_1$ is the eigenvector of $\lambda_1$).
#    * $\rho_k=\|A\vec x^{(k)}\|_{\infty}=|\lambda_1+\mathcal{O}(\eta^k)|$, slow convergence if $\eta \approx 1$.
# 2. Inverse Power Method
#    * Converges if $\eta:=|\frac{1/\lambda_{n-1}}{1/\lambda_n}|=|\frac{\lambda_{n}}{\lambda_{n-1}}|<1$ and $\langle \vec v_n,\vec x^{(0)}\rangle \neq 0$ ($\vec v_n$ is the eigenvector of $\lambda_n$).
#    * $\rho_k=\|A\vec x^{(k)}\|_{\infty}=|\lambda_n+\mathcal{O}(\eta^k)|$, slow convergence if $\eta \approx 1$.
# 3. Inverse Power Method with Shift
#    * Let $\lambda_j$ be the eigenvalue closet to $\mu$.
#    * Converges if $\eta:=\max_{m\neq j}|\frac{\lambda_{j}-\mu}{\lambda_{m}-\mu}|<1$ and $\langle \vec v_j,\vec x^{(0)}\rangle \neq 0$ ($\vec v_j$ is the eigenvector of $\lambda_j$).
#    * $\rho_k=\|A\vec x^{(k)}\|_{\infty}=|\lambda_j+\mathcal{O}(\eta^k)|$, slow convergence if $\eta \approx 1$.
# 
# 
# In all the methods above, if $\eta \approx 1$, then the convergence is slow. But in some cases, we can actually speed up the convergence. Let's focus on the Inverse Power Method with Shift.
# 

# 
# ### Rayleigh Quotient Iteration
# 
# When using Inverse Power Method with Shift, we can update $\mu$ in each iteration such that $\mu$ is closer to a real eigenvalue in each iteration. Then, $\eta:=\max_{m\neq j}|\frac{\lambda_{j}-\mu}{\lambda_{m}-\mu}|$ will become smaller and smaller, so that the convergence is faster and faster.
# 
# 
# 
# **Definition 1 (Rayleigh Quotient)** Let a non-zero vector $\vec v\in \mathbb C, A\in M_{n\times n} (\mathbb C)$. Then the Rayleigh quotient is defined as 
# 
# $$
# R(\vec v, A)=\frac{\vec v^* A\vec v}{\vec v^* \vec v}.
# $$
# 
# Remark: Let $A$ be a symmetric positive definite (SPD) matrix, then all eigenvalues $\lambda_1\geq \cdots\geq \lambda_n$ are real, and
# 
# * $\lambda_n\leq R(\vec v, A) \leq \lambda_1$;
# * $R(\vec v, A)=\lambda_j$ when $\vec v=\vec v_j=$ eigenvalue of $\lambda_j$.
#   
# Therefore, the  Rayleigh quotient $R(\vec v, A)$ can be regarded as an approximation of the eigenvalue $\lambda_j$, given that $\vec v$ is close to $\vec v_j$.

# #### Algorithm 4 (Rayleigh Quotient Iteration)
# 
# Goal: Take $\mu \in \mathbb R$, find the eigenvalue of $A$ that closet to $\mu$.
# 
# 1. Input: $\vec x^{(0)}$ with $\|\vec x^{(0)}\|_2=1$, and $\mu_0=\mu$ is the initial guess of desired eigenvalue.
# 
# 
# 2. For $k=1,2,\cdots$, repeat until convergence:
#    * Solve $(A-\mu_k I)\vec y=\vec x^{(k-1)}$.
#    * Compute $\vec x^{(k)}=\frac{\vec y}{\|\vec y\|_{2}}.$
#    * Compute $\mu_k=R(\vec x^{(k)}, A)=\frac{\vec x^{(k)*} A\vec  x^{(k)}}{\vec x^{(k)*} \vec x^{(k)}}$.
# 
# 3. Output: $\mu_k$ is the eigenvalue of $A$ that closet to $\mu$.
# 

# #### Code demo

# In[7]:


def RQI(A,mu,tol=10**(-7),N=100,x=None,seed=10):
  """Finding the eigenvalue of A that closet to mu via the Rayleigh Quotient Iteration."""
  
  random.seed(seed)

  # initilize x
  n=A.shape[1]
  x=np.random.rand(n,1)
  x=x/np.linalg.norm(x)

  # iterate for N times 
  for i in range(N):
    # inverse power iteration
    y=np.linalg.solve(A-mu*np.identity(n),x)
    x_iter = y/np.linalg.norm(y)
    mu_iter= (np.transpose(x_iter)@A@x_iter)/(np.transpose(x_iter)@x_iter)
    x = x_iter

    # check for convergence
    tol_iter = abs(mu_iter-mu)
    if(tol_iter < tol): 
      mu = mu_iter
      break
    mu = mu_iter

  x = np.transpose(x)

  return x, mu, i, tol_iter


# Consider the matrix $A=\begin{pmatrix}
# 5 & -2 & 3\\
# -3 & 9 & 1\\
# 2 & -1 & -7\\
# \end{pmatrix}$. Suppose we want to find eigenvalue of $A$ that closet to $\mu=6$.

# In[8]:


sol_RQI=RQI(A,mu=6)
print('The eigenvalue that closet to mu = %0.4f' %(sol_RQI[1]))
print('The corresponding eigenvector = ', sol_RQI[0])
print('Iteration = ', sol_RQI[2])
print('Tolerence = ', sol_RQI[3])


# From the code above, the eigenvalue that closet to $\mu=6$ is $|\lambda|=4.1830$. Note that the Rayleigh Quotient Iteration converges in only 3 iterations! Actually the Rayleigh Quotient Iteration has a cubic convergence, which is rare in numerical methods.
# 

# 
# ## QR Method
# 
# If we need to find all the eigenvalues of a matrix $A$, QR method is preferred. QR method is constructed based on the QR factorization of matrix and Gram-Schimidt process. We will discuss the latter two at first.

# 
# ### QR Factorization
# 
# Let $A=
# \begin{pmatrix}
# |& |& & |\\
# \vec a_1 & \vec a_2 &\cdots &\vec a_n\\
# |& |& & |
# \end{pmatrix}$ be linearly independent. The Gram-Schimidt process converts $\{\vec a_1,\vec a_2,\cdots,\vec a_n\}$ to an orthonormal set $\{\vec q_1,\vec q_2,\cdots,\vec q_n\}$. The details are as follows.

# 
# #### Algorithm 5 (Gram-Schimidt process)
# 
# 1. $\tilde q_1=\vec a_1$. 
#    * Normalize $\vec q_1=\tilde q_1/\|\tilde q_1\|_2$. 
#    * Define $\alpha_{11}=\|\tilde q_1\|_2$.
# 2. $\tilde q_2=\vec a_2-proj_{\tilde q_1}(\vec a_2)=\vec a_2-\alpha_{12}\vec q_1$.
#    * Normalize $\vec q_2=\tilde q_2/\|\tilde q_2\|_2$. 
#    * Define $\alpha_{12}=\vec q_1^T\vec a_2$.
#    * Define $\alpha_{22}=\|\tilde q_2\|_2$.
#   
# 3.  $\tilde q_3=\vec a_3-proj_{\tilde q_1}(\vec a_3)-proj_{\tilde q_2}(\vec a_3)=\vec a_3-\alpha_{13}\vec q_1-\alpha_{23}\vec q_2$
#        * Normalize $\vec q_3=\tilde q_3/\|\tilde q_3\|_2$. 
#        * Define $\alpha_{13}=\vec q_1^T\vec a_3,\alpha_{23}=\vec q_2^T\vec a_3$.
#        * Define $\alpha_{33}=\|\tilde q_3\|_2$.
#   
# 4. Suppose $\vec q_1, \vec q_2,\cdots,\vec q_{k-1}$ are constructed. Then
#    
#    $\tilde q_k=\vec a_k-\sum_{i=1}^{k-1} proj_{\tilde q_i}(\vec a_k)=\vec a_k-\sum_{i=1}^{k-1}\alpha_{ik}\vec q_i$.
#    * Normalize $\vec q_k=\tilde q_k/\|\tilde q_k\|_2$. 
#    * Define $\alpha_{jk}=\vec q_j^T\vec a_k$ for $j=1,\cdots,k-1$.
#    * Define $\alpha_{kk}=\|\tilde q_k\|_2$.
# 
# From the above process, we have
# * $\vec a_1=\alpha_{11} \vec q_1$;
# * $\vec a_2=\alpha_{12} \vec q_1+\alpha_{22} \vec q_2$;
# * $\vec a_3=\alpha_{13} \vec q_1+\alpha_{23} \vec q_2+\alpha_{33} \vec q_3$;
# * ...
# * $\vec a_k=\alpha_{1k} \vec q_1+\alpha_{2k} \vec q_2+\cdots+\alpha_{kk} \vec q_k$.
# 
# 

# 
# Write it in matrix form. For a matrix $A\in M_{m\times n}$:
# 
# $$
# A=
# \begin{pmatrix}
# |& |& & |\\
# \vec a_1 & \vec a_2 &\cdots &\vec a_n\\
# |& |& & |
# \end{pmatrix}
# =
# \begin{pmatrix}
# |& |& & |\\
# \vec q_1 & \vec q_2 &\cdots &\vec q_n\\
# |& |& & |
# \end{pmatrix}
# \begin{pmatrix}
# \alpha_{11}& \alpha_{12}&\cdots & \alpha_{1n}\\
#  & \alpha_{22} &\cdots &\alpha_{2n}\\
#  & & \ddots &\vdots\\
# & & & \alpha_{nn} 
# \end{pmatrix}
# =QR.
# $$
# 
# 
# We denote the orthonormal $Q$ matrix to be 
# 
# $$
# Q=\begin{pmatrix}
# |& |& & |\\
# \vec q_1 & \vec q_2 &\cdots &\vec q_n\\
# |& |& & |
# \end{pmatrix}\in M_{m\times n},
# $$
# 
# and the upper triangular $R$ matrix to be 
# 
# $$
# R=\begin{pmatrix}
# \alpha_{11}& \alpha_{12}&\cdots & \alpha_{1n}\\
#  & \alpha_{22} &\cdots &\alpha_{2n}\\
#  & & \ddots &\vdots\\
# & & & \alpha_{nn} 
# \end{pmatrix}\in M_{n\times n},
# $$
# 
# The QR factorization of $A$ is simply
# 
# $$
# A=QR.
# $$
# 
# Remark: If all the diagnoal entries of $R$ are positive, then the  QR factorization is unique.
# 

# 
# #### Algorithm 6 (QR Factorization)
# 
# Let $A=(\vec a_1,\cdots,\vec a_n)$ be full rank.
# 
# 1. Apply Gram-Schimidt process to obtain an orthonormal set $\{\vec q_1,\vec q_2,\cdots,\vec q_n\}$. 
# 2. Compute $\alpha_{jk}=\vec q_j^T \vec a_k$ for $j=1,\cdots,k$; or simply use $R=Q^TA$.
# 3. Construct QR Factorization
#    
# $$
# A=QR=
# \begin{pmatrix}
# |& |& & |\\
# \vec q_1 & \vec q_2 &\cdots &\vec q_n\\
# |& |& & |
# \end{pmatrix}
# \begin{pmatrix}
# \alpha_{11}& \alpha_{12}&\cdots & \alpha_{1n}\\
#  & \alpha_{22} &\cdots &\alpha_{2n}\\
#  & & \ddots &\vdots\\
# & & & \alpha_{nn} 
# \end{pmatrix}.
# $$

# #### Code demo

# In[9]:


def proj(v1, v2):
    """The projection of v2 onto v1"""
    coef = np.dot(v2, v1) / np.dot(v1, v1)
    proj = coef*v1
    return proj


# In[10]:


def QR_factor(A):
  """Finding the QR factorization of A (non-singular) via the Gram-Schimidt process."""
  A=np.transpose(A)
  Q=np.zeros((A.shape))

  for k in range(A.shape[1]):
    q=A[k]
    for i in range(k):
      q=q-proj(Q[i],A[k])
    Q[k]=q/np.linalg.norm(q)
   
  R= Q@np.transpose(A)
  Q=np.transpose(Q)

  return Q,R
  


# Consider the matrix $A=\begin{pmatrix}
# 5 & -2 & 3\\
# -3 & 9 & 1\\
# 2 & -1 & -7\\
# \end{pmatrix}$.

# In[11]:


A = np.array([[5,-3,2],[-2,9,-1],[3,1,-7]])
np.set_printoptions(suppress=True)
QR_factor(A)


# In[12]:


# Compare with the built-in numpy function, the results are the same
np.linalg.qr(A)


# ### QR Method for finding eigenvalues
# 
# Now we come to the QR Method to find all the eigenvalues of a matrix. The algorithm is very simple.

# #### Algorithm 7 (QR Method)
# 
# Input: $A\in M_{n\times n}(\mathbb R)$ is a real symmetric non-singular matrix.
# 
# Output: All eigenvalues of $A$.
# 
# 
# 1. Initilize $A^{(0)}=A$. 
#    * Compute the QR factorization of $A^{(0)}$: $A^{(0)}=Q_0R_0$.
#    * $A^{(1)}=R_0 Q_0$.
# 
# 2. Assume $A^{(1)},\cdots,A^{(k)}$ are constructed.
#    * Compute the QR factorization of $A^{(k)}$: $A^{(k)}=Q_kR_k$.
#    * $A^{(k+1)}=R_k Q_k$.
#   
# 3. Repeat Step 2 until convergence.
# 4. The diagnoal entries of $A^{(k)}$ will converge to the eigenvalues of $A$.

# #### Code demo

# In[13]:


def QR(A,tol=10**(-7),N=100):
  """Finding all the eigenvalue of A via the QR method."""

  eigen = np.diagonal(A) # extract the diagonal of A

  # iterate for N times 
  for i in range(N):
    # QR iteration
    Q_iter, R_iter = QR_factor(A)
    A_iter=R_iter@Q_iter
    eigen_iter = np.diagonal(A_iter)
    A = A_iter

    # check for convergence
    tol_iter = np.linalg.norm(eigen_iter-eigen)
    if(tol_iter < tol): 
      eigen = eigen_iter
      break
    eigen = eigen_iter

  return A, eigen, i, tol_iter


# Consider the matrix $A=\begin{pmatrix}
# 0 & -2 \\
# 2 & 3 \\
# \end{pmatrix}$.

# In[14]:


A = np.array([[0,2],[2,3]])
QR(A)


# From the code above, the eigenvalues of $A$ are $\lambda_{1,2}=4,-1$.

# 
# At first glimpse, QR Method seems a little bit crazy: we do QR, RQ, QR, RQ,... sequentially, and we get the eigenvalues! How does it happen? Don't be confused, the intuition behind is really simple.
# 
# We can observe that the QR Method gives a sequence of matrices $\{A=A^{(0)},A^{(1)},\cdots,A^{(k)},\cdots\}$, where
# 
# $$
# A^{(1)}=R_0 Q_0=(Q_0^{-1}Q_0)R_0 Q_0=Q_0^{-1}(Q_0R_0) Q_0=Q_0^{-1}A^{(0)} Q_0.
# $$
# 
# So $A^{(1)}$ is similar to $A^{(0)}$, which implies that $A^{(1)}$  has the same set of eigenvalues as $A^{(0)}$, this is because
# 
# $$
# det(A^{(1)}-\lambda I)=det(Q_0^{-1}A^{(0)} Q_0-\lambda I)=det(Q_0^{-1}(A-\lambda I)Q_0)=det(A^{(0)}-\lambda I).
# $$
# 
# Similarly, 
# 
# $$
# A^{(2)}=R_1 Q_1=(Q_1^{-1}Q_1)R_1 Q_1=Q_1^{-1}(Q_1R_1) Q_1=Q_1^{-1}A^{(1)} Q_1.
# $$
# 
# Therefore, this sequence of matrices are all similar:
# 
# $$
# A=A^{(0)}\sim A^{(1)}\sim  A^{(2)} \sim\cdots\sim A^{(k)}\sim \cdots
# $$
# 
# 
# QR method works for all real matrix, but under certain condition of $A$, the convergence of QR method is guaranteed.

# 
# #### Convergence of QR Method
# 
# We state the following theorem without proof.
# 
# **(Theorem 1)** Let $A$ be real symmetric non-singular matrix. Then the sequence of matrices generated by the QR Method converges to an upper triangular matrix; and the diagnoal entries of $A^{(k)}$ will converge to the eigenvalues of $A$. 

# 
# ## The Equivalence of Power Method and QR Method
# 
# There is nothing magic in mathematics. Power Method and QR Method are essentially the same thing. We will prove it here. Before that, note that the most obvious difference between these two methods: Power Method finds the dominating eigenvalue, whereas QR Method finds all the eigenvalues. However, if we make a subtle modification to the Power Method, it will cater for finding all the eigenvalues.
# 
# 
# In Power method, the initialization $\vec x^{(0)}=a_1\vec x_1+a_2\vec x_2+\cdots+a_n\vec x_n,\ a_1\neq 0$ is random. Note that although we put a "constraint" $a_1\neq 0$ here, it doesn't affect our random initialization. From probability theory, if we randomly pick a vector $\vec x^{(0)}=a_1\vec x_1+a_2\vec x_2+\cdots+a_n\vec x_n$ in $\mathbb C^n$ the probability of $a_1=0$ is of measure 0. If you are so lucky to encounter the case that $a_1= 0$, no kidding, you should stop doing math and go to buy a lottery.
# 
# In the imaginary world, however, we can assume this case happens. Consider a matrix $A\in M_{n\times n}(\mathbb C)$ with $n$ eigenvalues $|\lambda_1|> |\lambda_2|>  \cdots>|\lambda_i|\cdots>|\lambda_n|$ and corresponding eigenvectors $\vec x_1, \vec x_2, \cdots,\vec x_i, \cdots, \vec x_n$. 

# 
# ### Simultaneous Iteration
# 
#  Imagine that we choose $\vec x^{(0)}=a_i\vec x_i+a_{i+1}\vec x_{i+1}+\cdots+a_n\vec x_n,\ a_i\neq 0$ (most likely not at random), then the Power iteration is
# 
# $$
# \begin{align*}
# \vec x^{(k)}&=\frac{A^k\vec x^{(0)}}{\|A^k\vec x^{(0)}\|_{\infty}}\\
# &=\frac{a_i\lambda_i^k\vec x_i+a_{i+1}\lambda_{i+1}^k\vec x_{i+1}+\cdots+a_n\lambda_n^k\vec x_n}{\|a_i\lambda_i^k\vec x_i+a_{i+1}\lambda_{i+1}^k\vec x_{i+1}+\cdots+a_n\lambda_n^k\vec x_n\|_{\infty}}\\
# &=\frac{a_i\lambda_i^k(\vec x_i+\sum_{j=i+1}^n\frac{a_j}{a_i}(\frac{\lambda_j}{\lambda_i})^k\vec x_j)}{\|a_i\lambda_i^k(\vec x_i+\sum_{j=i+1}^n\frac{a_j}{a_i}(\frac{\lambda_j}{\lambda_i})^k\vec x_j)\|_{\infty}}\\
# &\to \frac{a_i\lambda_i^k\vec x_i}{\|a_i\lambda_i^k\vec x_i\|_{\infty}}\qquad\text{as }k\to \infty
# \end{align*}
# $$
# 
# so $\vec x^{(k)}$ will converges to the direction of $\vec x_i$ (the eigenvector corresponding with $\lambda_i$), and 
# 
# $$
# \begin{align*}
# \|A\vec x^{(k)}\|_{\infty} &\to \| \frac{a_i\lambda_i^k}{|a_i||\lambda_i^k|\|\vec x_i\|_{\infty}}\cdot A\vec x_i\|_{\infty}\\
# &=\| \frac{1}{\|\vec x_i\|_{\infty}}\cdot \lambda_i\vec x_i\|_{\infty}\\
# &=|\lambda_i|\qquad\qquad\text{as }k\to \infty\\
# \end{align*}
# $$
# 
# that is, $\|A\vec x^{(k)}\|_{\infty}$ converges to eigenvalue $|\lambda_i|$.
# 
# If we are "smart" enough to choose $n$ such initial guess $\{\vec x_1^{(0)},\vec x_2^{(0)},\cdots,\vec x_n^{(0)} \}$ where $\vec x_i^{(0)}=a_i\vec x_i+a_{i+1}\vec x_{i+1}+\cdots+a_n\vec x_n,\ a_i\neq 0$, which forms a matrix
# 
# $$
# X^{(0)}=
# \begin{pmatrix}
# |& |& & |\\
# \vec x_1^{(0)} & \vec x_2^{(0)} &\cdots &\vec x_n^{(0)}\\
# |& |& & |
# \end{pmatrix}
# \in M_{n\times n} (\mathbb C),
# $$
# 
# we can apply the Power method on $X^{(0)}$ (this is equivalent to applying Power method $n$ times to $\{\vec x_1^{(0)},\vec x_2^{(0)},\cdots,\vec x_n^{(0)} \}$).
# 
# The Power iteration $X^{(k)}=AX^{(k-1)}$ will generate (without normalization)
# 
# $$
# A^kX^{(0)}=
# \begin{pmatrix}
# |& |& & |\\
# A^k \vec x_1^{(0)} & A^k \vec x_2^{(0)} &\cdots &A^k \vec x_n^{(0)}\\
# |& |& & |
# \end{pmatrix}\to 
# \begin{pmatrix}
# |& |& & |\\
# k_1 \vec x_1 & k_2 \vec x_2 &\cdots &k_n \vec x_n\\
# |& |& & |
# \end{pmatrix}
# $$
# 
# where $A^k \vec x_i^{(0)}$ $(i=1,\cdots,n)$ converges to the direction of $\vec x_i$, the eigenvector corresponding with $\lambda_i$.
# 
# 
# If we can make sure that $A^kX^{(0)}$ is orthogonal (after some iterations), then the convergence is guaranteed. How can we ensure the orthogonality of $A^kX^{(0)}$? By QR factorization!
# 
# Consider an intitial guess $X^{(0)}$ (usually $I_n$).
# 
# * We take out the "orthogonal" part of $X^{(0)}$ by QR factorization: $X^{(0)}=\overline Q^{(0)}R^{(0)}$ (assume that all diagonal entries of $R^{(0)}$ are positive so that the factorization is unique). 
# * Then we apply the Power method on the extracted orthogonal part $\overline Q^{(0)}$ to get $W=A\overline Q^{(0)}$.
# * Again, we take out the "orthogonal" part of $W$ by QR factorization: $W=\overline Q^{(1)}R^{(1)}$.
# * Then we apply the Power method on the extracted orthogonal part $\overline Q^{(1)}$ to get $W=A\overline Q^{(1)}$.
# * ...
# 
# By repeatedly extracting the orthogonal matrix and applying the Power method on the extracted matrix, we generate a sequence of orthogonal matrices $\overline Q^{(0)},\overline Q^{(1)},\cdots$, and the sequence will converge to $\begin{pmatrix}
# |& |& & |\\
# k_1 \vec x_1 & k_2 \vec x_2 &\cdots &k_n \vec x_n\\
# |& |& & |
# \end{pmatrix}$ when $k$ tends to infinity.
# 
# Now we formulate the algorithm.
# 

# #### Algorithm 8 (Simultaneous Iteration) 
# 
# Goal: Finding all eigenvalues of a real symmetric matrix $A$.
# 
# Input: An intitial guess $X^{(0)}=I_n
# \begin{pmatrix}
# |& |& & |\\
# \vec x_1^{(0)} & \vec x_2^{(0)} &\cdots &\vec x_n^{(0)}\\
# |& |& & |
# \end{pmatrix}
# \in M_{n\times n} (\mathbb C)$
# 
# 1. Obtain the QR factorization of $X^{(0)}=\overline Q^{(0)}R^{(0)}$.
# 2. For $k=1,2,\cdots$, repeat until convergence:
#    *  Apply Power method on $\overline Q^{(k-1)}$ to get $W=A\overline Q^{(k-1)}$.
#    *  Obtain the QR factorization of $W=\overline Q^{(k)}R^{(k)}$.
#    *  Record $A^{(k)}=(\overline Q^{(k)})^T A \overline Q^{(k)}$.
# 
# Output: A sequence of orthogonal matrices $\overline Q^{(0)},\overline Q^{(1)},\cdots$, where
# 
# $$
# \overline Q^{(k)}\to
# \begin{pmatrix}
# |& |& & |\\
# k_1 \vec x_1 & k_2 \vec x_2 &\cdots &k_n \vec x_n\\
# |& |& & |
# \end{pmatrix}
# \qquad\text{as }\ k\to \infty.
# $$
# 

# Now we are going to show that Algorithm 8 (Simultaneous Iteration) is equivalent to the QR method we discussed earlier. But to make the two methods comparable, we firstly modify some details in the original QR method.

# ### Algorithm 9 (QR Method, a variation) 
# 
# Goal: Finding all eigenvalues of a real symmetric matrix $A$.
# 
# 1. Initilization $A^{(0)}_{QR}=A$.
# 2. For $k=1,2,\cdots$, repeat until convergence:
#    * Obtain the QR factorization of $A_{QR}^{(k-1)}=Q_{QR}^{(k)}R_{QR}^{(k)}$.
#    * Compute $A_{QR}^{(k)}=R_{QR}^{(k)}Q_{QR}^{(k)}$.
#    * Record $\overline Q_{QR}^{(k)}=Q_{QR}^{(1)}Q_{QR}^{(2)}\cdots Q_{QR}^{(k)}$.
#    * Record $\overline R_{QR}^{(k)}=R_{QR}^{(k)}R_{QR}^{(k-1)}\cdots R_{QR}^{(1)}$.
# 
# Output: A sequence of matrices $A^{(k)}$ and $\overline Q^{(k)}$.
# 
# 

# 
# ### The Equivalence of Simultaneous Iteration and QR Method
# 
# **(Theorem 2)** *Algorithm 8 (Simultaneous Iteration)* and *Algorithm 9 (QR Method, a variation)* are equivalent. That is, for each $k$, the RHS represents Simultaneous Iteration, and the LHS represents QR Method are the same:
# 
# 1. $A_{QR}^{(k)}=A^{(k)}$;
# 2. $\overline Q_{QR}^{(k)}=\overline Q^{(k)}$;
# 3. $\overline R_{QR}^{(k)}=\overline R^{(k)}$;
# 4. $A^k=\overline Q_{QR}^{(k)}\overline R_{QR}^{(k)}=\overline Q^{(k)}\overline R^{(k)}$;
# 5. $(\overline Q_{QR}^{(k)})^T A \overline Q_{QR}^{(k)}=(\overline Q^{(k)})^T A \overline Q^{(k)}$.
# 
# The proof is very simple via induction. Just be careful about the notation. We ignore the proof here.
# 
