# Systems of linear equations

* Learning Objectives 
* Form and graphically interpret 2x2 and 3x3 systems of linear equations 
* Determine the number of solutions to a 2x2 and 3x3 system of linear equations 
* Distinguish between singular and non-singular systems of equations 
* Determine the singularity of 2x2 and 3x3 system of equations by calculating the determinant


Here’s the updated Jupyter Notebook-compatible markdown with the missing <sup>T</sup> properly included:

# Notations

The following is a reference for the notations used in this course. It may assist you during the video lectures and assignments. Don't worry if you don't understand them yet; they will be covered during the course.


### Notations and Explanations

- **$A, B, C$**  
  Capital letters represent matrices.  

- **$u, v, w$**  
  Lowercase letters represent vectors.  

- **$A$ of size $m \times n$**  
  Matrix $A$ has $m$ rows and $n$ columns.  

- **$A^T$**  
  The transpose of matrix $A$.  

- **$v^T$**  
  The transpose of vector $v$.  

- **$A^{-1}$**  
  The inverse of matrix $A$.  

- **$\det(A)$**  
  The determinant of matrix $A$.  

- **$AB$**  
  Matrix multiplication of matrices $A$ and $B$.  

- **$u \cdot v$** or **$\langle u, v \rangle$**  
  Dot product of vectors $u$ and $v$.  

- **$\mathbb{R}$**  
  The set of real numbers, e.g., $0, -0.642, 2, 3.456$.  

- **$\mathbb{R}^2$**  
  The set of two-dimensional vectors, e.g., $v = \begin{bmatrix} 1 \\ 3 \end{bmatrix}^T$.  

- **$\mathbb{R}^n$**  
  The set of $n$-dimensional vectors.  

- **$v \in \mathbb{R}^2$**  
  Vector $v$ is an element of $\mathbb{R}^2$.  

- **$\|v\|_1$**  
  L1-norm of a vector.  

- **$\|v\|_2$, $\|v\|$, $|v|$**  
  L2-norm of a vector.  

- **$T: \mathbb{R}^2 \to \mathbb{R}^3$**  
  Transformation $T$ of a vector $v \in \mathbb{R}^2$ into the vector $w \in \mathbb{R}^3$.  

- **$T(v) = w$**  
  Transformation applied to vector $v$ resulting in vector $w$.  

---


# Determinant of a 3×3 Matrix

The determinant of a **3×3 matrix** can be calculated using the following formula:

For a matrix $A$:

$$
A =
\begin{bmatrix}
a & b & c \\
d & e & f \\
g & h & i
\end{bmatrix}
$$

The determinant of $A$, denoted as $\text{det}(A)$ or $|A|$, is given by:

$$
\text{det}(A) = a(ei - fh) - b(di - fg) + c(dh - eg)
$$

---
Given a matrix:

$
A =
\begin{bmatrix}
2 & 1 & 3 \\
4 & 5 & 6 \\
7 & 8 & 9
\end{bmatrix}
$

Solution: $\text{det}(A)$

$\text{det}(A)$ = 2 $\begin{vmatrix}
5 & 6 \\
8 & 9
\end{vmatrix}$
\- 1 $\begin{vmatrix}
4 & 6 \\
7 & 9
\end{vmatrix}$
\+ 3 $\begin{vmatrix}
4 & 5 \\
7 & 8
\end{vmatrix}
$

$\text{det}(A) = 2(-3) - 1(-6) + 3(-3)
$

$
\text{det}(A) = -6 + 6 - 9 = -9
$

$
\text{det}(A) = -9
$

---

## Notes

- Determinants are crucial for solving linear equations, finding the inverse of matrices, and determining matrix singularity.
- The determinant of a singular matrix (non-invertible matrix) is $0$.
