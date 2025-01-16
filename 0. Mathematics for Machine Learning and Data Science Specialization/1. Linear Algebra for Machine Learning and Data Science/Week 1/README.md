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

- **$A, B, C$** : &emsp;&emsp;&emsp;&emsp;&emsp;&emsp; Capital letters represent matrices.  

- **$u, v, w$** : &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; Lowercase letters represent vectors.  

- **$A (m \times n)$** : &emsp; Matrix $A$ has $m$ rows and $n$ columns.  

- **$A^T$** : &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; The transpose of matrix $A$.  

- **$v^T$** : &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; The transpose of vector $v$.  

- **$A^{-1}$** : &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; The inverse of matrix $A$.  

- **$\det(A)$**  : &emsp;&emsp;&emsp;&emsp;&emsp;&emsp; The determinant of matrix $A$.  

- **$AB$**: &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; Matrix multiplication of matrices $A$ and $B$.  

- **$u \cdot v$** or **$\langle u, v \rangle$** :  Dot product of vectors $u$ and $v$.  

- **$\mathbb{R}$** : &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; The set of real numbers, e.g., $0, -0.642, 2, 3.456$.  

- **$\mathbb{R}^2$** : &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; The set of two-dimensional vectors, e.g., $v = \begin{bmatrix} 1 \\ 3 \end{bmatrix}^T$.  

- **$\mathbb{R}^n$** : &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; The set of $n$-dimensional vectors.  

- **$v \in \mathbb{R}^2$** : &emsp;&emsp;&emsp;&emsp;&emsp;&emsp; Vector $v$ is an element of $\mathbb{R}^2$.  

- **$\|v\|_1$** : &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; L1-norm of a vector.  

- **$\|v\|_2$, $\|v\|$, $|v|$** : L2-norm of a vector.  

- **$T: \mathbb{R}^2 \to \mathbb{R}^3$** : Transformation $T$ of a vector $v \in \mathbb{R}^2$ into the vector $w \in \mathbb{R}^3$.  

- **$T(v) = w$** : &emsp;&emsp;&emsp;&emsp;&emsp; Transformation applied to vector $v$ resulting in vector $w$.  

---
## Determinant of a Matrix

The determinant of a square matrix \(A\) is a scalar value that provides important information about the matrix:
- Whether the matrix is invertible (non-singular) or not (singular).
- Geometric interpretation such as area (2D) or volume (3D) scaling factor of the linear transformation defined by the matrix.

Mathematically:
\[$det(A) = \text{some scalar based on the entries of } A.$\]

## Determinant of a 3×3 Matrix

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

## Singular matrix and non-singular matrix

- A **singular matrix** is a square matrix that does not have an inverse. This happens when its determinant is **zero**.
- A **non-singular matrix** is a square matrix that has an inverse. This happens when its determinant is **non-zero**.


| **Feature**                  | **Singular Matrix**                                 | **Non-Singular Matrix**                             |
|----------------------------- |-----------------------------------------------------|-----------------------------------------------------|
| **Determinant**             | $( \det(A) = 0 )$                                   | $( \det(A) \neq 0 )$                                |
| **Invertibility**            | Not invertible (no $(A^{-1})$ exists)               | Invertible (exists $(A^{-1})$ with $(AA^{-1} = I))$ |
| **Rank**                     | Less than its dimension (not full rank)             | Equal to its dimension (full rank)                  |
| **Linear Dependence**        | Rows (or columns) are linearly dependent            | Rows (or columns) are linearly independent          |
| **Solutions to \(A\mathbf{x} = \mathbf{b}\)** | No unique solution (either none or infinitely many) | Exactly one unique solution                         |
| **Geometric Interpretation** | Transformation “collapses” space (volume/area = 0)  | Transformation is bijective (no collapse)           |
| **Examples**                 | $\begin{pmatrix}1 & 2 \\ 2 & 4\end{pmatrix}$        | $\begin{pmatrix}2 & 1 \\ 1 & 3\end{pmatrix}$        |

-------



## Notes

- Determinants are crucial for solving linear equations, finding the inverse of matrices, and determining matrix singularity.
- The determinant of a singular matrix (non-invertible matrix) is $0$.
