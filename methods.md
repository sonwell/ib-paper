# Methods {#sec:methods}
The matrix $\mathcal{S}^\dagger$ in [@eq:interp-disc] is sparse, with
$n_\gamma$ rows, $n_\omega$ columns, and a fixed number of potentially nonzero
entries per row. This matrix is used in interpolating the fluid velocity from
the Eulerian grid to the Lagrangian grid. Interpolation is therefore a single
scaled sparse matrix-vector multiplication. Having a fixed number of entries
per row is beneficial for parallelization. By assigning a thread to each row,
each thread computes a sparse dot product, scales by $h^d$, and writes to the
corresponding component of the output vector. Each thread does approximately
the same amount of work because the number of nonzero entries per row is
uniform among the rows. We use this idea to parallelize interpolation, but
avoid explicitly constructing the matrix.

The difficulty comes from _spreading_ the cell forces from the Lagrangian grid
to the Eulerian grid. The values of $\delta_h$ in [@eq:spread-disc] comprise
a matrix, $S$, with $n_\omega$ rows, $n_\gamma$ columns, a fixed number
of entries per column. Unlike having an equal number of nonzero entries per
row, this is not easily parallelizable without constructing part or all of the
matrix. Instead, we seek submatrices $S_1$, $S_2$, \ldots, $S_m$ such that
\begin{equation*}
    \mathcal{S}\vec{F} = S_1\vec{F}_1 + S_2\vec{F}_2 + \cdots + S_m\vec{F}
\end{equation*}
and the products $S_i\vec{F}_i$, $i=1,\,2,\,\ldots,\,m$ can be computed
efficiently in parallel.

## Preliminaries
Let $\Omega^h$ be a fixed, regular grid such that any grid point can be written
$\vec{x} = h(\vec{i}+\vec{g})$ for $\vec{g} \in [0, 1)^d$ and
$\vec{i}\in\mathbb{Z}^d$. Let $\square_{\vec{i}}$ denote the grid cell with
opposite corners at $h(\vec{i}+\vec{g})$ and $h(\vec{i}+\vec{1}+\vec{g})$.

## Parallelization schemes {#sec:parallel}
Consider a point $\vec{X}_j\in\Gamma^h$. Let $K=\{1,\,\ldots,\,n_k\}$ be an
enumeration of the support points for $\vec{X}_j$. Let $i = i(j,\,k)$ be the
index of the $k$^th^ support point for $\vec{X}_j$, namely $\vec{x}_i$.
The function $\vec{\nu}$ is defined as in [@sec:optim].

\begin{algorithm}[thb]
    \begin{algorithmic}
        \STATE $\vec{U} \leftarrow \vec{0}$
        \FORALL {$\vec{X}_j\in\Gamma^h$ \textbf{parallel}}
        \STATE $\vec{w} \leftarrow \vec{\nu}(\vec{X}_j)$
        \FORALL {$k \in K$}
        \STATE $\ell \leftarrow i(j,\,k)$
        \IF {$\vec{x}_\ell \in \Omega$}
        \STATE $U_j \leftarrow U_j + h^dw_ku_\ell$
        \ENDIF
        \ENDFOR
        \ENDFOR
    \end{algorithmic}
    \caption{Parallel fluid velocity interpolation}
    \label{lst:interpolation}
\end{algorithm}

[@lst:interpolation] is a straightforward parallelization of [@eq:interp-disc]
which computes the fluid velocity at each of the Lagrangian points. The outer
loop loops over the Lagrangian points in parallel, and for each, updates the
velocity at the Lagrangian point with each of the contributions of its support
point. Omitting **parallel** from the outer loop reduces this algorithm into
its serial counterpart.

It is a simple matter to compute the Eulerian force density in a serial
context. The algorithm looks similar to [@lst:interpolation], replacing the
parallel **for** loop with a serial one, and the indices of the input and
output vectors (in the case of spreading, $\vec{F}\circ\vec{A}$ and $\vec{f}$,
respectively) would be swapped. Naïve attempts to parallelize over the
Lagrangian points, as we did in [@lst:interpolation], can miscalculate the
Eulerian forces if the support points for two different Lagrangian points
overlap, which is often the case. Parallelizing instead the loop over $K$ is
useful only up to $n_k$ threads, and would require thread synchronization
for every Lagrangian point. Neither of these are satisfactory solutions.

To properly compute the Eulerian force density in parallel, we seek a set of
submatrices $S_1$, $S_2$, \dots, $S_m$, where $S = S_1 + S_2 + \ldots + S_m$,
such that for submatrix $S_s$, there is a permutation matrix, $P_s$, for which
$(S_sP_s)P_s^{-1}(\vec{F}\circ\vec{A})$ is easy to compute in parallel. By
that, we mean that we can compute the pattern of nonzero entries in any row, or
set of rows, of $S_sP_s$ with little to no information about other rows.

The variable $s$ keeps track of the _sweep_, which is a stage of the algorithm
during which threads are working in parallel and at the end of which the
threads must be synchronized. Ideally, we would like as few sweeps as
possible, so that the threads spend as much time working in parallel as
possible. We also want each thread to perform approximately equal work, so
that one overburdened thread does not hold up synchronization. The number of
sweeps is $m$, the number of submatrics. Unfortunately, we don't necessarily
know _a priori_ where the Lagrangian points are, so we may need to use
a heuristic approach to construct the submatrices.

## Domain Decomposition Methods
A heuristic approach presented by [@McQueen:1997kw] partitions the Eulerian
grid into columns of points. Lagrangian points in columns that are $4h$ apart
will never have overlapping support points. There are $4^{d-1}$ sets of columns
that are all $4h$ apart, and each set contains up to $\lceil 4^{1-d}n_\omega \rceil$
columns. Lagrangian points are organized into linked lists according to which
column they are in. For $s=1,\,\ldots,\,4^{d-1}$, each thread is assigned
a column (or more) from the $s$^th^ set of columns and the values for each
support point of each Lagrangian point in that column are computed and written.
That is, the submatrix $S_s$ is constructed by taking only the columns of $S$
corresponding to Lagrangian points in the $s$^th^ set of columns.

\begin{figure}[tbh]
    \centering
    \begin{tikzpicture}[
        scale=2,
        declare function={
            ex(\t) = 3 + cos(10) * 2.1 * cos(\t) - sin(10) * 0.8 * sin(\t);
            ey(\t) = 2 + cos(10) * 0.8 * sin(\t) + sin(10) * 2.1 * cos(\t);
        }
    ]

    \draw[color=black!20, fill] (1.5, 0.25) rectangle (2.0, 3.25);
    \draw[color=black!20, fill] (3.5, 0.25) rectangle (4.0, 3.25);
    \draw[color=black!20, fill] (5.5, 0.25) rectangle (5.75, 3.25);
    \draw[step=0.5] (0.25, 0.25) grid (5.75, 3.25);

    \node[scale=0.8, shape=circle, color=tol/contrast/blue, fill] at (1.0, 0.5) {};
    \node[scale=0.8, shape=circle, color=tol/contrast/blue, fill] at (1.5, 0.5) {};
    \node[scale=0.8, shape=circle, color=tol/contrast/blue, fill] at (2.0, 0.5) {};
    \node[scale=0.8, shape=circle, color=tol/contrast/blue, fill] at (2.5, 0.5) {};
    \node[scale=0.8, shape=circle, color=tol/contrast/blue, fill] at (1.0, 1.0) {};
    \node[scale=0.8, shape=circle, color=tol/contrast/blue, fill] at (1.5, 1.0) {};
    \node[scale=0.8, shape=circle, color=tol/contrast/blue, fill] at (2.0, 1.0) {};
    \node[scale=0.8, shape=circle, color=tol/contrast/blue, fill] at (2.5, 1.0) {};
    \node[scale=0.5, line width=0, shape=circle split, circle split part fill={tol/contrast/red,tol/contrast/blue}] at (1.0, 1.5) {};
    \node[scale=0.5, line width=0, shape=circle split, circle split part fill={tol/contrast/red,tol/contrast/blue}] at (1.5, 1.5) {};
    \node[scale=0.5, line width=0, shape=circle split, circle split part fill={tol/contrast/red,tol/contrast/blue}] at (2.0, 1.5) {};
    \node[scale=0.5, line width=0, shape=circle split, circle split part fill={tol/contrast/red,tol/contrast/blue}] at (2.5, 1.5) {};
    \node[scale=0.5, line width=0, shape=circle split, circle split part fill={tol/contrast/red,tol/contrast/blue}] at (1.0, 2.0) {};
    \node[scale=0.5, line width=0, shape=circle split, circle split part fill={tol/contrast/red,tol/contrast/blue}] at (1.5, 2.0) {};
    \node[scale=0.5, line width=0, shape=circle split, circle split part fill={tol/contrast/red,tol/contrast/blue}] at (2.0, 2.0) {};
    \node[scale=0.5, line width=0, shape=circle split, circle split part fill={tol/contrast/red,tol/contrast/blue}] at (2.5, 2.0) {};
    \node[scale=0.8, shape=circle, color=tol/contrast/red, fill] at (1.0, 2.5) {};
    \node[scale=0.8, shape=circle, color=tol/contrast/red, fill] at (1.5, 2.5) {};
    \node[scale=0.8, shape=circle, color=tol/contrast/red, fill] at (2.0, 2.5) {};
    \node[scale=0.8, shape=circle, color=tol/contrast/red, fill] at (2.5, 2.5) {};
    \node[scale=0.8, shape=circle, color=tol/contrast/red, fill] at (1.0, 3.0) {};
    \node[scale=0.8, shape=circle, color=tol/contrast/red, fill] at (1.5, 3.0) {};
    \node[scale=0.8, shape=circle, color=tol/contrast/red, fill] at (2.0, 3.0) {};
    \node[scale=0.8, shape=circle, color=tol/contrast/red, fill] at (2.5, 3.0) {};

    \node[scale=0.8, shape=circle, color=tol/contrast/yellow, fill] at (3.0, 0.5) {};
    \node[scale=0.8, shape=circle, color=tol/contrast/yellow, fill] at (3.5, 0.5) {};
    \node[scale=0.8, shape=circle, color=tol/contrast/yellow, fill] at (4.0, 0.5) {};
    \node[scale=0.8, shape=circle, color=tol/contrast/yellow, fill] at (4.5, 0.5) {};
    \node[scale=0.8, shape=circle, color=tol/contrast/yellow, fill] at (3.0, 1.0) {};
    \node[scale=0.8, shape=circle, color=tol/contrast/yellow, fill] at (3.5, 1.0) {};
    \node[scale=0.8, shape=circle, color=tol/contrast/yellow, fill] at (4.0, 1.0) {};
    \node[scale=0.8, shape=circle, color=tol/contrast/yellow, fill] at (4.5, 1.0) {};
    \node[scale=0.8, shape=circle, color=tol/contrast/yellow, fill] at (3.0, 1.5) {};
    \node[scale=0.8, shape=circle, color=tol/contrast/yellow, fill] at (3.5, 1.5) {};
    \node[scale=0.8, shape=circle, color=tol/contrast/yellow, fill] at (4.0, 1.5) {};
    \node[scale=0.8, shape=circle, color=tol/contrast/yellow, fill] at (4.5, 1.5) {};
    \node[scale=0.8, shape=circle, color=tol/contrast/yellow, fill] at (3.0, 2.0) {};
    \node[scale=0.8, shape=circle, color=tol/contrast/yellow, fill] at (3.5, 2.0) {};
    \node[scale=0.8, shape=circle, color=tol/contrast/yellow, fill] at (4.0, 2.0) {};
    \node[scale=0.8, shape=circle, color=tol/contrast/yellow, fill] at (4.5, 2.0) {};

    \draw[very thick] plot[domain=-180:180, variable=\x, samples=100] ({ex(\x)}, {ey(\x)});
    \draw[thin, black, fill=tol/contrast/red] plot[only marks, mark=square*] coordinates {
        ({ex(123.03253921294208)}, {ey(123.03253921294208)})
    };
    \draw[thin, black, fill=tol/contrast/blue] plot[only marks, mark=square*] coordinates {
        ({ex(225.60289396961724)}, {ey(225.60289396961724)})
        ({ex(236.9674607870579)}, {ey(236.9674607870579)})
    };
    \draw[thin, black, fill=tol/contrast/yellow] plot[only marks, mark=square*] coordinates {
        ({ex(283.4525276213703)}, {ey(283.4525276213703)})
        ({ex(292.8699014257434)}, {ey(292.8699014257434)})
    };
    \draw[thin, black, fill=white] plot[only marks, mark=square*] coordinates {
        ({ex(16.902383319583745)}, {ey(16.902383319583745)})
        ({ex(32.4225566445974)}, {ey(32.4225566445974)})
        ({ex(45.60289396961725)}, {ey(45.60289396961725)})
        ({ex(56.967460787057895)}, {ey(56.967460787057895)})
        ({ex(67.13009857425658)}, {ey(67.13009857425658)})
        ({ex(76.54747237862972)}, {ey(76.54747237862972)})
        ({ex(85.5588196210609)}, {ey(85.5588196210609)})
        ({ex(94.44118037893911)}, {ey(94.44118037893911)})
        ({ex(103.4525276213703)}, {ey(103.4525276213703)})
        ({ex(112.86990142574342)}, {ey(112.86990142574342)})
        %({ex(123.03253921294208)}, {ey(123.03253921294208)})
        ({ex(134.39710603038273)}, {ey(134.39710603038273)})
        ({ex(147.57744335540258)}, {ey(147.57744335540258)})
        ({ex(163.09761668041625)}, {ey(163.09761668041625)})
        ({ex(180.0)}, {ey(180.0)})
        ({ex(196.90238331958375)}, {ey(196.90238331958375)})
        ({ex(212.4225566445974)}, {ey(212.4225566445974)})
        %({ex(225.60289396961724)}, {ey(225.60289396961724)})
        %({ex(236.9674607870579)}, {ey(236.9674607870579)})
        ({ex(247.13009857425658)}, {ey(247.13009857425658)})
        ({ex(256.5474723786297)}, {ey(256.5474723786297)})
        ({ex(265.55881962106093)}, {ey(265.55881962106093)})
        ({ex(274.4411803789391)}, {ey(274.4411803789391)})
        %({ex(283.4525276213703)}, {ey(283.4525276213703)})
        %({ex(292.8699014257434)}, {ey(292.8699014257434)})
        ({ex(303.0325392129421)}, {ey(303.0325392129421)})
        ({ex(314.3971060303827)}, {ey(314.3971060303827)})
        ({ex(327.5774433554026)}, {ey(327.5774433554026)})
        ({ex(343.09761668041625)}, {ey(343.09761668041625)})
        ({ex(360.0)}, {ey(360.0)})
    };
    \end{tikzpicture}
    \caption{%
        A portion of $\Omega$ containing $\Gamma$ (thick black curve). The
        Lagrangian points $\Gamma^h$ are represented by squares. All Lagrangian
        points that lie in the grey region are processed in a single sweep.
        Lagrangian points in different grey columns do not have overlapping
        support points, while Lagrangian points in the same column may have
        overlapping support points. If only one thread writes per column,
        there are no write contentions. The circular support points' colors
        indicate the corresponding square Lagrangian point(s). The red/blue
        semicircles indicate overlap between the support points of the red and
        blue Lagrangian points.
    }
    \label{fig:pib-columns}
\end{figure}

Let $\vec{X}_1,\,\ldots,\,\vec{X}_{30}$ be the 30 Lagrangian points illustrated
in [@fig:pib-columns], starting on the far right and moving along the object
counterclockwise. To each column, assign a unique sort index. The location of
each Lagrangian point will be identified by the sort index of the column that
contains it. Careful indexing will order the Lagrangian points by column and
then by sweep. This ordering defined the permutation matrices $\{P_s\}$. Define
a function $\sigma(\vec{X})$ which computes the sort index of the column
containing $\vec{X}$. In [@fig:pib-columns], there are 12 columns of grid
cells. Three of them are highlighted in grey, and the Lagrangian points in the
grey region will be processed in the same sweep. Assign the left-most column
a sort index of 1, the column $4h$ to the right sort index 2, and the column
$4h$ to the right of that has sort index 3. There is no next column $4h$ to the
right, so we return to the 2^nd^ column from the left and assign it a sort
index of 4. In this fashion, we assign sort indices to each of the columns; the
three grey columns will have sort indices 10, 11, and 12, and will be processed
in sweep $s = 4$. The matrix $S_4P_4$ has no nonzero entries in its first 25
columns, corresponding to the 25 Lagrangian points processed in sweeps 1--3,
and then five columns of 16 (possibly) nonzero values each. These columns
correspond to some permutation of the 12^th^, 19^th^, and 20^th^ columns of
$S$, followed by some permutation of the 25^th^ and 26^th^ columns of $S$.

\begin{algorithm}[tbh]
    \begin{algorithmic}
        \STATE $\vec{f} \leftarrow \vec{0}$
        \FORALL {$\vec{X}_j\in\Gamma^h$ \textbf{parallel}}
        \STATE \texttt{index}$_j \leftarrow j$
        \STATE \texttt{sort}$_j \leftarrow \sigma(\vec{X}_j)$
        \ENDFOR
        \STATE \texttt{sort-by-key}(\texttt{sort}, \texttt{index})
        \STATE \texttt{start}$\ \leftarrow  1$
        \FOR {$s=1,\,\ldots,\,m$}
        \STATE \texttt{end}$\ \leftarrow\ $\texttt{binary-search}($\lceil 4^{1-d}n_\omega \rceil s + 1$, \texttt{sort}, \texttt{start}, $n_\gamma+1$)
        \FORALL {$p\in\texttt{compact}(\texttt{sort},\,\texttt{start},\,\texttt{end})$ \textbf{parallel}}
        \STATE $\texttt{head} =\ $\texttt{binary-search}($p$, \texttt{sort}, \texttt{start}, \texttt{end})
        \WHILE {$\texttt{sort}_\texttt{head} = p$}
        \STATE $j \leftarrow\ $\texttt{index}$_\texttt{head}$
        \STATE $\Delta\vec{f} \leftarrow F_jA_j\vec{\nu}(\vec{X}_j)$
        \FOR {$k \in K$}
        \STATE $\ell = i(j,\,k)$
        \IF {$\vec{x}_\ell \in \Omega$}
        \STATE $f_\ell \leftarrow f_\ell + \Delta f_k$
        \ENDIF
        \ENDFOR
        \STATE $\texttt{head} \leftarrow \texttt{head} + 1$
        \ENDWHILE
        \ENDFOR
        \STATE \texttt{start} $\leftarrow$ \texttt{end}
        \ENDFOR
    \end{algorithmic}
    \caption{Columnwise force spreading with pruning}
    \label{lst:column-parallel}
\end{algorithm}

Using the sort index as a key to sort the points, points in the same column are
grouped together, and columns associated with a particular sweep are also
grouped together. After sorting, binary search tells us where a sweep begins,
and each thread can perform binary search to find where its associated
column begins. Thus, we can replace the linked list construction with two
arrays (one for points/point indices, and the other the sort index) and use the
well-studied key-value sort for list maintenance in addition to computing the
permutation matrices $\{P_s\}$. The resulting algorithm is listed in
[@lst:column-parallel] where the function `compact` returns, unaltered, its
first argument. Again, we employ the function $\nu(\vec{X})$, which organizes
the values of $\delta_h$ at each support point of $\vec{X}$ into a vector.

The introduction of a sorting step with $\mathcal{O}(n_\gamma\log n_\gamma)$ 
complexity seems counterproductive to parallelizing an $\mathcal{O}(n_\gamma)$
serial algorithm, but for any fixed the number of grid points, the complexity
of radix sort in this context is $\mathcal{O}(n_\gamma)$. A possible
improvement mimics the list maintenance of [@McQueen:1997kw] by using the sort
only initially. At each time step, (1) compute new sort indices for all of the
Lagrangian points, (2) partition the points predicated on the sort index
changing, (3) sort only the points with a changed sort index, and (4) merge the
(sorted) point partitions and sort indices. However, this has the same
$\mathcal{O}(n_\gamma)$ complexity.

We can optimize further based on the array-based construction. Having found the
bounds for columns involved in a single sweep, we can then find just the unique
columns within the sub-array. The result gives us the sort indices of only the
nonempty columns of grid cells and the number of nonempty columns of grid
cells. Then, threads are assigned only to nonempty columns. This can improve
load balancing when empty columns are possible. For this algorithm, the
function `compact` in [@lst:column-parallel] returns the unique entries in its
first argument between indices indicated by the 2^nd^ and 3^rd^ arguments,
respectively. Since `sort` is already sorted, this is a simple task. For each
sweep, $S_sP_s$ has not changed, but we have effectively reduced the work to
a sparse matrix-vector multiply with a sparse matrix that contains no or very
few empty rows. In other words, we avoid having threads attempt to process
an empty column of grid cells (e.g., the first and last columns in
[@fig:pib-columns]), during which the thread would idle while other
threads process nonempty columns of grid cells.

\begin{figure}[tbh]
    \centering
    \begin{tikzpicture}[
        scale=2,
        declare function={
            ex(\t) = 3 + cos(10) * 2.1 * cos(\t) - sin(10) * 0.8 * sin(\t);
            ey(\t) = 2 + cos(10) * 0.8 * sin(\t) + sin(10) * 2.1 * cos(\t);
        }
    ]
    \draw[color=black!20, fill] (1.5, 1) rectangle (2.0, 1.5);
    \draw[color=black!20, fill] (1.5, 3) rectangle (2.0, 3.25);
    \draw[color=black!20, fill] (3.5, 1) rectangle (4.0, 1.5);
    \draw[color=black!20, fill] (3.5, 3) rectangle (4.0, 3.25);
    \draw[color=black!20, fill] (5.5, 1) rectangle (5.75, 1.5);
    \draw[color=black!20, fill] (5.5, 3) rectangle (5.75, 3.25);
    \draw[step=0.5] (0.25, 0.25) grid (5.75, 3.25);

    \node[scale=0.8, shape=circle, color=tol/contrast/blue, fill] at (1.0, 0.5) {};
    \node[scale=0.8, shape=circle, color=tol/contrast/blue, fill] at (1.5, 0.5) {};
    \node[scale=0.8, shape=circle, color=tol/contrast/blue, fill] at (2.0, 0.5) {};
    \node[scale=0.8, shape=circle, color=tol/contrast/blue, fill] at (2.5, 0.5) {};
    \node[scale=0.8, shape=circle, color=tol/contrast/blue, fill] at (1.0, 1.0) {};
    \node[scale=0.8, shape=circle, color=tol/contrast/blue, fill] at (1.5, 1.0) {};
    \node[scale=0.8, shape=circle, color=tol/contrast/blue, fill] at (2.0, 1.0) {};
    \node[scale=0.8, shape=circle, color=tol/contrast/blue, fill] at (2.5, 1.0) {};
    \node[scale=0.8, shape=circle, color=tol/contrast/blue, fill] at (1.0, 1.5) {};
    \node[scale=0.8, shape=circle, color=tol/contrast/blue, fill] at (1.5, 1.5) {};
    \node[scale=0.8, shape=circle, color=tol/contrast/blue, fill] at (2.0, 1.5) {};
    \node[scale=0.8, shape=circle, color=tol/contrast/blue, fill] at (2.5, 1.5) {};
    \node[scale=0.8, shape=circle, color=tol/contrast/blue, fill] at (1.0, 2.0) {};
    \node[scale=0.8, shape=circle, color=tol/contrast/blue, fill] at (1.5, 2.0) {};
    \node[scale=0.8, shape=circle, color=tol/contrast/blue, fill] at (2.0, 2.0) {};
    \node[scale=0.8, shape=circle, color=tol/contrast/blue, fill] at (2.5, 2.0) {};

    \node[scale=0.8, shape=circle, color=tol/contrast/yellow, fill] at (3.0, 0.5) {};
    \node[scale=0.8, shape=circle, color=tol/contrast/yellow, fill] at (3.5, 0.5) {};
    \node[scale=0.8, shape=circle, color=tol/contrast/yellow, fill] at (4.0, 0.5) {};
    \node[scale=0.8, shape=circle, color=tol/contrast/yellow, fill] at (4.5, 0.5) {};
    \node[scale=0.8, shape=circle, color=tol/contrast/yellow, fill] at (3.0, 1.0) {};
    \node[scale=0.8, shape=circle, color=tol/contrast/yellow, fill] at (3.5, 1.0) {};
    \node[scale=0.8, shape=circle, color=tol/contrast/yellow, fill] at (4.0, 1.0) {};
    \node[scale=0.8, shape=circle, color=tol/contrast/yellow, fill] at (4.5, 1.0) {};
    \node[scale=0.8, shape=circle, color=tol/contrast/yellow, fill] at (3.0, 1.5) {};
    \node[scale=0.8, shape=circle, color=tol/contrast/yellow, fill] at (3.5, 1.5) {};
    \node[scale=0.8, shape=circle, color=tol/contrast/yellow, fill] at (4.0, 1.5) {};
    \node[scale=0.8, shape=circle, color=tol/contrast/yellow, fill] at (4.5, 1.5) {};
    \node[scale=0.8, shape=circle, color=tol/contrast/yellow, fill] at (3.0, 2.0) {};
    \node[scale=0.8, shape=circle, color=tol/contrast/yellow, fill] at (3.5, 2.0) {};
    \node[scale=0.8, shape=circle, color=tol/contrast/yellow, fill] at (4.0, 2.0) {};
    \node[scale=0.8, shape=circle, color=tol/contrast/yellow, fill] at (4.5, 2.0) {};

    \draw[very thick] plot[domain=-180:180, variable=\x, samples=100] ({ex(\x)}, {ey(\x)});
    \draw[thin, black, fill=tol/contrast/blue] plot[only marks, mark=square*] coordinates {
        ({ex(225.60289396961724)}, {ey(225.60289396961724)})
        ({ex(236.9674607870579)}, {ey(236.9674607870579)})
    };
    \draw[thin, black, fill=tol/contrast/yellow] plot[only marks, mark=square*] coordinates {
        ({ex(283.4525276213703)}, {ey(283.4525276213703)})
        ({ex(292.8699014257434)}, {ey(292.8699014257434)})
    };
    \draw[thin, black, fill=white] plot[only marks, mark=square*] coordinates {
        ({ex(16.902383319583745)}, {ey(16.902383319583745)})
        ({ex(32.4225566445974)}, {ey(32.4225566445974)})
        ({ex(45.60289396961725)}, {ey(45.60289396961725)})
        ({ex(56.967460787057895)}, {ey(56.967460787057895)})
        ({ex(67.13009857425658)}, {ey(67.13009857425658)})
        ({ex(76.54747237862972)}, {ey(76.54747237862972)})
        ({ex(85.5588196210609)}, {ey(85.5588196210609)})
        ({ex(94.44118037893911)}, {ey(94.44118037893911)})
        ({ex(103.4525276213703)}, {ey(103.4525276213703)})
        ({ex(112.86990142574342)}, {ey(112.86990142574342)})
        ({ex(123.03253921294208)}, {ey(123.03253921294208)})
        ({ex(134.39710603038273)}, {ey(134.39710603038273)})
        ({ex(147.57744335540258)}, {ey(147.57744335540258)})
        ({ex(163.09761668041625)}, {ey(163.09761668041625)})
        ({ex(180.0)}, {ey(180.0)})
        ({ex(196.90238331958375)}, {ey(196.90238331958375)})
        ({ex(212.4225566445974)}, {ey(212.4225566445974)})
        %({ex(225.60289396961724)}, {ey(225.60289396961724)})
        %({ex(236.9674607870579)}, {ey(236.9674607870579)})
        ({ex(247.13009857425658)}, {ey(247.13009857425658)})
        ({ex(256.5474723786297)}, {ey(256.5474723786297)})
        ({ex(265.55881962106093)}, {ey(265.55881962106093)})
        ({ex(274.4411803789391)}, {ey(274.4411803789391)})
        %({ex(283.4525276213703)}, {ey(283.4525276213703)})
        %({ex(292.8699014257434)}, {ey(292.8699014257434)})
        ({ex(303.0325392129421)}, {ey(303.0325392129421)})
        ({ex(314.3971060303827)}, {ey(314.3971060303827)})
        ({ex(327.5774433554026)}, {ey(327.5774433554026)})
        ({ex(343.09761668041625)}, {ey(343.09761668041625)})
        ({ex(360.0)}, {ey(360.0)})
    };
    
    \end{tikzpicture}
    \caption{%
        A portion of $\Omega$ containing $\Gamma$ (thick black curve). The
        Lagrangian points $\Gamma^h$ are represented by squares. All Lagrangian
        points that lie in the grey region are processed in a single sweep.
        Lagrangian points in different grey cells do not have overlapping
        support points, while Lagrangian points in the same cell will all have
        identical support points. If only one thread writes per cell, there
        are no write contentions. The circular support points' colors indicate
        the corresponding square Lagrangian point(s).
    }
    \label{fig:pib-cells}
\end{figure}

An extension of this idea is to refine the partition to as fine as possible.
That is, the Eulerian grid can be divided into $m=n_k$ sets of at most
$\lceil n_\omega/n_k \rceil$ grid cells that are $2Rh$ apart in each direction.
Each grid cell is assigned a unique sort index that orders Lagrangian points by
cell and then by sweep. In sweep $s$, any Lagrangian points in different cells
within the $s$^th^ set of cells will have non-overlapping support points, while
Lagrangian points in the same cell will have the same support points. An
example of this is illustrated in [@fig:pib-cells]. This modification requires
$4^d$ sweeps, but pruning empty grid cells, the work is nearly equal for all
threads, and no thread has no work to do. With one thread per cell in
a sweep, we can compute the support points indices ($i(j,\,k)$) before entering
the **while** loop in [@lst:column-parallel], since they will be the same for
each Lagrangian point in that cell. Other than to precompute the support point
indices, we do not need to change [@lst:column-parallel]; simply change the
matrices $S_s$ to contain only the columns of $S$ corresponding to Lagrangian
points contained in the $s$^th^ set of grid cells and modify the sort-indexing
function $\sigma$ to give unique indices to each grid cell.

\begin{figure}[tbh]
    \centering
    \begin{tikzpicture}[
        scale=2,
        declare function={
            ex(\t) = 3 + cos(10) * 2.1 * cos(\t) - sin(10) * 0.8 * sin(\t);
            ey(\t) = 2 + cos(10) * 0.8 * sin(\t) + sin(10) * 2.1 * cos(\t);
        }
    ]
    \draw[step=0.5] (0.25, 0.25) grid (5.75, 3.25);

    \draw[very thick] plot[domain=-180:180, variable=\x, samples=100] ({3 + 0.98480775301 * 2.1 * cos(\x) - 0.17364817766 * 0.8 * sin(\x)}, {2 + 0.98480775301 * 0.8 * sin(\x) + 0.17364817766 * 2.1 * cos(\x)});
    \foreach \t in {
        16.902383319583745,
        32.4225566445974,
        45.60289396961725,
        56.967460787057895,
        67.13009857425658,
        76.54747237862972,
        85.5588196210609,
        94.44118037893911,
        103.4525276213703,
        112.86990142574342,
        123.03253921294208,
        134.39710603038273,
        147.57744335540258,
        163.09761668041625,
        180.0,
        196.90238331958375,
        212.4225566445974,
        225.60289396961724,
        236.9674607870579,
        247.13009857425658,
        256.5474723786297,
        265.55881962106093,
        274.4411803789391,
        283.4525276213703,
        292.8699014257434,
        303.0325392129421,
        314.3971060303827,
        327.5774433554026,
        343.09761668041625,
        360.0
    } {%
        \draw[very thick, color=tol/contrast/red] ({ex(\t)}, {ey(\t)}) -- %node[draw, black, fill=white] at 
        ({3 + 0.5 * floor(2 * (ex(\t) - 3))}, {2 + 0.5 * floor(2 * (ey(\t) - 2))}) node[fill, circle, scale=0.5] {}
        ;
    }

    \draw[thin, black, fill=white] plot[only marks, mark=square*] coordinates {
        ({ex(16.902383319583745)}, {ey(16.902383319583745)})
        ({ex(32.4225566445974)}, {ey(32.4225566445974)})
        ({ex(45.60289396961725)}, {ey(45.60289396961725)})
        ({ex(56.967460787057895)}, {ey(56.967460787057895)})
        ({ex(67.13009857425658)}, {ey(67.13009857425658)})
        ({ex(76.54747237862972)}, {ey(76.54747237862972)})
        ({ex(85.5588196210609)}, {ey(85.5588196210609)})
        ({ex(94.44118037893911)}, {ey(94.44118037893911)})
        ({ex(103.4525276213703)}, {ey(103.4525276213703)})
        ({ex(112.86990142574342)}, {ey(112.86990142574342)})
        ({ex(123.03253921294208)}, {ey(123.03253921294208)})
        ({ex(134.39710603038273)}, {ey(134.39710603038273)})
        ({ex(147.57744335540258)}, {ey(147.57744335540258)})
        ({ex(163.09761668041625)}, {ey(163.09761668041625)})
        ({ex(180.0)}, {ey(180.0)})
        ({ex(196.90238331958375)}, {ey(196.90238331958375)})
        ({ex(212.4225566445974)}, {ey(212.4225566445974)})
        ({ex(225.60289396961724)}, {ey(225.60289396961724)})
        ({ex(236.9674607870579)}, {ey(236.9674607870579)})
        ({ex(247.13009857425658)}, {ey(247.13009857425658)})
        ({ex(256.5474723786297)}, {ey(256.5474723786297)})
        ({ex(265.55881962106093)}, {ey(265.55881962106093)})
        ({ex(274.4411803789391)}, {ey(274.4411803789391)})
        ({ex(283.4525276213703)}, {ey(283.4525276213703)})
        ({ex(292.8699014257434)}, {ey(292.8699014257434)})
        ({ex(303.0325392129421)}, {ey(303.0325392129421)})
        ({ex(314.3971060303827)}, {ey(314.3971060303827)})
        ({ex(327.5774433554026)}, {ey(327.5774433554026)})
        ({ex(343.09761668041625)}, {ey(343.09761668041625)})
        ({ex(360.0)}, {ey(360.0)})
    };
   
    \end{tikzpicture}
    \caption{%
        For each Lagrangian point, we process a single support point per sweep;
        in this case, the grid point at the lower left corner of the cell the
        Lagrangian point occupies. The index of the support point will be
        different for Lagrangian points in different grid cells, and all
        Lagrangian points in a single grid cell will have the same support
        point index. Using one thread per (nonempty) grid cell for writes,
        there are no write contentions.
    }
    \label{fig:pib-support}
\end{figure}

## A load-balanced partitioning
Until now, we have decomposed the domain into segments to avoid overlapping
support points between these segments. This has led to the potential need for
pruning when segments contain no Lagrangian points. Even with pruning, there
is no guarantee that threads will perform similar amounts of work. However,
ignoring boundaries, we know that each Lagrangian point has $n_k$ (possibly
zero) $\delta_h$ values to compute. Consider using a sweep to compute one of
the $\delta_h$ values. [@fig:pib-support] illustrates a sweep in which for each
Lagrangian point, $\delta_h$ is evaluated at the support point at the lower
left corner of the grid cell containing the Lagrangian point. We store these
values in a temporary buffer.


\begin{algorithm}[tbh]
    \begin{algorithmic}
        \REQUIRE $K = K_1 \cup K_2 \cup \cdots \cup K_n$
        \STATE $\vec{f} \leftarrow \vec{0}$
        \FORALL {$\vec{X}_j\in \Gamma^h$ \textbf{parallel}}
        \STATE \texttt{index}$_j \leftarrow j$
        \STATE \texttt{sort}$_j \leftarrow \sigma(\vec{X}_j)$
        \ENDFOR
        \STATE \texttt{sort-by-key}(\texttt{sort}, \texttt{index})
        \FOR {$s = 1,\,\ldots,\,n$}
        \FORALL {$k \in K_s$}
        \STATE $\vec{f}^{\ k} \leftarrow \vec{0}$
        \ENDFOR
        \FORALL {$\vec{X}_j \in \Gamma^h$ \textbf{parallel}}
        \STATE $\ell \leftarrow \texttt{index}_j$
        \STATE \texttt{buffer}$_j \leftarrow F_\ell A_\ell\vec{\nu}(\vec{X}_\ell,\,s)$
        \ENDFOR
        \STATE $\texttt{keys},\,\texttt{values} \leftarrow \texttt{reduce-by-key}(\texttt{sort},\,\texttt{buffer})$
        \FORALL {\texttt{key}, \texttt{value} in \texttt{keys}, \texttt{values} \textbf{parallel}}
        \FORALL {$k \in K_s$}
        \STATE $\ell \leftarrow \texttt{grid-index}(\texttt{key},\,k)$
        \IF {$\vec{x}_\ell \in \Omega$}
        \STATE $f^{\ k}_\ell \leftarrow f^{\ k}_\ell + \texttt{value}$
        \ENDIF
        \ENDFOR
        \ENDFOR
        \FORALL {$k \in K_s$}
        \STATE $\vec{f} \leftarrow \vec{f} + \vec{f}^{\ k}$
        \ENDFOR
        \ENDFOR
    \end{algorithmic}
    \caption{Sweep-fused support-point-wise force spreading}
    \label{lst:lagrange-parallel}
\end{algorithm}

To each grid cell, we assign a unique sort index, e.g., the index of the grid
cell in lexicographical ordering. By sorting the Lagrangian points according
to the sort index of the grid cell that contains it, the indices of the support
point for each sweep are also grouped, though not necessarily ordered. The
$\delta_h$ values can then be accumulated using a \emph{segmented reduce}
routine, which sums consecutive values given some predicate; in this case, that
the values have the same associated sort index. This algorithm is listed in
[@lst:lagrange-parallel]. We satisfy the requirement that $K$ be partitioned
by letting $K_s = \{s\}$, so that $n = n_k$ in the algorithm.

The function `grid-index` in [@lst:lagrange-parallel] performs a similar
role as the function $i$ in the previous sections. Since all Lagrangian points
in the same grid cell will have the same sort index, so will a theoretical
point at the bottom left corner of the cell. These points will all have the
same support points, so we identify these points with the sort index of this
theoretical point. For a Lagrangian point $\vec{X}_j$ with sort index
`key`, $\texttt{grid-index}(\texttt{key},\,k) \equiv i(j,\,k)$.

One thing to notice that differs from [@lst:column-parallel] is that we have
eliminated the \textbf{while} loop, which prevented us from proper load
balancing. The work done in the two parallel \textbf{for} loops can be divided
evenly among the threads, and the functions `sort-by-key` and
`reduce-by-key` can also be parallelized effectively.

In each sweep, we compute one $\delta_h$ value per Lagrangian point. Since
`sort` is supplied to `reduce-by-key` each sweep, the values will be
accumulated according to the same pattern, and `keys` will be identical each
sweep, containing only the unique sort indices among all of the Lagrangian
points. For each `key` in `keys`, we write one value to $\vec{f}$. Thus, each
sweep performs identical work, up to permutations in threads. One benefit of
each sweep performing the same work is that we can fuse some of the sweeps and
simply write to separate buffers for each support point. While it may be
tempting to fuse all of the sweeps into one, for large numbers of Lagrangian
points, this can lead to heavy memory pressure and can hurt performance. It
seems that, depending on the device, fusing 1--8 sweeps may be optimal.
This variation is listed in [@lst:lagrange-parallel] with the partition
$\{K_s\}$ chosen to have a maximum number of entries, depending on the device,
and perhaps other properties that aid in minimizing calls to $\varphi$.
