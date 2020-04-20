# Introduction {#sec:intro}
Many problems in biophysics involve the interaction of an incompressible fluid
and an immersed elastic interface. The solution to these problems can be
approximated using the immersed boundary (IB) method, which was developed by
Peskin to simulate blood flow through the heart [@Peskin:1972wa]. This method
couples the equations governing fluid velocity and those governing elastic
forces via operations we will refer to as _interpolation_ and _spreading_. We
treat solving the fluid equations and computing elastic forces as an
implementation detail; we assume that they are parallelizable and limit our
focus to the interpolation and spreading operations.

<!--[@Layton:2011um] compare the efficacy of GPUs, compared to CPUs, to simulate
the 2d flow in the presence of an object with prescribed surface motion. In
their formulation, they have an object fixed in space, and the surface tension
forces are computed as a result of the constraint on the object's motion. With
a stationary object, spreading and interpolation weights can be precomputed and
reused throughout the simulation. For an object moving in space, the weights
will differ at each timestep.-->

McQueen and Peskin present a domain decomposition scheme to parallelize these
operations on the Cray C-90 computer with a modest number of vector processors
[@McQueen:1997kw]. Unfortunately, the trend in computing is to use _more_
processors rather than _faster_ processors. They also illustrate the need for
a fast interpolation and spreading -- even parallelized, they spend roughly
half of the wall clock time spreading and interpolating. Here, we introduce
a novel parallelization scheme for the interpolation and spreading operations.
We compare this to the method of Peskin and McQueen and a variant of their
ideas on a massively parallel general purpose graphical processing unit
(GPGPU).

## Summary of the immersed boundary method {#sec:ib}
Consider a $d$-dimensional ($d=2$ or 3) rectangular domain $\Omega$, which is
filled with a viscous incompressible fluid with constant viscosity $\mu$ and
density $\rho$, and contains an immersed elastic structure, $\Gamma$. The
structure is impermeable to the fluid and moves at the local fluid velocity, is
deformed by this motion, and imparts a force on the fluid. Otherwise, the
interface is treated as part of the fluid. 

The fluid velocity, $\vec{u} = \vec{u}(\vec{x},\,t)$, is governed by the
incompressible Navier-Stokes equations for a Newtonian fluid,
\begin{gather*}
    \rho(\vec{u}_t + \vec{u}\cdot\grad\vec{u}) = \mu\Delta\vec{u} - \grad p + \vec{f}, \\
    \grad\cdot\vec{u} = 0,
\end{gather*}
where $p$ is the fluid pressure and $\vec{f}$ is the elastic force density.
This is a set of $d+1$ equations in $d+1$ unknowns: the $d$ components of
$\vec{u}$, and $p$. The equations are written relative to the Eulerian frame,
so that the coordinates $\vec{x}$ are independent variables. Quantities in the
Eulerian frame are written in the lower case Latin alphabet.

Let $\vec{X}=\vec{X}(\vec{\theta},\,t)$ represent a parametrization of the
Cartesian coordinates of an immersed interface with material coordinates
$\vec{\theta}$ at time $t$. Let $\mathcal{E}[\vec{X}]$ be the energy density
functional for the elastic interface material. The elastic force density is
computed by evaluating the Fréchet derivative of $\mathcal{E}$,
\begin{gather*}
    \vec{F} = \delta \mathcal{E}[\vec{X}],
\end{gather*}
where $\delta$ represents the first variation. Upper case Latin letters
represent Lagrangian quantities and are functions of $\vec{\theta}$ and $t$.

To couple the fluid and interface, we employ the Dirac delta function,
$\delta(\vec{x}-\vec{X}(\vec{\theta},\,t))$. Analaytically, the fluid-interface
interactions can be written
\begin{gather}
    \label{eq:interpolation}
    \dot{\vec{X}} = \int_\Omega \delta(\vec{x}-\vec{X}) \vec{u}(\vec{x},\,t)\d\vec{x}, \text{and} \\
    \label{eq:spreading}
    \vec{f} = -\int_\Gamma \delta(\vec{x}-\vec{X})\vec{F}(\vec{X})\d\vec{X},
\end{gather}
where $\dot{\vec{X}}$ represents the derivative of $\vec{X}$ with respect to
$t$. [@eq:interpolation] is called _interpolation_, and the result of the
right-hand side is the fluid velocity at $\vec{X}$; namely, $\dot{\vec{X}}
= \vec{u}(\vec{X},\,t)$. [@eq:spreading] is called _spreading_, because while
$\vec{F}$ has units of force per unit _area_ on $\Gamma$, $\vec{f}$ has units
of force per unit _volume_ in $\Omega$. The force $\vec{F}\d\vec{X}$ over area
$\d\vec{X}$ is "spread" to the force $\vec{f}\d\vec{x}$ over volume
$\d\vec{x}$. 

Each of the fluid equations is discretized on a regular grid of spacing $h$ so
that $\Omega$ is divided into cubic cells of side length $h$. The grids can be
collocated or staggered. Because of the checkerboard instability (see, e.g.,
[@Wesseling:2001ci]) in solving the Navier-Stokes equations on collocated
regular grids, we will assume that the grids are staggered. This means that
different components of a single Eulerian vector quantity, such as $\vec{u}$,
might be evaluated at different locations. However, corresponding components of
vector-valued Eulerian quantities, and those of $\vec{u}$ and $\vec{f}$ in
particular, are assumed to be discretized on the same grid. A fixed Lagrangian
point may reside in different grid cells for each different grid. For any
one of the Eulerian grids, we use $\vec{x}_i$ to refer to the $i$^th^ Eulerian
grid point for the chosen grid. The set of Eulerian grid points for this grid
will be denoted $\Omega^h$, and define $n_\omega = |\Omega^h|$.

The Lagrangian force density $\vec{F}$ is evaluated at a set of points,
usually a _fixed_ set of points in the Lagrangian variables, $\vec{\theta}$.
The notation $\vec{X}_j=\vec{X}(\vec{\theta}_j,\,t)$ refers to an individual
Lagrangian point. The typical heuristic for distributing the points $\vec{X}_j$
on the elastic interface is that neighboring Lagrangian points be at most $h$
apart from one another, and often at most $h/2$ apart. We denote the set of
Lagrangian points by $\Gamma^h$ and define $n_\gamma = |\Gamma^h|$ to be the number
of Lagrangian points.

\begin{figure}[thb]
    \centering
    \begin{tikzpicture}
        \begin{axis}[ymin=-0.09, ymax=0.59, xmin=-2.5, xmax=2.5, axis lines=center, xlabel={$r$}, ylabel={$y$}, xlabel style={right}, ylabel style={above}, smooth, no markers]
            \addplot[samples=400, domain=-2: 2, very thick, black] {0.25 * (1+cos(90*x))} node[midway, above right] {$y=\varphi(r)$};
            \addplot[samples=400, domain=-3:-2, very thick, black] {0};
            \addplot[samples=400, domain= 2: 3, very thick, black] {0};
        \end{axis}
    \end{tikzpicture}
    \caption{%
        A compactly-supported approximation to the Dirac delta function. The
        quantity $r$ is the difference in position of an Eulerian and
        a Lagrangian point in units of grid spaces. For $r\in[0,\,1)$, only
        $\varphi(r-2)$, $\varphi(r-1)$, $\varphi(r)$, and $\varphi(r+1)$ are
        nonzero for points spaced 1 apart.
    }
    \label{fig:base-delta}
\end{figure}

The singular [@eq:interpolation;@eq:spreading] do not lend themselves easily to
evaluation. In particular, it is unlikely that Lagrangian points and Eulerian
grid points will coincide. For a regular grid with spacing $h$, we replace the
Dirac $\delta$-function with a regularized kernel, $\delta_h$, which is
a Cartesian product of one-dimensional kernels, $h^{-1}\varphi(h^{-1}x)$. One
choice for $\varphi$ is shown in [@fig:base-delta]. For kernel $\varphi$, we
define $s_\varphi = |\textrm{supp}(\varphi)\cap\mathbb{Z}|-1$, which is the
number of grid spaces of support for $h^{-1}\varphi(h^{-1}x)$. The kernel
$\delta_h(\cdot-\vec{X})$ is therefore nonzero at, at most, $s_\varphi^d$ grid
points, called the \emph{support} (\emph{grid}) \emph{points} of
$\delta_h(\cdot-\vec{X})$. We will drop the subscript $s = s_\varphi$ when
unambiguous.

Discretizing a single component of [@eq:interpolation;@eq:spreading] yields
\begin{gather}
    \label{eq:interp-disc}
    \dot{X}_j = \sum_i h^d\delta_{h}(\vec{x}_i-\vec{X}_j)u_i := \mathcal{S}^\dagger\vec{u},\ \text{and} \\
    \label{eq:spread-disc}
    f_i = \sum_j \delta_h(\vec{x}_i-\vec{X}_j)F_j := \mathcal{S}\vec{F},
\end{gather}
respectively. The quantities $\dot{X}_j$ and $F_j$ are the fluid velocity and
Lagrangian force, respectively, at $\vec{X}_j$; $u_i$ and $f_i$ are the fluid
velocity and Eulerian force density, respectively, at $\vec{x}_i$. In
[@eq:interp-disc], $S^\dagger=(\delta_h(\vec{x}_i-\vec{X}_j))$ is the matrix
of kernel values, and $\mathcal{S}$ is its transpose. If $s^d \ll n_\omega$,
$\mathcal{S}^\dagger$ and $\mathcal{S}$ are sparse.

We treat the method of solving the incompressible Navier-Stokes equations as
a black box to which we give a routine that accepts fluid velocities and
computes Eulerian force densities. The solver couples the Eulerian grids, so
the fluid velocities and force densities are represented by one vector per
component. We will use the notation $\{\vec{u}\}$ and $\{\vec{f}\}$ for the
set of vectors containing fluid velocities and Eulerian force densities,
respectively.

## A survey of parallel architectures {#sec:arch}
The algorithm described below aims to be device agnostic. To that end, the
algorithm ought to behave nicely on devices with strict limitations on
parallelization, like graphical processing units (GPUs). GPUs use a
single-instruction, multiple thread (SIMT) execution model in which a group of
threads execute the same instruction on different data, in parallel.

* Branch divergence hurts performance
* Can't do two independent things at the same time
* Time taken is the time it takes for the slowest thread to finish (hence a
  need for load balancing)
* SIMT can be mimicked to some extent on CPUs

Henceforth, we will use the term _thread_ to refer to a single computational
unit (e.g., thread, processor, &c.) under the restrictions outlined above with
the caveat that parallel primitives may be optimized for a specific devices.
