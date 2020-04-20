% A fine-grained parallel implementation of the immersed boundary method

---
authors:
    - name: Andrew Kassen
      abbreviation: A. Kassen
      institution: |
        Department of Mathematics \newline
        University of Utah
    - name: Varun Shankar
      abbreviation: V. Shankar
      institution: |
        School of Computing \newline
        University of Utah
    - name: Aaron Fogelson
      abbreviation: A. Fogelson
      institution: |
          Department of Mathematics \newline
          University of Utah \newline
          155 South 1400 East, JWB 233 \newline
          Salt Lake City, UT 84112
      telephone: +1801...
      fax: +1801...
      email: fogelson@math.utah.edu
cref: True
abstract:
    \noindent This note describes a parallel implementation of the immersed
    boundary method which relies on the well-studied parallel primitives
    key-value sort and segmented reduce. This makes the algorithms feasible for
    use on general purpose graphical processing units in addition to most other
    architectures. This is compared to existing parallelization methods.
---
