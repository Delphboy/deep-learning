# Set Theory

## Definitions

A set is simply a collection of objects where order doesn’t matter.

$$
A := \{1, 2, 3, 4\}
$$
$$
B := \{4, 2, 3, 1\}
$$

As order doesn’t matter, $A = B$

The elements of a set are assumed to be distinct, even if duplicates exist. Therefore:

$$
C := \{1,1,2,3,4\}
$$
$$
A = C
$$

A large set, or an infinite set can be defined with a vertical bar that reads “such that”, followed by a condition

$$
D := \{x | x\text{ is a positive, even integer}\}
$$

It is also possible to have a set of sets: $E = \{ \mathbb{R}, \mathbb{Z} \}$

The **cardinality** of a set is denoted $|A|$ and gives the number of elements in the set. ie) $|A| = 4$ and $|\mathbb{Z}| = \text{infinity}$


An empty set is referred to as the **null** set or the **void** set and is denoted with $\emptyset$, where $\emptyset = \{\}$.

An ordered set is denoted using $()$

$$
P = (a, b)
$$
$$
Q = (c, d)
$$
$$
(a, b) \neq (b, a)
$$
$$
P = Q \text{ iff } a = c \text{ and } b = d
$$


## Equality and Membership

If $x$ belongs to the set $X$ then $x \in X$, else $x \notin X$

For $X = Y$ both the following must be true:

For every $x$, if $x \in X$, then $x \in Y$

AND

For every $x$, if $x \in Y$, then $x \in X$

## Subsets

When a set contains all the elements of a subset in addition to others, the second set is said to be a subset of the first set.

$$
A = \{1, 2, 3, 4\} \\
B = \{3, 2\} \\
B \subseteq A
$$

<!-- <aside> -->
**Note:** Any set $X$ is a subset of itself. The null set is a subset of all sets
<!-- </aside> -->

If $X$ is a subset of $Y$ and $X \neq Y$, then $X$ is a proper subset of $Y$, denoted $X \subset Y$

The set of subsets of a set $X$, regardless of whether they are proper or not, is called the **power set** and denoted $\mathcal{P}(X)$

$$
A = \{a, b ,c\}
$$
$$
\mathcal{P}(A) = \emptyset, \{a\}, \{b\}, \{c\}, \{a, b\}, \{a, c\}, \{b, c\}, \{a, b, c\}
$$

It is worth noting that in general $|\mathcal{P}(X)| = 2^{|X|}$

## Set Operations

### Union $\cup$

The union operator creates a new set consisting of all the elements belonging to either of the sets being joined or elements that belong to both of them.

Formally: $X \cup Y = \{x | x \in X \text{   or   } x \in Y\}$


$$
A = \{1, 3, 4\}
$$
$$
B = \{2, 5\}
$$
$$
A \cup B = \{1, 3, 4, 2, 5\}
$$

### Intersection $\cap$

The intersection operator creates a new set containing only the elements that belong to **both** sets.

Formally: $X \cap Y = \{x | x\in X \text{ and } x \in Y\}$

$$
A = \{2, 3\} 
$$
$$
B = \{1, 2\}
$$
$$
A \cap B = \{2\}
$$

### Difference (Relative Complement) $-$

The difference operator returns the set of elements that are in the first set but not in the second. Formally: $X - Y = \{x | x \in X \text{ and } x \notin Y\}$

$$
A = \{1, 2, 3\}
$$
$$
B = \{1, 2\} 
$$
$$
A - B = \{3\}
$$

<!-- <aside> -->
**Note:** $A - B \neq B - A$
<!-- </aside> -->


### Cartesian Product $\times$
The cartesian product of $X$ and $Y$ gives the set of all ordered pairs $(x, y)$ where

$x\in X \text{ and } y \in Y$

$$
X = \{1, 2, 3\}
$$
$$
Y = \{a, b\}
$$
$$
X \times Y = \{(1, a), (1, b), (2, a), (2, b), (3, a), (3, b)\}
$$
$$
Y \times X = \{(a, 1), (b, 1), (a, 2), (b, 2), (a, 3), (b, 3)\}
$$
$$
X \times Y \neq Y \times X
$$


<!-- <aside> -->
**Note:** $|X \times Y| = |X| \cdot |Y|$

<!-- </aside> -->

## Disjoint and Complement Sets

A pair of sets are said to be **disjoint** if no elements belong to both sets, in other words:

 $X \cap Y = \empty$

Given a set of sets $\mathcal{S}$, it is said to be **pairwise disjoint** if given two distinct sets $X$ and $Y$, then $X \cap Y = \empty$. ie) $\mathcal{S} = \{\{1, 4, 5\}, \{2,6\}, \{3\}\}$ is pairwise disjoint.

If working with sets that are all a subset of some larger set, $U$, this larger set is referred to as the **universal set**. It must be explicitly given or inferred through context. 

If $X \subset U$, the **complement** of $X$ is given as $\bar X = U - X$ 
