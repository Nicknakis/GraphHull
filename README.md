# GraphHull

Python 3.8.3 and Pytorch 1.12.1 implementation of the Archetypal Graph Generative Models: Explainable and Identifiable Communities via Anchor-Dominant Convex Hulls spotlight paper, as is to appear in the 29th International Conference on Artificial Intelligence and Statistics (AISTATS) 2026, Tangier, Morocco. PMLR: Volume 300. Copyright 2026 by the
author(s).

## Description

Representation learning has been essential for graph machine learning
tasks such as link prediction, community detection, and network visualization. Despite recent advances in achieving high performance on these downstream tasks, little progress has been made toward self-explainable models. Understanding the patterns behind predictions is equally important, motivating recent interest in explainable machine learning. In this paper, we present GraphHull, an explainable generative model that represents networks using two levels of convex hulls. At the global level, the vertices of a convex hull are treated as \emph{archetypes}, each corresponding to a pure community in the network. At the local level, each community is refined by a prototypical hull whose vertices act as representative profiles, capturing community-specific variation. This two-level construction yields clear multi-scale explanations: a node’s position relative to global archetypes and its local prototypes directly accounts for its edges. The geometry is well-behaved by design, while local hulls are kept disjoint by construction. To further encourage diversity and stability, we place principled priors, including determinantal point processes, and fit the model under MAP estimation with scalable subsampling. Experiments on real networks demonstrate the ability of GraphHull to recover multi-level community structure and to achieve competitive or superior performance in link prediction and community detection, while naturally providing interpretable predictions.



## Reference

[Archetypal Graph Generative Models: Explainable and Identifiable Communities via Anchor-Dominant Convex Hulls](). Nikolaos Nakis, Chrysoula Kosma, Panagiotis Promponas, Michail Chatzianastasis, and Giannis Nikolentzos, AISTATS 26 (Spotlight Paper)

