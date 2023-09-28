

---
<p align = "center" style="font-size:400%;">
<b>NX Link Prediction Module Extension</b></p>

---

<p align = "center">Kamil P. Orzechowski</p>
<p align = "justify"> Complex networks are a useful tool for exploring numerically large, statistically described systems.
An interesting, interdisciplinary topic in the context of their exploration is predictive analysis in the
form of predicting the existence of internodal connections [1]. An important element of the discussed
issue are the so-called local node similarity measures [1–3] enabling the quantification of this feature
in relation to a pair of nodes taking into account their immediate neighborhood.

Their study very often requires access to digital computing resources which makes them a tool eagerly
used by programmers. In their work they use various types of ready-made and optimized solutions -
libraries (dedicated to networks) of various programming languages. The most popular is NetworkX [4]
for Python. This library contains existing link prediction modules but they are very poor and contain
a small number of similarity measures offered for calculations and only for undirected networks.

<b> NX_Link_Prediction_KO</b> module is carried out as part of the project:

<p align = "center"><i>Exploration and implementation of link prediction methods for various types of tie asymmetry in complex networks</i></p>

realized within CyberSummer@WUT-3 competition | POB Research Centre Cybersecurity and Data Science of Warsaw University of Technology.

The efforts were made to extend the currently existing Link Prediction [5] module of the NetworkX library. The
module was implemented and made publicly available, compatible with the one present in the library,
providing users with 22 new similarity measures for undirected and/or weighted networks (almost half
of which were measures based on apparent edge asymmetry, i.e. such that determine the directions
of edges in undirected networks). Additionally, 10 metrics are implemented for naturally asymmetric
directed (and/or weighted) networks. All 32 measures have not been previously available in the library
and the module created as part of this project</p>


### References
[1] Liben-Nowell D., Kleinberg J. The link-prediction problem for social networks. J. Am. Soc. Inf. Sci. Technol., 58:1019–1031, 2007.
[2] Martinčić-Ipšić S., Močibob E., Perc M. Link prediction on Twitter. PLOS ONE, 12:7, 2017. <br>
[3] Martínez V., Berzal F., Cubero J.-C. A Survey of Link Prediction in Complex Networks. ACM Comput. Surv., 49:1–33, 2016. <br>
[4] NetworkX Library https://networkx.org/documentation/stable/index.html. <br>
[5] NetworkX Link Prediction. https://networkx.org/documentation/stable/reference/algorithms/link_prediction.html. <br>
