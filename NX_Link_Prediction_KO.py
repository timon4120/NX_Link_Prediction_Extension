# ################################################################################
# Extension of the NetworkX link prediction module as part of the project:
#
# "Exploration and implementation of link prediction methods for various types of
# tie asymmetry in complex networks" 
# 
# CyberSummer@WUT-3 competition funded by POB Research Centre Cybersecurity and
# Data Science. Warsaw University of Technology.
# ################################################################################
# Filename: 	NX_Link_Prediction_KO.py		
# Author: 	    Kamil P. Orzechowski <01141361@pw.edu.pl>
# Version:		0.1
# ################################################################################
# NOTE: Some of the code contained in the module (for its correct operation and 
# compatibility with NetworkX) comes from module link_prediction of this library. 
# Source: https://networkx.org/documentation/stable/_modules/networkx/algorithms/link_prediction.html
# ################################################################################



"""
Link prediction algorithms.
"""

from math import log
from math import sqrt
import random #NEW IMPORT

import networkx as nx
from networkx.utils import not_implemented_for

__all__ = [
    #ALREADY EXISTED MEASURES (8)
    "resource_allocation_index",
    "jaccard_coefficient",
    "adamic_adar_index",
    "preferential_attachment",
    "cn_soundarajan_hopcroft",
    "ra_index_soundarajan_hopcroft",
    "within_inter_cluster",
    "common_neighbor_centrality",
    #NEW MEASURES (32)
    "AT1",
    "AT2",
    "AT3",
    "DAA",
    "DCN",
    "DRA",
    "QA",
    "QQ",
    "directed_HDI_index",
    "directed_HPI_index",
    "directed_Jaccard_coefficient",
    "directed_LHN_index",
    "directed_bifan_index",
    "directed_salton_index",
    "directed_sørensen_index",
    "hub_depressed_index",
    "hub_promoted_index",
    "inverse_selectivity",
    "leicht_holme_newman_index",
    "mix_index_1",
    "mix_index_2",
    "not_implemented_for",
    "salton_index",
    "selectivity",
    "sorensen_index",
    "wAT1",
    "wAT2",
    "wAT3",
    "weighted_adamic_adar_index",
    "weighted_common_neighbours_index",
    "weighted_jaccard_coefficient",
    "weighted_preferential_attachment",
    "weighted_resource_allocation_index"
]

def _apply_prediction(G, func, ebunch=None):
    """Applies the given function to each edge in the specified iterable
    of edges.

    `G` is an instance of :class:`networkx.Graph`.

    `func` is a function on two inputs, each of which is a node in the
    graph. The function can return anything, but it should return a
    value representing a prediction of the likelihood of a "link"
    joining the two nodes.

    `ebunch` is an iterable of pairs of nodes. If not specified, all
    non-edges in the graph `G` will be used.

    """
    if ebunch is None:
        ebunch = nx.non_edges(G)
    return ((u, v, func(u, v)) for u, v in ebunch)

# ----------------------------------------------------------------------------------------------------
# HERE IS THE PLACE FOR SIMILARITY MEASURES ALREADY EXISTING IN NETWORKX
# https://networkx.org/documentation/stable/_modules/networkx/algorithms/link_prediction.html
# ...
# ----------------------------------------------------------------------------------------------------

# ====================================================================================================
# NEW UNDIRECTED SIMILARITY MEASURES
# ====================================================================================================

@not_implemented_for("directed")
@not_implemented_for("multigraph")
def weighted_jaccard_coefficient(G, ebunch=None):
    r"""

    Parameters
    ----------
    G : graph
        NetworkX undirected weighted graph.

    ebunch : iterable of node pairs, optional (default = None)
        Weighted Jaccard Coefficient will be computed for each pair of nodes given
        in the iterable. The pairs must be given as 2-tuples (u, v)
        where u and v are nodes in the graph. If ebunch is None then all
        non-existent edges in the graph will be used.
        Default value: None.

    Returns
    -------
    piter : iterator
        An iterator of 3-tuples in the form (u, v, p) where (u, v) is a
        pair of nodes and p is their weighted Jaccard Coefficient.

    Examples
    --------
    >>> G = nx.Graph()
    >>> G.add_weighted_edges_from((u,v,random.random()) for u,v in nx.complete_graph(5).edges())
    >>> preds = weighted_jaccard_coefficient(G, [(0, 1), (2, 3)])
    >>> for u, v, p in preds:
    ...     print(f"({u}, {v}) -> {p:.8f}")
    (0, 1) -> 0.47875632
    (2, 3) -> 0.62803138

    References
    ----------
    .. [1] Martinčić-Ipšić S., Močibob E., Perc M. Link prediction on Twitter. PLOS ONE, 12:7, 2017
        https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0181079&type=printable
    """
    def predict(u, v):
        cn = nx.common_neighbors(G, u, v)
        cw = sum([G.get_edge_data(u,z)["weight"] + G.get_edge_data(v,z)["weight"] for z in cn])
        s_u_v = G.degree(u,weight='weight') + G.degree(v,weight='weight')        
        return cw / s_u_v

    return _apply_prediction(G, predict, ebunch)

@not_implemented_for("directed")
@not_implemented_for("multigraph")
def weighted_adamic_adar_index(G, ebunch=None):
    r"""

    Parameters
    ----------
    G : graph
        NetworkX undirected weighted graph.

    ebunch : iterable of node pairs, optional (default = None)
        Weighted Adamic Adar Index will be computed for each pair of nodes given
        in the iterable. The pairs must be given as 2-tuples (u, v)
        where u and v are nodes in the graph. If ebunch is None then all
        non-existent edges in the graph will be used.
        Default value: None.

    Returns
    -------
    piter : iterator
        An iterator of 3-tuples in the form (u, v, p) where (u, v) is a
        pair of nodes and p is their weighted Adamic Adar Index.

    Examples
    --------
    >>> G = nx.Graph()
    >>> G.add_weighted_edges_from((u,v,random.random()) for u,v in nx.complete_graph(5).edges())
    >>> preds = weighted_adamic_adar_index(G, [(0, 1), (2, 3)])
    >>> for u, v, p in preds:
    ...     print(f"({u}, {v}) -> {p:.8f}")
    (0, 1) -> 2.50521604
    (2, 3) -> 3.63121955

    References
    ----------
    .. [1] Martinčić-Ipšić S., Močibob E., Perc M. Link prediction on Twitter. PLOS ONE, 12:7, 2017
        https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0181079&type=printable
    """
    def predict(u, v):
        cn = nx.common_neighbors(G, u, v)     
        return sum([(G.get_edge_data(u,z)["weight"] + G.get_edge_data(v,z)["weight"])/log(1 + G.degree(z,weight='weight')) for z in cn]) 

    return _apply_prediction(G, predict, ebunch)

@not_implemented_for("directed")
@not_implemented_for("multigraph")
def weighted_resource_allocation_index(G, ebunch=None):
    r"""

    Parameters
    ----------
    G : graph
        NetworkX undirected weighted graph.

    ebunch : iterable of node pairs, optional (default = None)
        Weighted Resource Allocation Index will be computed for each pair of nodes given
        in the iterable. The pairs must be given as 2-tuples (u, v)
        where u and v are nodes in the graph. If ebunch is None then all
        non-existent edges in the graph will be used.
        Default value: None.

    Returns
    -------
    piter : iterator
        An iterator of 3-tuples in the form (u, v, p) where (u, v) is a
        pair of nodes and p is their weighted Resource Allocation Index.

    Examples
    --------
    >>> G = nx.Graph()
    >>> G.add_weighted_edges_from((u,v,random.random()) for u,v in nx.complete_graph(5).edges())
    >>> preds = weighted_resource_allocation_index(G, [(0, 1), (2, 3)])
    >>> for u, v, p in preds:
    ...     print(f"({u}, {v}) -> {p:.8f}")
    (0, 1) -> 2.50521604
    (2, 3) -> 3.63121955

    References
    ----------
    .. [1] Martinčić-Ipšić S., Močibob E., Perc M. Link prediction on Twitter. PLOS ONE, 12:7, 2017
        https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0181079&type=printable
    """
    def predict(u, v):
        cn = nx.common_neighbors(G, u, v)     
        return sum([(G.get_edge_data(u,z)["weight"] + G.get_edge_data(v,z)["weight"])/G.degree(z,weight='weight') for z in cn]) 

    return _apply_prediction(G, predict, ebunch)

@not_implemented_for("directed")
@not_implemented_for("multigraph")
def weighted_common_neighbours_index(G, ebunch=None):
    r"""

    Parameters
    ----------
    G : graph
        NetworkX undirected weighted graph.

    ebunch : iterable of node pairs, optional (default = None)
        Weighted Common Neighbours Index will be computed for each pair of nodes given
        in the iterable. The pairs must be given as 2-tuples (u, v)
        where u and v are nodes in the graph. If ebunch is None then all
        non-existent edges in the graph will be used.
        Default value: None.

    Returns
    -------
    piter : iterator
        An iterator of 3-tuples in the form (u, v, p) where (u, v) is a
        pair of nodes and p is their weighted Common Neighbours Index.

    Examples
    --------
    >>> G = nx.Graph()
    >>> G.add_weighted_edges_from((u,v,random.random()) for u,v in nx.complete_graph(5).edges())
    >>> preds = weighted_common_neighbours_index(G, [(0, 1), (2, 3)])
    >>> for u, v, p in preds:
    ...     print(f"({u}, {v}) -> {p:.8f}")
    (0, 1) -> 3.68159689
    (2, 3) -> 3.72592359

    References
    ----------
    .. [1] Martinčić-Ipšić S., Močibob E., Perc M. Link prediction on Twitter. PLOS ONE, 12:7, 2017
        https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0181079&type=printable
    """
    def predict(u, v):
        cn = nx.common_neighbors(G, u, v)     
        return sum([(G.get_edge_data(u,z)["weight"] + G.get_edge_data(v,z)["weight"]) for z in cn]) 

    return _apply_prediction(G, predict, ebunch)

@not_implemented_for("directed")
@not_implemented_for("multigraph")
def weighted_preferential_attachment(G, ebunch=None):
    r"""

    Parameters
    ----------
    G : graph
        NetworkX undirected weighted graph.

    ebunch : iterable of node pairs, optional (default = None)
        Weighted Preferential Attachment Index will be computed for each pair of nodes given
        in the iterable. The pairs must be given as 2-tuples (u, v)
        where u and v are nodes in the graph. If ebunch is None then all
        non-existent edges in the graph will be used.
        Default value: None.

    Returns
    -------
    piter : iterator
        An iterator of 3-tuples in the form (u, v, p) where (u, v) is a
        pair of nodes and p is their weighted Preferential Attachment Index.

    Examples
    --------
    >>> G = nx.Graph()
    >>> G.add_weighted_edges_from((u,v,random.random()) for u,v in nx.complete_graph(5).edges())
    >>> preds = weighted_preferential_attachment(G, [(0, 1), (2, 3)])
    >>> for u, v, p in preds:
    ...     print(f"({u}, {v}) -> {p:.8f}")
    (0, 1) -> 4.06317531
    (2, 3) -> 2.98878114

    References
    ----------
    .. [1] Martinčić-Ipšić S., Močibob E., Perc M. Link prediction on Twitter. PLOS ONE, 12:7, 2017
        https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0181079&type=printable
    """
    def predict(u, v):  
        return G.degree(u,weight='weight') * G.degree(v,weight='weight')

    return _apply_prediction(G, predict, ebunch)

@not_implemented_for("directed")
@not_implemented_for("multigraph")
def selectivity(G, ebunch=None):
    r"""

    Parameters
    ----------
    G : graph
        NetworkX undirected weighted graph.

    ebunch : iterable of node pairs, optional (default = None)
        Selectivity will be computed for each pair of nodes given
        in the iterable. The pairs must be given as 2-tuples (u, v)
        where u and v are nodes in the graph. If ebunch is None then all
        non-existent edges in the graph will be used.
        Default value: None.

    Returns
    -------
    piter : iterator
        An iterator of 3-tuples in the form (u, v, p) where (u, v) is a
        pair of nodes and p is their Selectivity.

    Examples
    --------
    >>> G = nx.Graph()
    >>> G.add_weighted_edges_from((u,v,random.random()) for u,v in nx.complete_graph(5).edges())
    >>> preds = selectivity(G, [(0, 1), (2, 3)])
    >>> for u, v, p in preds:
    ...     print(f"({u}, {v}) -> {p:.8f}")
    (0, 1) -> 1.45605311
    (2, 3) -> 1.54955745

    References
    ----------
    .. [1] Martinčić-Ipšić S., Močibob E., Perc M. Link prediction on Twitter. PLOS ONE, 12:7, 2017
        https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0181079&type=printable
    """
    def predict(u, v):  
        return sum(G.degree(z,weight='weight') / G.degree(z) for z in nx.common_neighbors(G, u, v))

    return _apply_prediction(G, predict, ebunch)

@not_implemented_for("directed")
@not_implemented_for("multigraph")
def inverse_selectivity(G, ebunch=None):
    r"""

    Parameters
    ----------
    G : graph
        NetworkX undirected weighted graph.

    ebunch : iterable of node pairs, optional (default = None)
        Inverse Selectivity will be computed for each pair of nodes given
        in the iterable. The pairs must be given as 2-tuples (u, v)
        where u and v are nodes in the graph. If ebunch is None then all
        non-existent edges in the graph will be used.
        Default value: None.

    Returns
    -------
    piter : iterator
        An iterator of 3-tuples in the form (u, v, p) where (u, v) is a
        pair of nodes and p is their Inverse Selectivity.

    Examples
    --------
    >>> G = nx.Graph()
    >>> G.add_weighted_edges_from((u,v,random.random()) for u,v in nx.complete_graph(5).edges())
    >>> preds = inverse_selectivity(G, [(0, 1), (2, 3)])
    >>> for u, v, p in preds:
    ...     print(f"({u}, {v}) -> {p:.8f}")
    (0, 1) -> 7.56368937
    (2, 3) -> 5.91879487

    References
    ----------
    .. [1] Martinčić-Ipšić S., Močibob E., Perc M. Link prediction on Twitter. PLOS ONE, 12:7, 2017
        https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0181079&type=printable
    """
    def predict(u, v):  
        return sum(G.degree(z) / G.degree(z,weight='weight') for z in nx.common_neighbors(G, u, v))

    return _apply_prediction(G, predict, ebunch)

@not_implemented_for("directed")
@not_implemented_for("multigraph")
def hub_promoted_index(G, ebunch=None):
    r"""

    Parameters
    ----------
    G : graph
        NetworkX undirected graph.

    ebunch : iterable of node pairs, optional (default = None)
        Hub Promoted Index will be computed for each pair of nodes given
        in the iterable. The pairs must be given as 2-tuples (u, v)
        where u and v are nodes in the graph. If ebunch is None then all
        non-existent edges in the graph will be used.
        Default value: None.

    Returns
    -------
    piter : iterator
        An iterator of 3-tuples in the form (u, v, p) where (u, v) is a
        pair of nodes and p is their Hub Promoted Index.

    Examples
    --------
    >>> G = nx.complete_graph(5)
    >>> preds = hub_promoted_index(G, [(0, 1), (2, 3)])
    >>> for u, v, p in preds:
    ...     print(f"({u}, {v}) -> {p:.8f}")
    (0, 1) -> 0.75000000
    (2, 3) -> 0.75000000

    References
    ----------
    .. [1] Martínez V., Berzal F., Cubero J.-C. A Survey of Link Prediction in Complex Networks. ACM Comput.Surv., 49:1–33, 2016
        https://www.researchgate.net/publication/310912568_A_Survey_of_Link_Prediction_in_Complex_Networks
    """
    def predict(u, v):
        try:  
            return len(list(nx.common_neighbors(G, u, v))) / min(G.degree(u),G.degree(v))
        except ZeroDivisionError:
            return 0

    return _apply_prediction(G, predict, ebunch)

@not_implemented_for("directed")
@not_implemented_for("multigraph")
def hub_depressed_index(G, ebunch=None):
    r"""

    Parameters
    ----------
    G : graph
        NetworkX undirected graph.

    ebunch : iterable of node pairs, optional (default = None)
        Hub Depressed Index will be computed for each pair of nodes given
        in the iterable. The pairs must be given as 2-tuples (u, v)
        where u and v are nodes in the graph. If ebunch is None then all
        non-existent edges in the graph will be used.
        Default value: None.

    Returns
    -------
    piter : iterator
        An iterator of 3-tuples in the form (u, v, p) where (u, v) is a
        pair of nodes and p is their Hub Depressed Index.

    Examples
    --------
    >>> G = nx.complete_graph(5)
    >>> preds = hub_depressed_index(G, [(0, 1), (2, 3)])
    >>> for u, v, p in preds:
    ...     print(f"({u}, {v}) -> {p:.8f}")
    (0, 1) -> 0.75000000
    (2, 3) -> 0.75000000

    References
    ----------
    .. [1] Martínez V., Berzal F., Cubero J.-C. A Survey of Link Prediction in Complex Networks. ACM Comput.Surv., 49:1–33, 2016
        https://www.researchgate.net/publication/310912568_A_Survey_of_Link_Prediction_in_Complex_Networks
    """
    def predict(u, v):
        try:  
            return len(list(nx.common_neighbors(G, u, v))) / max(G.degree(u),G.degree(v))
        except ZeroDivisionError:
            return 0
        
    return _apply_prediction(G, predict, ebunch)

@not_implemented_for("directed")
@not_implemented_for("multigraph")
def salton_index(G, ebunch=None):
    r"""

    Parameters
    ----------
    G : graph
        NetworkX undirected graph.

    ebunch : iterable of node pairs, optional (default = None)
        Salton Index will be computed for each pair of nodes given
        in the iterable. The pairs must be given as 2-tuples (u, v)
        where u and v are nodes in the graph. If ebunch is None then all
        non-existent edges in the graph will be used.
        Default value: None.

    Returns
    -------
    piter : iterator
        An iterator of 3-tuples in the form (u, v, p) where (u, v) is a
        pair of nodes and p is their Salton Index.

    Examples
    --------
    >>> G = nx.complete_graph(5)
    >>> preds = salton_index(G, [(0, 1), (2, 3)])
    >>> for u, v, p in preds:
    ...     print(f"({u}, {v}) -> {p:.8f}")
    (0, 1) -> 0.75000000
    (2, 3) -> 0.75000000

    References
    ----------
    .. [1] Martínez V., Berzal F., Cubero J.-C. A Survey of Link Prediction in Complex Networks. ACM Comput.Surv., 49:1–33, 2016
        https://www.researchgate.net/publication/310912568_A_Survey_of_Link_Prediction_in_Complex_Networks
    """
    def predict(u, v):
        try:  
            return len(list(nx.common_neighbors(G, u, v))) / sqrt(G.degree(u) * G.degree(v))
        except ZeroDivisionError:
            return 0

    return _apply_prediction(G, predict, ebunch)

@not_implemented_for("directed")
@not_implemented_for("multigraph")
def sorensen_index(G, ebunch=None):
    r"""

    Parameters
    ----------
    G : graph
        NetworkX undirected graph.

    ebunch : iterable of node pairs, optional (default = None)
        Sørensen Index will be computed for each pair of nodes given
        in the iterable. The pairs must be given as 2-tuples (u, v)
        where u and v are nodes in the graph. If ebunch is None then all
        non-existent edges in the graph will be used.
        Default value: None.

    Returns
    -------
    piter : iterator
        An iterator of 3-tuples in the form (u, v, p) where (u, v) is a
        pair of nodes and p is their Sørensen Index.

    Examples
    --------
    >>> G = nx.complete_graph(5)
    >>> preds = sorensen_index(G, [(0, 1), (2, 3)])
    >>> for u, v, p in preds:
    ...     print(f"({u}, {v}) -> {p:.8f}")
    (0, 1) -> 0.75000000
    (2, 3) -> 0.75000000

    References
    ----------
    .. [1] Martínez V., Berzal F., Cubero J.-C. A Survey of Link Prediction in Complex Networks. ACM Comput.Surv., 49:1–33, 2016
        https://www.researchgate.net/publication/310912568_A_Survey_of_Link_Prediction_in_Complex_Networks
    """
    def predict(u, v):
        try:  
            return 2*len(list(nx.common_neighbors(G, u, v))) / (G.degree(u) + G.degree(v))
        except ZeroDivisionError:
            return 0

    return _apply_prediction(G, predict, ebunch)

@not_implemented_for("directed")
@not_implemented_for("multigraph")
def leicht_holme_newman_index(G, ebunch=None):
    r"""

    Parameters
    ----------
    G : graph
        NetworkX undirected graph.

    ebunch : iterable of node pairs, optional (default = None)
        Leicht-Holme-Newman Index will be computed for each pair of nodes given
        in the iterable. The pairs must be given as 2-tuples (u, v)
        where u and v are nodes in the graph. If ebunch is None then all
        non-existent edges in the graph will be used.
        Default value: None.

    Returns
    -------
    piter : iterator
        An iterator of 3-tuples in the form (u, v, p) where (u, v) is a
        pair of nodes and p is their Leicht-Holme-Newman Index.

    Examples
    --------
    >>> G = nx.complete_graph(5)
    >>> preds = leicht_holme_newman_index(G, [(0, 1), (2, 3)])
    >>> for u, v, p in preds:
    ...     print(f"({u}, {v}) -> {p:.8f}")
    (0, 1) -> 0.18750000
    (2, 3) -> 0.18750000

    References
    ----------
    .. [1] Martínez V., Berzal F., Cubero J.-C. A Survey of Link Prediction in Complex Networks. ACM Comput.Surv., 49:1–33, 2016
        https://www.researchgate.net/publication/310912568_A_Survey_of_Link_Prediction_in_Complex_Networks
    """    
    def predict(u, v):
        try:  
            return len(list(nx.common_neighbors(G, u, v))) / (G.degree(u) * G.degree(v))
        except ZeroDivisionError:
            return 0

    return _apply_prediction(G, predict, ebunch)

@not_implemented_for("directed")
@not_implemented_for("multigraph")
def QQ(G, ebunch=None):
    r"""

    Parameters
    ----------
    G : graph
        NetworkX undirected graph.

    ebunch : iterable of node pairs, optional (default = None)
        Quasi Neighbourhood Overlap will be computed for each pair of nodes given
        in the iterable. The pairs must be given as 2-tuples (u, v)
        where u and v are nodes in the graph. If ebunch is None then all
        non-existent edges in the graph will be used.
        Default value: None.

    Returns
    -------
    piter : iterator
        An iterator of 3-tuples in the form (u, v, p) where (u, v) is a
        pair of nodes and p is their Quasi Neighbourhood Overlap.

    Examples
    --------
    >>> G = nx.complete_graph(5)
    >>> preds = QQ(G, [(0, 1), (2, 3)])
    >>> for u, v, p in preds:
    ...     print(f"({u}, {v}) -> {p:.8f}")
    (0, 1) -> 1.50000000
    (2, 3) -> 1.50000000

    References
    ----------
    .. [1] Orzechowski K.P., Mrowiński M.J., Fronczak A., Fronczak P. Asymmetry of social interactions and its role in link predictability: 
    the case of coauthorship networks. J Informetr., 17:101405, 2023.
        https://www.sciencedirect.com/science/article/pii/S1751157723000305?via%3Dihub
    """    
    def predict(u, v):
        cn = len(list(nx.common_neighbors(G, u, v)))
        try:
            return  cn / G.degree(u) + cn / G.degree(v)
        except ZeroDivisionError:      
            return 0

    return _apply_prediction(G, predict, ebunch)

@not_implemented_for("directed")
@not_implemented_for("multigraph")
def QA(G, ebunch=None):
    r"""

    Parameters
    ----------
    G : graph
        NetworkX undirected graph.

    ebunch : iterable of node pairs, optional (default = None)
        Quasi Adamic Index will be computed for each pair of nodes given
        in the iterable. The pairs must be given as 2-tuples (u, v)
        where u and v are nodes in the graph. If ebunch is None then all
        non-existent edges in the graph will be used.
        Default value: None.

    Returns
    -------
    piter : iterator
        An iterator of 3-tuples in the form (u, v, p) where (u, v) is a
        pair of nodes and p is their Quasi Adamic Index.

    Examples
    --------
    >>> G = nx.complete_graph(5)
    >>> preds = QA(G, [(0, 1), (2, 3)])
    >>> for u, v, p in preds:
    ...     print(f"({u}, {v}) -> {p:.8f}")
    (0, 1) -> 4.32808512
    (2, 3) -> 4.32808512

    References
    ----------
    .. [1] Orzechowski K.P., Mrowiński M.J., Fronczak A., Fronczak P. Asymmetry of social interactions and its role in link predictability: 
    the case of coauthorship networks. J Informetr., 17:101405, 2023.
        https://www.sciencedirect.com/science/article/pii/S1751157723000305?via%3Dihub
    """    
    def predict(u, v):
        cn = len(list(nx.common_neighbors(G, u, v)))
        try: 
            return  cn / log(G.degree(u)) + cn / log(G.degree(v))
        except ZeroDivisionError:
            return 0

    return _apply_prediction(G, predict, ebunch)

@not_implemented_for("directed")
@not_implemented_for("multigraph")
def AT1(G, ebunch=None):
    r"""

    Parameters
    ----------
    G : graph
        NetworkX undirected graph.

    ebunch : iterable of node pairs, optional (default = None)
        Asymmetric Triad Index 1 will be computed for each pair of nodes given
        in the iterable. The pairs must be given as 2-tuples (u, v)
        where u and v are nodes in the graph. If ebunch is None then all
        non-existent edges in the graph will be used.
        Default value: None.

    Returns
    -------
    piter : iterator
        An iterator of 3-tuples in the form (u, v, p) where (u, v) is a
        pair of nodes and p is their Asymmetric Triad Index 1.

    Examples
    --------
    >>> G = nx.complete_graph(5)
    >>> preds = AT1(G, [(0, 1), (2, 3)])
    >>> for u, v, p in preds:
    ...     print(f"({u}, {v}) -> {p:.8f}")
    (0, 1) -> 6.00000000
    (2, 3) -> 6.00000000

    References
    ----------
    .. [1] Orzechowski K.P., Mrowiński M.J., Fronczak A., Fronczak P. Asymmetry of social interactions and its role in link predictability: 
    the case of coauthorship networks. J Informetr., 17:101405, 2023.
        https://www.sciencedirect.com/science/article/pii/S1751157723000305?via%3Dihub
    """  
    def predict(u, v):
        cn = list(nx.common_neighbors(G, u, v))
        try:  
            return sum([(len(list(nx.common_neighbors(G, u, z))) + len(list(nx.common_neighbors(G, v, z)))) / (G.degree(z) - 1) for z in cn]) 
        except ZeroDivisionError:
            return 0
        
    return _apply_prediction(G, predict, ebunch)

@not_implemented_for("directed")
@not_implemented_for("multigraph")
def wAT1(G, ebunch=None):
    r"""

    Parameters
    ----------
    G : graph
        NetworkX undirected weighted graph.

    ebunch : iterable of node pairs, optional (default = None)
        Weighted Asymmetric Triad Index 1 will be computed for each pair of nodes given
        in the iterable. The pairs must be given as 2-tuples (u, v)
        where u and v are nodes in the graph. If ebunch is None then all
        non-existent edges in the graph will be used.
        Default value: None.

    Returns
    -------
    piter : iterator
        An iterator of 3-tuples in the form (u, v, p) where (u, v) is a
        pair of nodes and p is their weighted Asymmetric Triad Index 1.

    Examples
    --------
    >>> G = nx.Graph()
    >>> G.add_weighted_edges_from((u,v,random.random()) for u,v in nx.complete_graph(5).edges())
    >>> preds = wAT1(G, [(0, 1), (2, 3)])
    >>> for u, v, p in preds:
    ...     print(f"({u}, {v}) -> {p:.8f}")
    (0, 1) -> 1.37895263
    (2, 3) -> 1.83013705

    References
    ----------
    .. [1] Orzechowski K.P., Mrowiński M.J., Fronczak A., Fronczak P. Asymmetry of social interactions and its role in link predictability: 
    the case of coauthorship networks. J Informetr., 17:101405, 2023.
        https://www.sciencedirect.com/science/article/pii/S1751157723000305?via%3Dihub
    """  
    def predict(u, v):
        cn = list(nx.common_neighbors(G, u, v))
        try:  
            return sum([(G.get_edge_data(u,z)["weight"] + G.get_edge_data(v,z)["weight"]) / G.degree(z, weight="weight") for z in cn]) 
        except ZeroDivisionError:
            return 0
        
    return _apply_prediction(G, predict, ebunch)

@not_implemented_for("directed")
@not_implemented_for("multigraph")
def AT2(G, ebunch=None):
    r"""

    Parameters
    ----------
    G : graph
        NetworkX undirected graph.

    ebunch : iterable of node pairs, optional (default = None)
        Asymmetric Triad Index 2 will be computed for each pair of nodes given
        in the iterable. The pairs must be given as 2-tuples (u, v)
        where u and v are nodes in the graph. If ebunch is None then all
        non-existent edges in the graph will be used.
        Default value: None.

    Returns
    -------
    piter : iterator
        An iterator of 3-tuples in the form (u, v, p) where (u, v) is a
        pair of nodes and p is their Asymmetric Triad Index 2.

    Examples
    --------
    >>> G = nx.complete_graph(5)
    >>> preds = AT2(G, [(0, 1), (2, 3)])
    >>> for u, v, p in preds:
    ...     print(f"({u}, {v}) -> {p:.8f}")
    (0, 1) -> 6.00000000
    (2, 3) -> 6.00000000

    References
    ----------
    .. [1] Orzechowski K.P., Mrowiński M.J., Fronczak A., Fronczak P. Asymmetry of social interactions and its role in link predictability: 
    the case of coauthorship networks. J Informetr., 17:101405, 2023.
        https://www.sciencedirect.com/science/article/pii/S1751157723000305?via%3Dihub
    """  
    def predict(u, v):
        cn = list(nx.common_neighbors(G, u, v))
        try:  
            return sum([len(list(nx.common_neighbors(G, u, z))) / (G.degree(u) - 1) + len(list(nx.common_neighbors(G, v, z))) / (G.degree(v) - 1) for z in cn]) 
        except ZeroDivisionError:
            return 0
        
    return _apply_prediction(G, predict, ebunch)

@not_implemented_for("directed")
@not_implemented_for("multigraph")
def wAT2(G, ebunch=None):
    r"""

    Parameters
    ----------
    G : graph
        NetworkX undirected weighted graph.

    ebunch : iterable of node pairs, optional (default = None)
        Weighted Asymmetric Triad Index 2 will be computed for each pair of nodes given
        in the iterable. The pairs must be given as 2-tuples (u, v)
        where u and v are nodes in the graph. If ebunch is None then all
        non-existent edges in the graph will be used.
        Default value: None.

    Returns
    -------
    piter : iterator
        An iterator of 3-tuples in the form (u, v, p) where (u, v) is a
        pair of nodes and p is their weighted Asymmetric Triad Index 2.

    Examples
    --------
    >>> G = nx.Graph()
    >>> G.add_weighted_edges_from((u,v,random.random()) for u,v in nx.complete_graph(5).edges())
    >>> preds = wAT2(G, [(0, 1), (2, 3)])
    >>> for u, v, p in preds:
    ...     print(f"({u}, {v}) -> {p:.8f}")
    (0, 1) -> 1.42081771
    (2, 3) -> 1.38866840

    References
    ----------
    .. [1] Orzechowski K.P., Mrowiński M.J., Fronczak A., Fronczak P. Asymmetry of social interactions and its role in link predictability: 
    the case of coauthorship networks. J Informetr., 17:101405, 2023.
        https://www.sciencedirect.com/science/article/pii/S1751157723000305?via%3Dihub
    """ 
    def predict(u, v):
        cn = list(nx.common_neighbors(G, u, v))
        try:  
            return sum([G.get_edge_data(u,z)["weight"] / G.degree(u, weight="weight") + G.get_edge_data(v,z)["weight"] / G.degree(v, weight="weight") for z in cn]) 
        except ZeroDivisionError:
            return 0
        
    return _apply_prediction(G, predict, ebunch)

@not_implemented_for("directed")
@not_implemented_for("multigraph")
def AT3(G, ebunch=None):
    r"""

    Parameters
    ----------
    G : graph
        NetworkX undirected graph.

    ebunch : iterable of node pairs, optional (default = None)
        Asymmetric Triad Index 3 will be computed for each pair of nodes given
        in the iterable. The pairs must be given as 2-tuples (u, v)
        where u and v are nodes in the graph. If ebunch is None then all
        non-existent edges in the graph will be used.
        Default value: None.

    Returns
    -------
    piter : iterator
        An iterator of 3-tuples in the form (u, v, p) where (u, v) is a
        pair of nodes and p is their Asymmetric Triad Index 3.

    Examples
    --------
    >>> G = nx.complete_graph(5)
    >>> preds = AT3(G, [(0, 1), (2, 3)])
    >>> for u, v, p in preds:
    ...     print(f"({u}, {v}) -> {p:.8f}")
    (0, 1) -> 6.00000000
    (2, 3) -> 6.00000000

    References
    ----------
    .. [1] Orzechowski K.P., Mrowiński M.J., Fronczak A., Fronczak P. Asymmetry of social interactions and its role in link predictability: 
    the case of coauthorship networks. J Informetr., 17:101405, 2023.
        https://www.sciencedirect.com/science/article/pii/S1751157723000305?via%3Dihub
    """ 
    def predict(u, v):
        cn = list(nx.common_neighbors(G, u, v))
        try:  
            return sum([len(list(nx.common_neighbors(G, u, z))) / (G.degree(u) - 1) + len(list(nx.common_neighbors(G, v, z))) / (G.degree(z) - 1) for z in cn]) 
        except ZeroDivisionError:
            return 0
        
    return _apply_prediction(G, predict, ebunch)

@not_implemented_for("directed")
@not_implemented_for("multigraph")
def wAT3(G, ebunch=None):
    r"""

    Parameters
    ----------
    G : graph
        NetworkX undirected weighted graph.

    ebunch : iterable of node pairs, optional (default = None)
        Weighted Asymmetric Triad Index 3 will be computed for each pair of nodes given
        in the iterable. The pairs must be given as 2-tuples (u, v)
        where u and v are nodes in the graph. If ebunch is None then all
        non-existent edges in the graph will be used.
        Default value: None.

    Returns
    -------
    piter : iterator
        An iterator of 3-tuples in the form (u, v, p) where (u, v) is a
        pair of nodes and p is their weighted Asymmetric Triad Index 3.

    Examples
    --------
    >>> G = nx.Graph()
    >>> G.add_weighted_edges_from((u,v,random.random()) for u,v in nx.complete_graph(5).edges())
    >>> preds = wAT3(G, [(0, 1), (2, 3)])
    >>> for u, v, p in preds:
    ...     print(f"({u}, {v}) -> {p:.8f}")
    (0, 1) -> 1.03748070
    (2, 3) -> 1.57785248

    References
    ----------
    .. [1] Orzechowski K.P., Mrowiński M.J., Fronczak A., Fronczak P. Asymmetry of social interactions and its role in link predictability: 
    the case of coauthorship networks. J Informetr., 17:101405, 2023.
        https://www.sciencedirect.com/science/article/pii/S1751157723000305?via%3Dihub
    """ 
    def predict(u, v):
        cn = list(nx.common_neighbors(G, u, v))
        try:  
            return sum([G.get_edge_data(u,z)["weight"] / G.degree(u, weight="weight") + G.get_edge_data(v,z)["weight"] / G.degree(z, weight="weight") for z in cn]) 
        except ZeroDivisionError:
            return 0
        
    return _apply_prediction(G, predict, ebunch)

@not_implemented_for("directed")
@not_implemented_for("multigraph")
def mix_index_1(G, ebunch=None):
    r"""

    Parameters
    ----------
    G : graph
        NetworkX undirected weighted graph.

    ebunch : iterable of node pairs, optional (default = None)
        Mixed Index 1 will be computed for each pair of nodes given
        in the iterable. The pairs must be given as 2-tuples (u, v)
        where u and v are nodes in the graph. If ebunch is None then all
        non-existent edges in the graph will be used.
        Default value: None.

    Returns
    -------
    piter : iterator
        An iterator of 3-tuples in the form (u, v, p) where (u, v) is a
        pair of nodes and p is their mixed Index 1.

    Examples
    --------
    >>> G = nx.Graph()
    >>> G.add_weighted_edges_from((u,v,random.random()) for u,v in nx.complete_graph(5).edges())
    >>> preds = mix_index_1(G, [(0, 1), (2, 3)])
    >>> for u, v, p in preds:
    ...     print(f"({u}, {v}) -> {p:.8f}")
    (0, 1) -> 3.16955396
    (2, 3) -> 3.32474090

    References
    ----------
    .. [1] Orzechowski K.P., Mrowiński M.J., Fronczak A., Fronczak P. Asymmetry of social interactions and its role in link predictability: 
    the case of coauthorship networks. J Informetr., 17:101405, 2023.
        https://www.sciencedirect.com/science/article/pii/S1751157723000305?via%3Dihub
    """ 
    def predict(u, v): 
        return list(*wAT1(G, [(u, v)]))[2] + list(*QQ(G, [(u, v)]))[2]

    return _apply_prediction(G, predict, ebunch)

@not_implemented_for("directed")
@not_implemented_for("multigraph")
def mix_index_2(G, ebunch=None):
    r"""

    Parameters
    ----------
    G : graph
        NetworkX undirected weighted graph.

    ebunch : iterable of node pairs, optional (default = None)
        Mixed Index 2 will be computed for each pair of nodes given
        in the iterable. The pairs must be given as 2-tuples (u, v)
        where u and v are nodes in the graph. If ebunch is None then all
        non-existent edges in the graph will be used.
        Default value: None.

    Returns
    -------
    piter : iterator
        An iterator of 3-tuples in the form (u, v, p) where (u, v) is a
        pair of nodes and p is their mixed Index 2.

    Examples
    --------
    >>> G = nx.Graph()
    >>> G.add_weighted_edges_from((u,v,random.random()) for u,v in nx.complete_graph(5).edges())
    >>> preds = mix_index_2(G, [(0, 1), (2, 3)])
    >>> for u, v, p in preds:
    ...     print(f"({u}, {v}) -> {p:.8f}")
    (0, 1) -> 5.98939093
    (2, 3) -> 6.01348707

    References
    ----------
    .. [1] Orzechowski K.P., Mrowiński M.J., Fronczak A., Fronczak P. Asymmetry of social interactions and its role in link predictability: 
    the case of coauthorship networks. J Informetr., 17:101405, 2023.
        https://www.sciencedirect.com/science/article/pii/S1751157723000305?via%3Dihub
    """ 
    def predict(u, v): 
        return list(*wAT1(G, [(u, v)]))[2] + list(*QA(G, [(u, v)]))[2]

    return _apply_prediction(G, predict, ebunch)

# ====================================================================================================
# NEW DIRECTED SIMILARITY MEASURES
# ====================================================================================================

@not_implemented_for("multigraph")
def DCN(G, ebunch=None):
    r"""

    Parameters
    ----------
    G : graph
        NetworkX directed graph.

    ebunch : iterable of node pairs, optional (default = None)
        Directed Common Neighbours Index will be computed for each pair of nodes given
        in the iterable. The pairs must be given as 2-tuples (u, v)
        where u and v are nodes in the graph. If ebunch is None then all
        non-existent edges in the graph will be used.
        Default value: None.

    Returns
    -------
    piter : iterator
        An iterator of 3-tuples in the form (u, v, p) where (u, v) is a
        pair of nodes and p is their directed Common Neighbours Index.

    Examples
    --------
    >>> G = nx.DiGraph()
    >>> G.add_edges_from([(0,1),(1,3),(0,4),(0,5),(0,6),(0,9),(10,0),(4,1),(11,1),(7,1),(7,0),(4,7),(9,1)])
    >>> preds = DCN(G, [(0, 1), (0, 7)])
    >>> for u, v, p in preds:
    ...        print(f"({u}, {v}) -> {p:.8f}")
    (0, 1) -> 2.00000000
    (0, 7) -> 1.00000000

    References
    ----------
    .. [1] Li J. et al. Link Prediction in Directed Networks Utilizing the Role of Reciprocal Links., IEEE Access 8:28668–28680, 2020.
        https://ieeexplore.ieee.org/document/8985354
    """ 
    def predict(u, v): 
        return len(set(G.successors(u)).intersection(set(G.predecessors(v)))) 

    return _apply_prediction(G, predict, ebunch)

@not_implemented_for("multigraph")
def DAA(G, ebunch=None):
    r"""

    Parameters
    ----------
    G : graph
        NetworkX directed graph.

    ebunch : iterable of node pairs, optional (default = None)
        Directed Adamic Adar Index will be computed for each pair of nodes given
        in the iterable. The pairs must be given as 2-tuples (u, v)
        where u and v are nodes in the graph. If ebunch is None then all
        non-existent edges in the graph will be used.
        Default value: None.

    Returns
    -------
    piter : iterator
        An iterator of 3-tuples in the form (u, v, p) where (u, v) is a
        pair of nodes and p is their directed Adamic Adar Index.

    Examples
    --------
    >>> G = nx.DiGraph()
    >>> G.add_edges_from([(0,1),(1,3),(0,4),(0,5),(0,6),(0,9),(10,0),(4,1),(11,1),(7,1),(7,0),(4,7),(9,1)])
    >>> preds = DAA(G, [(0, 1), (0, 7)])
    >>> for u, v, p in preds:
    ...        print(f"({u}, {v}) -> {p:.8f}")
    (0, 1) -> 1.44269504
    (0, 7) -> 1.44269504

    References
    ----------
    .. [1] Li J. et al. Link Prediction in Directed Networks Utilizing the Role of Reciprocal Links., IEEE Access 8:28668–28680, 2020.
        https://ieeexplore.ieee.org/document/8985354
    """ 
    def predict(u, v):
        cn = set(G.successors(u)).intersection(set(G.predecessors(v)))
        return sum(1 / log(G.out_degree(z)) if log(G.out_degree(z)) != 0 else 0 for z in cn) 

    return _apply_prediction(G, predict, ebunch)

@not_implemented_for("multigraph")
def DRA(G, ebunch=None):
    r"""

    Parameters
    ----------
    G : graph
        NetworkX directed graph.

    ebunch : iterable of node pairs, optional (default = None)
        Directed Resource Allocation Index will be computed for each pair of nodes given
        in the iterable. The pairs must be given as 2-tuples (u, v)
        where u and v are nodes in the graph. If ebunch is None then all
        non-existent edges in the graph will be used.
        Default value: None.

    Returns
    -------
    piter : iterator
        An iterator of 3-tuples in the form (u, v, p) where (u, v) is a
        pair of nodes and p is their directed Resource Allocation Index.

    Examples
    --------
    >>> G = nx.DiGraph()
    >>> G.add_edges_from([(0,1),(1,3),(0,4),(0,5),(0,6),(0,9),(10,0),(4,1),(11,1),(7,1),(7,0),(4,7),(9,1)])
    >>> preds = DRA(G, [(0, 1), (0, 7)])
    >>> for u, v, p in preds:
    ...        print(f"({u}, {v}) -> {p:.8f}")
    (0, 1) -> 1.50000000
    (0, 7) -> 0.50000000

    References
    ----------
    .. [1] Li J. et al. Link Prediction in Directed Networks Utilizing the Role of Reciprocal Links., IEEE Access 8:28668–28680, 2020.
        https://ieeexplore.ieee.org/document/8985354
    """ 
    def predict(u, v):
        cn = set(G.successors(u)).intersection(set(G.predecessors(v)))
        return sum(1 / G.out_degree(z) if G.out_degree(z) != 0 else 0 for z in cn) 

    return _apply_prediction(G, predict, ebunch)

@not_implemented_for("multigraph")
def directed_Jaccard_coefficient(G, ebunch=None):
    r"""

    Parameters
    ----------
    G : graph
        NetworkX directed graph.

    ebunch : iterable of node pairs, optional (default = None)
        Directed Jaccard Coefficient will be computed for each pair of nodes given
        in the iterable. The pairs must be given as 2-tuples (u, v)
        where u and v are nodes in the graph. If ebunch is None then all
        non-existent edges in the graph will be used.
        Default value: None.

    Returns
    -------
    piter : iterator
        An iterator of 3-tuples in the form (u, v, p) where (u, v) is a
        pair of nodes and p is their directed Jaccard Coefficient.

    Examples
    --------
    >>> G = nx.DiGraph()
    >>> G.add_edges_from([(0,1),(1,3),(0,4),(0,5),(0,6),(0,9),(10,0),(4,1),(11,1),(7,1),(7,0),(4,7),(9,1)])
    >>> preds = directed_Jaccard_coefficient(G, [(0, 1), (0, 7)])
    >>> for u, v, p in preds:
    ...        print(f"({u}, {v}) -> {p:.8f}")
    (0, 1) -> 0.25000000
    (0, 7) -> 0.12500000

    References
    ----------
    .. [1] Li J. et al. Link Prediction in Directed Networks Utilizing the Role of Reciprocal Links., IEEE Access 8:28668–28680, 2020.
        https://ieeexplore.ieee.org/document/8985354
    """
    
    def predict(u, v):
        union_size = len(set(G.successors(u)).union(set(G.predecessors(v))))
        try: 
            return len(set(G.successors(u)).intersection(set(G.predecessors(v)))) / union_size 
        except ZeroDivisionError:
            return 0
        
    return _apply_prediction(G, predict, ebunch)

@not_implemented_for("multigraph")
def directed_salton_index(G, ebunch=None):
    r"""

    Parameters
    ----------
    G : graph
        NetworkX directed graph.

    ebunch : iterable of node pairs, optional (default = None)
        Directed Salton Index will be computed for each pair of nodes given
        in the iterable. The pairs must be given as 2-tuples (u, v)
        where u and v are nodes in the graph. If ebunch is None then all
        non-existent edges in the graph will be used.
        Default value: None.

    Returns
    -------
    piter : iterator
        An iterator of 3-tuples in the form (u, v, p) where (u, v) is a
        pair of nodes and p is their directed Salton Index.

    Examples
    --------
    >>> G = nx.DiGraph()
    >>> G.add_edges_from([(0,1),(1,3),(0,4),(0,5),(0,6),(0,9),(10,0),(4,1),(11,1),(7,1),(7,0),(4,7),(9,1)])
    >>> preds = directed_salton_index(G, [(0, 1), (0, 7)])
    >>> for u, v, p in preds:
    ...        print(f"({u}, {v}) -> {p:.8f}")
    (0, 1) -> 0.40000000
    (0, 7) -> 0.44721360

    References
    ----------
    .. [1] Benhidour H., Almeshkhas L., Kerrache S., An Approach for Link Prediction in Directed Complex Networks based on Asymmetric Similarity-Popularity, 
    Preprint ArXiv, 2022.
        https://arxiv.org/abs/2207.07399
    """
    def predict(u, v):
        try: 
            return len(set(G.successors(u)).intersection(set(G.predecessors(v)))) / (sqrt(G.out_degree(u) * G.in_degree(v))) 
        except ZeroDivisionError:
            return 0
        
    return _apply_prediction(G, predict, ebunch)

@not_implemented_for("multigraph")
def directed_sørensen_index(G, ebunch=None):
    r"""

    Parameters
    ----------
    G : graph
        NetworkX directed graph.

    ebunch : iterable of node pairs, optional (default = None)
        Directed Sørensen Index will be computed for each pair of nodes given
        in the iterable. The pairs must be given as 2-tuples (u, v)
        where u and v are nodes in the graph. If ebunch is None then all
        non-existent edges in the graph will be used.
        Default value: None.

    Returns
    -------
    piter : iterator
        An iterator of 3-tuples in the form (u, v, p) where (u, v) is a
        pair of nodes and p is their directed Sørensen Index.

    Examples
    --------
    >>> G = nx.DiGraph()
    >>> G.add_edges_from([(0,1),(1,3),(0,4),(0,5),(0,6),(0,9),(10,0),(4,1),(11,1),(7,1),(7,0),(4,7),(9,1)])
    >>> preds = directed_sørensen_index(G, [(0, 1), (0, 7)])
    >>> for u, v, p in preds:
    ...        print(f"({u}, {v}) -> {p:.8f}")
    (0, 1) -> 0.40000000
    (0, 7) -> 0.33333333

    References
    ----------
    .. [1] Benhidour H., Almeshkhas L., Kerrache S., An Approach for Link Prediction in Directed Complex Networks based on Asymmetric Similarity-Popularity, 
    Preprint ArXiv, 2022.
        https://arxiv.org/abs/2207.07399
    """
    def predict(u, v):
        try: 
            return 2*len(set(G.successors(u)).intersection(set(G.predecessors(v)))) / (G.out_degree(u) + G.in_degree(v)) 
        except ZeroDivisionError:
            return 0
        
    return _apply_prediction(G, predict, ebunch)

@not_implemented_for("multigraph")
def directed_LHN_index(G, ebunch=None):
    r"""

    Parameters
    ----------
    G : graph
        NetworkX directed graph.

    ebunch : iterable of node pairs, optional (default = None)
        Directed Leicht-Holme-Newman Index will be computed for each pair of nodes given
        in the iterable. The pairs must be given as 2-tuples (u, v)
        where u and v are nodes in the graph. If ebunch is None then all
        non-existent edges in the graph will be used.
        Default value: None.

    Returns
    -------
    piter : iterator
        An iterator of 3-tuples in the form (u, v, p) where (u, v) is a
        pair of nodes and p is their directed Leicht-Holme-Newman Index.

    Examples
    --------
    >>> G = nx.DiGraph()
    >>> G.add_edges_from([(0,1),(1,3),(0,4),(0,5),(0,6),(0,9),(10,0),(4,1),(11,1),(7,1),(7,0),(4,7),(9,1)])
    >>> preds = directed_LHN_index(G, [(0, 1), (0, 7)])
    >>> for u, v, p in preds:
    ...        print(f"({u}, {v}) -> {p:.8f}")
    (0, 1) -> 0.08000000
    (0, 7) -> 0.20000000

    References
    ----------
    .. [1] Benhidour H., Almeshkhas L., Kerrache S., An Approach for Link Prediction in Directed Complex Networks based on Asymmetric Similarity-Popularity, 
    Preprint ArXiv, 2022.
        https://arxiv.org/abs/2207.07399
    """
    def predict(u, v):
        try: 
            return len(set(G.successors(u)).intersection(set(G.predecessors(v)))) / (G.out_degree(u) * G.in_degree(v)) 
        except ZeroDivisionError:
            return 0
        
    return _apply_prediction(G, predict, ebunch)

@not_implemented_for("multigraph")
def directed_HPI_index(G, ebunch=None):
    r"""

    Parameters
    ----------
    G : graph
        NetworkX directed graph.

    ebunch : iterable of node pairs, optional (default = None)
        Directed Hub Promoted Index will be computed for each pair of nodes given
        in the iterable. The pairs must be given as 2-tuples (u, v)
        where u and v are nodes in the graph. If ebunch is None then all
        non-existent edges in the graph will be used.
        Default value: None.

    Returns
    -------
    piter : iterator
        An iterator of 3-tuples in the form (u, v, p) where (u, v) is a
        pair of nodes and p is their directed Hub Promoted Index.

    Examples
    --------
    >>> G = nx.DiGraph()
    >>> G.add_edges_from([(0,1),(1,3),(0,4),(0,5),(0,6),(0,9),(10,0),(4,1),(11,1),(7,1),(7,0),(4,7),(9,1)])
    >>> preds = directed_HPI_index(G, [(0, 1), (0, 7)])
    >>> for u, v, p in preds:
    ...        print(f"({u}, {v}) -> {p:.8f}")
    (0, 1) -> 0.40000000
    (0, 7) -> 1.00000000

    References
    ----------
    .. [1] Benhidour H., Almeshkhas L., Kerrache S., An Approach for Link Prediction in Directed Complex Networks based on Asymmetric Similarity-Popularity, 
    Preprint ArXiv, 2022.
        https://arxiv.org/abs/2207.07399
    """
    def predict(u, v):
        try: 
            return len(set(G.successors(u)).intersection(set(G.predecessors(v)))) / (min(G.out_degree(u),G.in_degree(v))) 
        except ZeroDivisionError:
            return 0
        
    return _apply_prediction(G, predict, ebunch)

@not_implemented_for("multigraph")
def directed_HDI_index(G, ebunch=None):
    r"""

    Parameters
    ----------
    G : graph
        NetworkX directed graph.

    ebunch : iterable of node pairs, optional (default = None)
        Directed Hub Depressed Index will be computed for each pair of nodes given
        in the iterable. The pairs must be given as 2-tuples (u, v)
        where u and v are nodes in the graph. If ebunch is None then all
        non-existent edges in the graph will be used.
        Default value: None.

    Returns
    -------
    piter : iterator
        An iterator of 3-tuples in the form (u, v, p) where (u, v) is a
        pair of nodes and p is their directed Hub Depressed Index.

    Examples
    --------
    >>> G = nx.DiGraph()
    >>> G.add_edges_from([(0,1),(1,3),(0,4),(0,5),(0,6),(0,9),(10,0),(4,1),(11,1),(7,1),(7,0),(4,7),(9,1)])
    >>> preds = directed_HDI_index(G, [(0, 1), (0, 7)])
    >>> for u, v, p in preds:
    ...        print(f"({u}, {v}) -> {p:.8f}")
    (0, 1) -> 0.40000000
    (0, 7) -> 0.20000000

    References
    ----------
    .. [1] Benhidour H., Almeshkhas L., Kerrache S., An Approach for Link Prediction in Directed Complex Networks based on Asymmetric Similarity-Popularity, 
    Preprint ArXiv, 2022.
        https://arxiv.org/abs/2207.07399
    """
    def predict(u, v): 
        try:
            return len(set(G.successors(u)).intersection(set(G.predecessors(v)))) / (max(G.out_degree(u),G.in_degree(v))) 
        except ZeroDivisionError:
            return 0
        
    return _apply_prediction(G, predict, ebunch)

@not_implemented_for("multigraph")
def directed_bifan_index(G, ebunch=None):
    r"""

    Parameters
    ----------
    G : graph
        NetworkX directed graph.

    ebunch : iterable of node pairs, optional (default = None)
        Directed Bifan Index will be computed for each pair of nodes given
        in the iterable. The pairs must be given as 2-tuples (u, v)
        where u and v are nodes in the graph. If ebunch is None then all
        non-existent edges in the graph will be used.
        Default value: None.

    Returns
    -------
    piter : iterator
        An iterator of 3-tuples in the form (u, v, p) where (u, v) is a
        pair of nodes and p is their directed Bifan Index.

    Examples
    --------
    >>> G = nx.DiGraph()
    >>> G.add_edges_from([(0,1),(1,3),(0,4),(0,5),(0,6),(0,9),(10,0),(4,1),(11,1),(7,1),(7,0),(4,7),(9,1)])
    >>> preds = directed_bifan_index(G, [(0, 1), (0, 7)])
    >>> for u, v, p in preds:
    ...        print(f"({u}, {v}) -> {p:.8f}")
    (0, 1) -> 5.00000000
    (0, 7) -> 1.00000000

    References
    ----------
    .. [1] Li J. et al. Link Prediction in Directed Networks Utilizing the Role of Reciprocal Links., IEEE Access 8:28668–28680, 2020.
        https://ieeexplore.ieee.org/document/8985354
    """    

    def predict(u, v):
        u_in_out = set([x for z in G.successors(u) for x in G.predecessors(z)]) 
        return len(u_in_out.intersection(set(G.predecessors(v))))

    return _apply_prediction(G, predict, ebunch)