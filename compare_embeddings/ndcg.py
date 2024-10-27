import math


def dcg_at_k(relevance, k):
    """Calculate DCG@k for a single query"""
    dcg = 0
    for i in range(min(len(relevance), k)):
        dcg += relevance[i] / math.log2(i + 2)
    return dcg


def ndcg_at_k(relevance, k):
    """Calculate NDCG@k for a single query"""
    dcg = dcg_at_k(relevance, k)
    ideal_relevance = sorted(relevance, reverse=True)
    idcg = dcg_at_k(ideal_relevance, k)
    return dcg / idcg if idcg > 0 else 0


def calculate_ndcg_from_list(relevance_list):
    return ndcg_at_k(relevance_list, len(relevance_list))

