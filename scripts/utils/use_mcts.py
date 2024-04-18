from collections import namedtuple
from random import choice, randrange
from utils.monte_carlo_tree_search import Node
from utils.models import cluster_and_simipc

_PhaseHyperparam = namedtuple("PhaseHyperparam", "features decision_count clustering fsize k terminal all_features, max_fsize, max_k")
MAX_FSIZE = 50
MAX_K = 8

# Inheriting from a namedtuple is convenient because it makes the class
# immutable and predefines __init__, __repr__, __hash__, __eq__, and others
class PhaseHyperparam(_PhaseHyperparam, Node):
        
    def find_children(ymap):
        if ymap.decision_count >= len(ymap.all_features):
            if ymap.clustering == None:
                return {
                    PhaseHyperparam(ymap.features, ymap.decision_count+1, 'gmm', None, None, False, ymap.all_features, ymap.max_fsize, ymap.max_k),
                    PhaseHyperparam(ymap.features, ymap.decision_count+1, 'kmeans', None, None, False, ymap.all_features, ymap.max_fsize, ymap.max_k),
                }
            elif ymap.fsize == None:
                return {
                    PhaseHyperparam(ymap.features, ymap.decision_count+1, ymap.clustering, f, None, False, ymap.all_features, ymap.max_fsize, ymap.max_k) for f in range(1, ymap.max_fsize, 2)
                }
            elif ymap.k == None:
                return {
                    PhaseHyperparam(ymap.features, ymap.decision_count+1, ymap.clustering, ymap.fsize, k, True, ymap.all_features, ymap.max_fsize, ymap.max_k) for k in range(2, ymap.max_k)
                }
            else:
                # Terminal state
                return set()
        else:
            add_feature = list(ymap.features)
            add_feature.append(ymap.all_features[ymap.decision_count])
            add_feature = tuple(add_feature)
            if ymap.all_features[ymap.decision_count] == 'IPC':
                return {PhaseHyperparam(add_feature, ymap.decision_count+1, None, None, None, False, ymap.all_features, ymap.max_fsize, ymap.max_k)}
            else:
                return {
                    PhaseHyperparam(ymap.features, ymap.decision_count+1, None, None, None, False, ymap.all_features, ymap.max_fsize, ymap.max_k),
                    PhaseHyperparam(add_feature, ymap.decision_count+1, None, None, None, False, ymap.all_features, ymap.max_fsize, ymap.max_k)
                }

    def find_random_child(ymap):
        rand_0_or_1 = choice(range(2))
        if ymap.decision_count < len(ymap.all_features):
            if rand_0_or_1 == 1 or \
            ymap.all_features[ymap.decision_count] == 'IPC' or \
            (len(ymap.features) == 1 and (len(ymap.all_features) - ymap.decision_count) == 1):
                add_feature = list(ymap.features)
                add_feature.append(ymap.all_features[ymap.decision_count])
                return PhaseHyperparam(tuple(add_feature), ymap.decision_count+1, None, None, None, False, ymap.all_features, ymap.max_fsize, ymap.max_k)
            else:
                return PhaseHyperparam(ymap.features, ymap.decision_count+1, None, None, None, False, ymap.all_features, ymap.max_fsize, ymap.max_k)
        else:
            if ymap.clustering == None:
                method = 'gmm' if rand_0_or_1 == 0 else 'kmeans'
                return PhaseHyperparam(ymap.features, ymap.decision_count+1, method, None, None, False, ymap.all_features, ymap.max_fsize, ymap.max_k)
            elif ymap.fsize == None:
                fsize = randrange(1, ymap.max_fsize, 2)
                return PhaseHyperparam(ymap.features, ymap.decision_count+1, ymap.clustering, fsize, None, False, ymap.all_features, ymap.max_fsize, ymap.max_k)
            elif ymap.k == None:
                k = randrange(2, ymap.max_k)
                return PhaseHyperparam(ymap.features, ymap.decision_count+1, ymap.clustering, ymap.fsize, k, True, ymap.all_features, ymap.max_fsize, ymap.max_k)
            else:
                # Terminal state
                return None


    def reward(ymap, args):
        if len(ymap.features) < 2:
            return 0
        # Assume input_data is already "processed" (ie, a df created with get_processed_counters)
        features = list(ymap.features)
        if 'IPC' not in features:
            features.append('IPC')
        input_data = args[1].stack(0)[features].unstack().swaplevel(axis=1).sort_index(axis=1)
        _, ipc = cluster_and_simipc(input_data, ymap.clustering, ymap.fsize, ymap.k, 'bopOnly')
        
        return ipc / args[0]

    def is_terminal(ymap):
        return ymap.terminal
