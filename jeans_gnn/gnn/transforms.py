
import torch
import torch_geometric.transforms as T

class GraphPreprocessors(T.BaseTransform):
    pass

# class GraphTransforms(T.BaseTransform):
#     graph_dict = {
#         "KNN": T.KNNGraph,
#         "RADIUS": T.RadiusGraph,
#     }
#     def __init__(self,
#         graph_name, graph_hparams, feature_hparams={},
#         edge_weight=False, edge_weight_norm=False):

#         self.transforms = []
#         self.transforms.append(FeaturePreprocess(**feature_hparams))
#         if graph_name in self.graph_dict:
#             self.transforms.append(self.graph_dict[graph_name](**graph_hparams))
#         else:
#             raise KeyError(
#                 f"Unknown graph name \"{graph_name}\"."\
#                 f"Available models are: {str(self.graph_dict.keys())}")
#         if edge_weight:
#             self.transforms.append(EdgeWeightDistance(edge_weight_norm))
#         self.transforms = T.Compose(self.transforms)

#     def __call__(self, data):
#         return self.transforms(data)
