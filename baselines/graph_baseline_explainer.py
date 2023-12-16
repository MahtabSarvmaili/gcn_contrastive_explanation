import torch
from torch_geometric.data import Data
from torch_geometric.explain import Explainer, GNNExplainer, PGExplainer


def gnnexplainer(data, model, id):
    model.cpu()
    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=200),
        explanation_type='model',
        node_mask_type='attributes',
        edge_mask_type='object',
        model_config=dict(
            mode='multiclass_classification',
            task_level='graph',
            return_type='raw',  # Model returns log probabilities.
        ),
    )
    # Generate explanation for the node at index `10`:
    explanation = explainer(data.x.cpu().to(torch.float32), data.edge_index.cpu(), index=id, batch=None)
    return explanation.edge_mask


def pgexplainer(data, model, dt):
    from torch_geometric.data import Data
    model.cpu()

    explainer = Explainer(
        model=model,
        algorithm=PGExplainer(epochs=30, lr=0.003),
        explanation_type='phenomenon',
        edge_mask_type='object',
        model_config=dict(
            mode='multiclass_classification',
            task_level='graph',
            return_type='raw',
        ),
    )

    # PGExplainer needs to be trained separately since it is a parametric
    # explainer i.e it uses a neural network to generate explanations:
    for epoch in range(30):
        for dt_batch in data:
            dt_batch.cpu()

            explainer.algorithm.train(
                epoch, model, dt_batch.x.to(torch.float32), dt_batch.edge_index,
                target=dt_batch.y.reshape(-1).to(torch.int64), batch=dt_batch.batch
            )

    # Generate the explanation for a particular graph:
    explanation = explainer(
        dt.x.cpu().to(torch.float32), dt.edge_index.cpu(),
        target=dt.y.cpu().reshape(-1).to(torch.int64), batch=None
    )
    return explanation.edge_mask