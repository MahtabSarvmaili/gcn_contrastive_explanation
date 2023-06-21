from torch_geometric.data import Data
from torch_geometric.explain import Explainer, GNNExplainer, PGExplainer


def gnnexplainer(data, model):
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
    explanation = explainer(data.x.cpu(), data.edge_index.cpu(), index=10)
    print(explanation.edge_mask)
    print(explanation.node_mask)
    a = (explanation.edge_mask > 0.5)
    masked_edge_index = data.edge_index[:, a]
    return masked_edge_index

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
            loss = explainer.algorithm.train(
                epoch, model, dt_batch.x, dt_batch.edge_index, target=dt_batch.y, batch=dt_batch.batch)

    # Generate the explanation for a particular graph:
    explanation = explainer(dt.x.cpu(), dt.edge_index.cpu(), target=dt.y.cpu(), batch=None)
    return explanation.edge_mask