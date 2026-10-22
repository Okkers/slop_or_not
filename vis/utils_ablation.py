import torch
import pandas as pd
import os

@torch.no_grad()
def predict_ablation(model, result, device, mode="original"):

    x = result["x"].to(device)
    edge_index = result["edge_index"].to(device)
    edge_weight = result["edge_weight"].to(device)

    if mode == "no_edges":
        edge_index = edge_index[:, :0]
        edge_weight = edge_weight[:0]

    elif mode == "no_features":
        x = torch.zeros_like(x)

    elif mode == "no_edge_weights":
        edge_weight = torch.ones_like(edge_weight)

    elif mode == "no_edges_no_features":
        x = torch.zeros_like(x)
        edge_index = edge_index[:, :0]
        edge_weight = edge_weight[:0]

    elif mode == "no_count":
        x[:, 0] = 0

    elif mode == "no_frequency":
        x[:, 1] = 0

    elif mode == "no_embeddings":
        x[:, 2:] = 0

    elif mode != "original":
        raise ValueError(f"Unknown ablation mode: {mode}")


    # All nodes belong to one graph
    batch = torch.zeros(
        x.shape[0],
        dtype=torch.long,
        device=device
    )

    model.eval()

    logits = model(
        x,
        edge_index,
        edge_weight=edge_weight,
        batch=batch
    )

    probabilities = torch.softmax(logits, dim=-1)

    return probabilities[0].cpu()

def evaluate_ablations(results, folder, path_base, model, device):
    os.makedirs(folder, exist_ok=True)
    modes = [
    "original",
    "no_edges",
    "no_features",
    "no_count",
    "no_frequency",
    "no_embeddings",
    "no_edge_weights",
    "no_edges_no_features",
    ]

    rows = []

    for result in results:

        label = result["label"]

        p_original = predict_ablation(
            model, result, device, mode="original"
        )

        if int(p_original.argmax()) != result["pred"]:
            raise ValueError(
                f"Prediction mismatch for graph {result['row']}"
            )

        for mode in modes:

            probabilities = predict_ablation(
                model, result, device, mode=mode
            )

            prediction = int(probabilities.argmax())

            rows.append({
                "row": result["row"],
                "class": "Human" if label == 0 else "Machine",
                "mode": mode,
                "label": label,
                "prediction": prediction,
                "correct": int(prediction == label),
                "p_true_class": probabilities[label].item(),
                "confidence_drop": (
                    p_original[label] - probabilities[label]
                ).item()
            })

    df = pd.DataFrame(rows)
    path = os.path.join(
        folder,
        f"{path_base}_abl.csv"
    )
    df.to_csv(
        path,
        index=False
    )
    return df

