import random

from argparse import ArgumentParser

import torch

from src.esmc_protein_function.model import ESMCProteinFunction

from torch.cuda import is_available as cuda_is_available
from torch.backends.mps import is_available as mps_is_available

import obonet

import networkx as nx

import plotly.graph_objects as go
import plotly.io as pio

from plotly.subplots import make_subplots


def _get_dag_layers(subgraph):
    generations = list(nx.topological_generations(subgraph))

    return {i: list(gen) for i, gen in enumerate(generations)}


def build_aspect_figure(subgraph, probabilities, title):
    layers = _get_dag_layers(subgraph)

    pos = nx.multipartite_layout(subgraph, subset_key=layers)

    for node in pos:
        pos[node] = (pos[node][0], -pos[node][1])

    edge_x = []
    edge_y = []

    for u, v in subgraph.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]

        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        line=dict(width=0.5, color="#888"),
        hoverinfo="none",
        showlegend=False,
    )

    node_x = []
    node_y = []
    node_text = []
    node_hovertext = []
    node_color = []

    for go_term in subgraph.nodes():
        x, y = pos[go_term]

        node_x.append(x)
        node_y.append(y)

        name = subgraph.nodes[go_term].get("name", "")

        node_text.append(name)

        prob = probabilities.get(go_term, 0.0)

        node_color.append(prob)

        node_hovertext.append(
            f"GO: {go_term}<br>Name: {name}<br>Probability: {prob:.4f}"
        )

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=node_text,
        hovertext=node_hovertext,
        hoverinfo="text",
        showlegend=False,
        marker=dict(
            size=40,
            color=node_color,
            colorscale="PiYG",
            cmin=0,
            cmax=1,
            showscale=True,
            colorbar=dict(title="Probability", thickness=15, len=0.5),
        ),
        textposition="bottom center",
        textfont=dict(size=20, color="black"),
    )

    fig = go.Figure(data=[edge_trace, node_trace])

    fig.update_layout(
        title=title,
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        hovermode="closest",
        margin=dict(l=20, r=20, t=60, b=20),
    )

    return fig


def build_combined_figure(aspect_results, titles):
    figs = []

    for (subgraphs, probabilities), title in zip(aspect_results, titles):
        subgraph = subgraphs[0]
        probabilities = probabilities[0]

        fig = build_aspect_figure(subgraph, probabilities, title)

        figs.append(fig)

    fig = make_subplots(
        rows=3,
        cols=1,
        subplot_titles=titles,
        vertical_spacing=0.02,
    )

    for i, f in enumerate(figs):
        for trace in f.data:
            fig.add_trace(trace, row=i + 1, col=1)

        fig.update_xaxes(
            showgrid=False, zeroline=False, showticklabels=False, row=i + 1, col=1
        )

        fig.update_yaxes(
            showgrid=False, zeroline=False, showticklabels=False, row=i + 1, col=1
        )

    fig.update_layout(
        hovermode="closest",
        margin=dict(l=20, r=20, t=40, b=20),
    )

    return fig


def main():
    parser = ArgumentParser(
        description="Predict the gene ontology (GO) subgraph associated with a protein sequence."
    )

    parser.add_argument(
        "--checkpoint_path", default="./checkpoints/checkpoint.pt", type=str
    )

    parser.add_argument("--go_db_path", default="./dataset/go-basic.obo", type=str)
    parser.add_argument("--context_length", default=2048, type=int)
    parser.add_argument("--top_p", default=0.5, type=float)
    parser.add_argument("--quantize_weights", action="store_true")
    parser.add_argument("--quant_group_size", default=192, type=int)
    parser.add_argument("--device", default="cpu", type=str)
    parser.add_argument("--seed", default=None, type=int)

    args = parser.parse_args()

    if args.context_length < 1:
        raise ValueError(
            f"Context length must be greater than 0, {args.context_length} given."
        )

    if args.top_p < 0.0 or args.top_p > 1.0:
        raise ValueError(f"Top p must be between 0 and 1, {args.top_p} given.")

    if "cuda" in args.device and not cuda_is_available():
        raise RuntimeError("Cuda is not available.")

    if "mps" in args.device and not mps_is_available():
        raise RuntimeError("MPS is not available.")

    torch.set_float32_matmul_precision("high")

    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    checkpoint = torch.load(
        args.checkpoint_path, map_location="cpu", weights_only=False
    )

    model = ESMCProteinFunction.from_esm_pretrained(**checkpoint["model_args"])

    model.load_state_dict(checkpoint["model"])

    model.remove_fake_quantized_tensors()

    if args.quantize_weights:
        model.quantize_weights(group_size=args.quant_group_size)

    model = model.to(args.device)

    model.eval()

    print("Checkpoint loaded successfully.")

    graph = obonet.read_obo(args.go_db_path)

    model.load_gene_ontology(graph)

    print("Gene ontology loaded successfully.")

    titles = ["Molecular Function", "Biological Process", "Cellular Component"]

    while True:
        sequence = input("Enter a sequence: ").replace(" ", "").replace("\n", "")

        out = model.tokenizer(
            sequence,
            max_length=args.context_length,
            truncation=True,
        )

        input_ids = torch.tensor(out["input_ids"], dtype=torch.int64).to(args.device)

        input_ids = input_ids.unsqueeze(0)

        aspect_results = model.predict_all_subgraphs(input_ids, top_p=args.top_p)

        fig = build_combined_figure(aspect_results, titles)

        fig.show(config={"responsive": False})

        if "y" not in input("Go again? (yes|no): ").lower():
            break


if __name__ == "__main__":
    main()
