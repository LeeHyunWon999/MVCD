"""Graph <-> JSON helpers retained from the MLLM-CD integration.

A graph is stored as a plain dict so predicted graphs, ground-truth graphs and
faithful-FCI graphs all share one representation:

    {
      "nodes": ["color", "size", ...],
      "edges": [
        {"from": "defects", "to": "aroma", "ep_from": "TAIL", "ep_to": "ARROW"},
        ...
      ]
    }

Endpoints follow causal-learn's PAG convention: TAIL ("-"), ARROW (">"),
CIRCLE ("o"). A directed edge A->B is (ep_from=TAIL, ep_to=ARROW).
"""
import json


def endpoint_name(ep):
    """causal-learn Endpoint enum -> 'TAIL'/'ARROW'/'CIRCLE' (robust to import)."""
    name = getattr(ep, "name", str(ep))
    return name.upper()


def graph_to_dict(g, labels):
    """Convert a causal-learn GeneralGraph (nodes X1..Xn) + label list to a dict.

    `labels[i]` is the human-readable name of internal node "X{i+1}".
    """
    idx_to_label = {f"X{i + 1}": labels[i] for i in range(len(labels))}
    edges = []
    for e in g.get_graph_edges():
        n1 = e.get_node1().get_name()
        n2 = e.get_node2().get_name()
        edges.append({
            "from": idx_to_label.get(n1, n1),
            "to": idx_to_label.get(n2, n2),
            "ep_from": endpoint_name(e.get_endpoint1()),
            "ep_to": endpoint_name(e.get_endpoint2()),
        })
    return {"nodes": list(labels), "edges": edges}


def directed_graph_dict(nodes, directed_edges):
    """Build a dict for a plain DAG given (from, to) directed edges."""
    edges = [{"from": a, "to": b, "ep_from": "TAIL", "ep_to": "ARROW"}
             for (a, b) in directed_edges]
    return {"nodes": list(nodes), "edges": edges}


def relabel(graph_dict, mapping):
    """Return a copy of graph_dict with node names remapped (e.g. score->diagnosis)."""
    def m(x):
        return mapping.get(x, x)
    return {
        "nodes": [m(n) for n in graph_dict["nodes"]],
        "edges": [{**e, "from": m(e["from"]), "to": m(e["to"])}
                  for e in graph_dict["edges"]],
    }


def save_graph_json(graph_dict, path):
    with open(path, "w") as f:
        json.dump(graph_dict, f, indent=2, ensure_ascii=False)


def load_graph_json(path):
    with open(path) as f:
        return json.load(f)


def save_graph_from_causal_learn(g, labels, path):
    """Convenience for the pipeline: dump a live FCI graph object to JSON."""
    save_graph_json(graph_to_dict(g, labels), path)
    return path


# --- DOT interop ----------------------------------------------------------
# The authors' metrics.py evaluates graphs given as DOT strings, so we bridge
# between our graph dicts and DOT. Endpoints follow the same causal-learn PAG
# convention used by GraphUtils.to_pydot (dir=both + arrowtail/arrowhead).
_EP_TO_DOT = {"TAIL": "none", "ARROW": "normal", "CIRCLE": "odot"}
_DOT_TO_EP = {"none": "TAIL", "normal": "ARROW", "vee": "ARROW",
              "empty": "ARROW", "odot": "CIRCLE", "star": "ARROW"}


def to_dot_str(graph_dict):
    """Serialize a graph dict to a DOT string the authors' metrics.py can parse.

    Node ids are X1..Xn with the real name in the label (the authors' parser
    reads the label and skips bare 'X..' ids), so factor names with characters
    like ':' that would otherwise be read as DOT ports are kept intact.
    """
    idx = {n: f"X{i + 1}" for i, n in enumerate(graph_dict["nodes"])}
    lines = ["digraph {"]
    for name, nid in idx.items():
        safe = name.replace('"', '\\"')
        lines.append(f'{nid} [label="{safe}"];')
    for e in graph_dict["edges"]:
        at = _EP_TO_DOT.get(e["ep_from"], "none")
        ah = _EP_TO_DOT.get(e["ep_to"], "normal")
        lines.append(f'{idx[e["from"]]} -> {idx[e["to"]]} '
                     f'[dir=both, arrowtail={at}, arrowhead={ah}];')
    lines.append("}")
    return "\n".join(lines)


def from_dot_str(dot_str):
    """Parse a DOT string (our export or a pipeline pdy_str) back to a graph dict.

    Assumes the causal-learn 'dir=both, arrowtail=.., arrowhead=..' edge style
    that both GraphUtils.to_pydot and to_dot_str emit. Node names come from the
    label if present, else the raw id.
    """
    import pydot
    (g,) = pydot.graph_from_dot_data(dot_str)

    def name_of(nid):
        objs = g.get_node(nid)
        if isinstance(objs, list) and objs:
            lab = objs[0].get("label")
        elif objs is not None and not isinstance(objs, list):
            lab = objs.get("label")
        else:
            lab = None
        return lab.strip('"') if lab else nid.strip('"')

    nodes, seen = [], set()
    for nd in g.get_nodes():
        nid = nd.get_name()
        if nid in ("node", "edge", "graph"):
            continue
        nm = name_of(nid)
        if nm not in seen:
            seen.add(nm)
            nodes.append(nm)

    edges = []
    for e in g.get_edges():
        a, b = name_of(e.get_source()), name_of(e.get_destination())
        for nm in (a, b):
            if nm not in seen:
                seen.add(nm)
                nodes.append(nm)
        at = (e.get("arrowtail") or "none").lower()
        ah = (e.get("arrowhead") or "normal").lower()
        edges.append({"from": a, "to": b,
                      "ep_from": _DOT_TO_EP.get(at, "TAIL"),
                      "ep_to": _DOT_TO_EP.get(ah, "ARROW")})
    return {"nodes": nodes, "edges": edges}
