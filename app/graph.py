"""Dependency / interface graph (networkx) with node-edge JSON and a self-contained pyvis HTML export."""
from __future__ import annotations

import html

import networkx as nx
from pyvis.network import Network

from app.db import Entity
from app.textutil import norm_key

COLORS = {"component": "#4C78A8", "interface": "#F2A93B", "undefined": "#E45756"}


def build_graph(ents: list[Entity]) -> nx.DiGraph:
    g = nx.DiGraph()
    comps = {e.key: e.name for e in ents if e.type == "component"}
    for k, n in comps.items():
        g.add_node(f"c:{k}", label=n, kind="component")

    def comp_node(name: str) -> str:
        k = norm_key(name)
        nid = f"c:{k}"
        if nid not in g:
            g.add_node(nid, label=name, kind="undefined")
        return nid

    for e in ents:
        a = e.attrs
        if e.type == "dependency" and a.get("from") and a.get("to"):
            g.add_edge(comp_node(a["from"]), comp_node(a["to"]), kind="dependency", label=a.get("dep_type") or "depends on")
        elif e.type == "signal" and a.get("sender") and a.get("receiver"):
            g.add_edge(comp_node(a["sender"]), comp_node(a["receiver"]), kind="signal", label=e.name)
        elif e.type == "port" and a.get("component") and a.get("interface"):
            inode = f"i:{norm_key(a['interface'])}"
            if inode not in g:
                g.add_node(inode, label=a["interface"], kind="interface")
            kind = "provides" if a.get("direction") == "provided" else "requires"
            g.add_edge(comp_node(a["component"]), inode, kind=kind, label=f"{kind}: {a.get('port', '')}")
    return g


def graph_json(g: nx.DiGraph) -> dict:
    return {"nodes": [{"id": n, **d} for n, d in g.nodes(data=True)],
            "edges": [{"from": u, "to": v, **d} for u, v, d in g.edges(data=True)]}


def _safe(text: str) -> str:
    """Entity names come from uploaded documents / LLM output = untrusted. pyvis embeds them in a <script>
    block as JSON, so '</script>' in a name could break out. Escaping '<', '>' and '&' prevents that."""
    return html.escape(str(text), quote=True)


def graph_html(g: nx.DiGraph, height: str = "600px") -> str:
    """Self-contained HTML (vis-network inlined: works offline). All labels are escaped."""
    net = Network(height=height, width="100%", directed=True, cdn_resources="in_line", notebook=False)
    for n, d in g.nodes(data=True):
        net.add_node(_safe(n), label=_safe(d["label"]), color=COLORS[d["kind"]], title=_safe(f"{d['kind']}: {d['label']}"),
                     shape="box" if d["kind"] != "interface" else "ellipse")
    for u, v, d in g.edges(data=True):
        net.add_edge(_safe(u), _safe(v), label=_safe(d["label"]), title=d["kind"], dashes=d["kind"] == "requires",
                     color="#888" if d["kind"] != "signal" else "#59A14F")
    net.set_options('{"physics":{"barnesHut":{"springLength":160}},"edges":{"font":{"size":10}}}')
    return net.generate_html()
