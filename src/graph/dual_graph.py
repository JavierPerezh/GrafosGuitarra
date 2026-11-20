from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from src.domain.musical_event import MusicalSequence
from src.graph.base_dag import LayeredDAG, PostureNode, SOURCE_NODE_ID
from src.graph.cost_vector import CostVector, CostVectorEvaluator, zero_cost_vector

DUAL_SOURCE_ID = "DUAL_SOURCE"
DUAL_SINK_ID = "DUAL_SINK"


@dataclass(frozen=True)
class CompoundNode:
    node_id: str
    layer_index: int  # Índice de la capa destino (t_i)
    prev_posture: Optional[PostureNode]
    curr_posture: PostureNode


@dataclass(frozen=True)
class CompoundEdge:
    source_id: str
    target_id: str
    cost_vector: CostVector


@dataclass(frozen=True)
class CompoundGraph:
    nodes_by_id: Dict[str, CompoundNode]
    layers: Tuple[Tuple[CompoundNode, ...], ...]
    adjacency: Dict[str, Tuple[CompoundEdge, ...]]
    reverse_adjacency: Dict[str, Tuple[CompoundEdge, ...]]
    source_id: str = DUAL_SOURCE_ID
    sink_id: str = DUAL_SINK_ID

    @property
    def num_compound_nodes(self) -> int:
        return len(self.nodes_by_id)

    @property
    def num_edges(self) -> int:
        return sum(len(edges) for edges in self.adjacency.values())

    def extract_posture_path(
        self, compound_path: Sequence[str]
    ) -> Tuple[PostureNode, ...]:
        postures: List[PostureNode] = []
        for cid in compound_path:
            if cid in (self.source_id, self.sink_id):
                continue
            node = self.nodes_by_id[cid]
            if not postures or postures[-1].node_id != node.curr_posture.node_id:
                postures.append(node.curr_posture)
        return tuple(postures)


class DualGraphBuilder:
    def __init__(self, cost_evaluator: CostVectorEvaluator) -> None:
        self.cost_evaluator = cost_evaluator

    def build(
        self,
        base_dag: LayeredDAG,
        sequence: MusicalSequence,
    ) -> CompoundGraph:
        if base_dag.num_layers != len(sequence):
            raise ValueError(
                f"Desajuste entre capas del DAG ({base_dag.num_layers}) "
                f"y eventos musicales ({len(sequence)})."
            )

        nodes_by_id: Dict[str, CompoundNode] = {}
        compound_layers: List[Tuple[CompoundNode, ...]] = []
        adj_lists: Dict[str, List[CompoundEdge]] = {
            DUAL_SOURCE_ID: [],
            DUAL_SINK_ID: [],
        }
        rev_lists: Dict[str, List[CompoundEdge]] = {
            DUAL_SOURCE_ID: [],
            DUAL_SINK_ID: [],
        }

        def register_node(cnode: CompoundNode) -> None:
            nodes_by_id[cnode.node_id] = cnode
            adj_lists[cnode.node_id] = []
            rev_lists[cnode.node_id] = []

        def add_edge(src_id: str, dst_id: str, cost: CostVector) -> None:
            edge = CompoundEdge(source_id=src_id, target_id=dst_id, cost_vector=cost)
            adj_lists[src_id].append(edge)
            rev_lists[dst_id].append(edge)

        # Capa 0: cápsulas iniciales (SOURCE -> u_0)
        layer_0: List[CompoundNode] = []
        for u0 in base_dag.layers[0]:
            cid = f"C0_{SOURCE_NODE_ID}__{u0.node_id}"
            cnode = CompoundNode(
                node_id=cid,
                layer_index=0,
                prev_posture=None,
                curr_posture=u0,
            )
            register_node(cnode)
            layer_0.append(cnode)
            add_edge(DUAL_SOURCE_ID, cid, zero_cost_vector())
        compound_layers.append(tuple(layer_0))

        # Capas 1..m-1: cápsulas de transición (u_{i-1} -> u_i) y enlaces de 2do orden
        for idx in range(1, base_dag.num_layers):
            prev_layer_nodes = compound_layers[idx - 1]
            curr_base_layer = base_dag.layers[idx]
            curr_compound_layer: List[CompoundNode] = []

            prev_by_curr_id: Dict[str, List[CompoundNode]] = {}
            for prev_cnode in prev_layer_nodes:
                prev_by_curr_id.setdefault(prev_cnode.curr_posture.node_id, []).append(
                    prev_cnode
                )

            event_prev = sequence.events[idx - 1]
            event_curr = sequence.events[idx]

            for u_prev in base_dag.layers[idx - 1]:
                incoming_cnodes = prev_by_curr_id.get(u_prev.node_id, [])
                for u_curr in curr_base_layer:
                    cid = f"C{idx}_{u_prev.node_id}__{u_curr.node_id}"
                    cnode = CompoundNode(
                        node_id=cid,
                        layer_index=idx,
                        prev_posture=u_prev,
                        curr_posture=u_curr,
                    )
                    register_node(cnode)
                    curr_compound_layer.append(cnode)

                    for prev_cnode in incoming_cnodes:
                        if prev_cnode.prev_posture is None:
                            cost = self.cost_evaluator.evaluate_first_order(
                                curr_node=u_prev,
                                next_node=u_curr,
                                curr_event=event_prev,
                                next_event=event_curr,
                            )
                        else:
                            cost = self.cost_evaluator.evaluate_second_order(
                                prev_node=prev_cnode.prev_posture,
                                curr_node=u_prev,
                                next_node=u_curr,
                                curr_event=event_prev,
                                next_event=event_curr,
                            )
                        add_edge(prev_cnode.node_id, cid, cost)

            compound_layers.append(tuple(curr_compound_layer))

        for final_cnode in compound_layers[-1]:
            add_edge(final_cnode.node_id, DUAL_SINK_ID, zero_cost_vector())

        return CompoundGraph(
            nodes_by_id=nodes_by_id,
            layers=tuple(compound_layers),
            adjacency={k: tuple(v) for k, v in adj_lists.items()},
            reverse_adjacency={k: tuple(v) for k, v in rev_lists.items()},
        )