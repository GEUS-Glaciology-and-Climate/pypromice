from typing import Mapping, Set, MutableMapping, Optional

import attr

__all__ = [
    "DependencyNode",
    "DependencyGraph",
]


@attr.define
class DependencyNode:
    name: str = attr.field()
    parents: Set["DependencyNode"] = attr.field(factory=set)
    children: Set["DependencyNode"] = attr.field(factory=set)

    def add_child(self, child: "DependencyNode"):
        self.children.add(child)
        child.parents.add(self)

    def get_children_closure(self, closure: Optional[Set[str]] = None) -> Set[str]:
        is_root = closure is None
        if closure is None:
            closure = set()
        if self.name in closure:
            return closure
        closure.add(self.name)
        for child in self.children:
            closure |= child.get_children_closure(closure)

        if is_root:
            closure.remove(self.name)
        return closure

    def get_parents_closure(self, closure: Optional[Set[str]] = None) -> Set[str]:
        is_root = closure is None
        if closure is None:
            closure = set()
        if self.name in closure:
            return closure
        closure.add(self.name)
        for parent in self.parents:
            closure |= parent.get_parents_closure(closure)

        if is_root:
            closure.remove(self.name)
        return closure

    def __hash__(self):
        return hash(self.name)


@attr.define
class DependencyGraph:
    nodes: MutableMapping[str, DependencyNode] = attr.field(factory=dict)

    def add_node(self, name: str) -> DependencyNode:
        if name not in self.nodes:
            self.nodes[name] = DependencyNode(name)
        return self.nodes[name]

    def add_edge(self, parent: str, child: str):
        parent_node = self.add_node(parent)
        child_node = self.add_node(child)
        parent_node.add_child(child_node)

    @classmethod
    def from_child_mapping(cls, mapping: Mapping[str, Set[str]]) -> "DependencyGraph":
        graph = cls()
        for var, children in mapping.items():
            graph.add_node(var)
            for child in children:
                graph.add_edge(var, child)
        return graph

    @classmethod
    def from_parent_mapping(cls, mapping: Mapping[str, Set[str]]) -> "DependencyGraph":
        graph = cls()
        for var, parents in mapping.items():
            graph.add_node(var)
            for parent in parents:
                graph.add_edge(parent, var)
        return graph

    def child_mapping(self) -> Mapping[str, Set[str]]:
        return {
            node.name: {child.name for child in node.children}
            for node in self.nodes.values()
        }

    def child_closure_mapping(self) -> Mapping[str, Set[str]]:
        return {node.name: node.get_children_closure() for node in self.nodes.values()}

    def parent_mapping(self) -> Mapping[str, Set[str]]:
        return {
            node.name: {parent.name for parent in node.parents}
            for node in self.nodes.values()
        }

    def parent_closure_mapping(self) -> Mapping[str, Set[str]]:
        return {node.name: node.get_parents_closure() for node in self.nodes.values()}
