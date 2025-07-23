from lib.config.map_config import MONASTARY_IDENTIFIER
from lib.interact.map import Map
from lib.interact.meeple import Meeple
from lib.interact.structure import StructureType
from lib.interact.tile import Tile, TileModifier

from collections import defaultdict, deque
from typing import Callable, Iterator, Protocol


class SharedGameState(Protocol):
    map: Map


class GameLogic(SharedGameState):
    def _get_claims_objs(self, tile: "Tile", edge: str) -> dict[int, list[Meeple]]:
        players = defaultdict(list)

        if edge == MONASTARY_IDENTIFIER:
            m = tile.internal_claims[edge]
            if not m:
                return {}

            return {m.player_id: [m]}

        for connected_tile, e in self._traverse_connected_component(tile, edge):
            meeple = connected_tile.internal_claims[e]
            if meeple is not None:
                players[meeple.player_id].append(meeple)

        return players

    def _get_claims(self, tile: "Tile", edge: str) -> list[int]:
        players: set[int] = set()

        if edge == MONASTARY_IDENTIFIER:
            m = tile.internal_claims[edge]
            if not m:
                return []

            return [m.player_id]

        for connected_tile, e in self._traverse_connected_component(tile, edge):
            meeple = connected_tile.internal_claims[e]
            if meeple is not None:
                players.add(meeple.player_id)

        return list(players)

    def _get_reward(self, tile: "Tile", edge: str, partial: bool = False) -> int:
        visited_tiles = set()
        structure_type = tile.internal_edges[edge]

        total_points = 0

        for connected_tile, _ in self._traverse_connected_component(tile, edge):
            if connected_tile in visited_tiles:
                continue

            visited_tiles.add(connected_tile)

            if not partial:
                total_points += StructureType.get_points(structure_type)

            else:
                total_points += StructureType.get_partial_points(structure_type)

            total_points = TileModifier.apply_point_modifiers(
                connected_tile, structure_type, total_points
            )

        return total_points

    def _check_completed_component(self, start_tile: Tile, edge: str) -> bool:
        print("THIS IS DEPRECATED")
        component = list(self._traverse_connected_component(start_tile, edge))

        for tile, edge in component:
            assert tile.placed_pos is not None
            if tile.get_external_tile(edge, tile.placed_pos, self.map._grid) is None:
                return False

        return True

    def check_any_complete(self, start_tile: "Tile") -> list[str]:
        print("THIS IS DEPRECATED")
        edges_complete: list[str] = []
        for edge, tile in start_tile.get_external_tiles(self.map._grid).items():
            if tile and self._check_completed_component(start_tile, edge):
                edges_complete.append(edge)

        return edges_complete

    def get_completed_components(
        self, start_tile: "Tile"
    ) -> dict[str, set[tuple["Tile", str]]]:
        """
        Get Completed Components
        _Takes a starting tile and returns a map of internal edge to connected component if complete_
        - Does not include internal edge if this was found earlier as part of another connected component
        """
        edge_to_connected_component: dict[str, set[tuple["Tile", str]]] = {}
        internal_visited_edges: set[str] = set()

        for internal_edge, tile in start_tile.get_external_tiles(
            self.map._grid
        ).items():
            if tile is None:
                continue

            if internal_edge in internal_visited_edges:
                continue

            connected_component = list(
                self._traverse_connected_component(start_tile, internal_edge)
            )

            for connected_tile, connected_edge in connected_component:
                assert connected_tile.placed_pos
                external_tile = Tile.get_external_tile(
                    connected_edge, connected_tile.placed_pos, self.map._grid
                )
                external_edge = Tile.get_opposite(connected_edge)

                # Break if no neighbour
                if not external_tile:
                    break

                assert (external_tile, external_edge) in connected_component

                # Check if we are revisiting tile
                if connected_tile == start_tile and internal_edge != connected_edge:
                    internal_visited_edges.add(connected_edge)

            # Runs if code didn't break
            else:
                edge_to_connected_component[internal_edge] = set(connected_component)

        return edge_to_connected_component

    def _traverse_connected_component(
        self,
        start_tile: "Tile",
        edge: str,
        yield_cond: Callable[[Tile, str], bool] = lambda _1, _2: True,
        modify: Callable[[Tile, str], None] = lambda _1, _2: None,
    ) -> Iterator[tuple["Tile", str]]:
        visited = set()

        # Not a traversable edge - ie monastary etc
        if edge not in start_tile.internal_edges.keys():
            return

        structure_type = start_tile.internal_edges[edge]
        structure_bridge = TileModifier.get_bridge_modifier(structure_type)

        queue = deque([(start_tile, edge)])

        while queue:
            tile, edge = queue.popleft()

            if (tile, edge) in visited:
                continue

            # Visiting portion of traversal
            visited.add((tile, edge))
            modify(tile, edge)

            if yield_cond(tile, edge):
                yield tile, edge

            connected_internal_edges = [edge]
            opposite_edge = Tile.get_opposite(edge)

            # Check directly adjacent edges
            for adjacent_edge in Tile.adjacent_edges(edge):
                if (
                    structure_type == StructureType.CITY
                    and TileModifier.BROKEN_CITY in tile.modifiers
                ):
                    continue

                if structure_type == StructureType.ROAD_START and (
                    tile.internal_edges[adjacent_edge] == StructureType.ROAD_START
                ):
                    continue

                if tile.internal_edges[adjacent_edge] == structure_type:
                    connected_internal_edges.append(adjacent_edge)

            # Opposite edge if adajcent connection
            if (
                len(connected_internal_edges) > 1
                and tile.internal_edges[opposite_edge] == structure_type
            ):
                connected_internal_edges.append(opposite_edge)

            # Caes of opposite bridge
            elif (
                tile.internal_edges[opposite_edge] == structure_type
                and structure_bridge
                and structure_bridge in tile.modifiers
            ):
                connected_internal_edges.append(opposite_edge)

            if structure_type == StructureType.ROAD_START:
                structure_type = StructureType.ROAD
                structure_bridge = TileModifier.get_bridge_modifier(structure_type)

            for adjacent_edge in connected_internal_edges[1:]:
                visited.add((tile, adjacent_edge))
                modify(tile, adjacent_edge)

                if yield_cond(tile, adjacent_edge):
                    yield tile, adjacent_edge

            # External Tiles
            for ce in connected_internal_edges:
                assert tile.placed_pos
                ce_neighbour = Tile.get_opposite(ce)

                external_tile = Tile.get_external_tile(
                    ce, tile.placed_pos, self.map._grid
                )

                if external_tile is None:
                    continue

                if not StructureType.is_compatible(
                    structure_type, external_tile.internal_edges[ce_neighbour]
                ):
                    continue

                if (external_tile, ce_neighbour) not in visited:
                    queue.append((external_tile, ce_neighbour))
