"""
Demo bot.

Blame: Maurice

Logic:
- Always place tiles next to the last placed tile. If not possible,
- Greedily place meeple in monastary -> next valid thing


Flaws:
- Does not take advantage of rotating the tiles
- Does not care about moves of other players other than for the sake of validation
- Does not account for other existing structures/linking other structures
- As of writing, cannot pass river stage

"""

from helper.game import Game
from lib.interact.tile import Tile
from lib.interface.events.moves.move_place_tile import MovePlaceTile
from lib.interface.events.moves.move_place_meeple import (
    MovePlaceMeeple,
    MovePlaceMeeplePass,
)
from lib.interface.queries.typing import QueryType
from lib.interface.queries.query_place_tile import QueryPlaceTile
from lib.interface.queries.query_place_meeple import QueryPlaceMeeple
from lib.interface.events.moves.typing import MoveType
from lib.config.map_config import MAX_MAP_LENGTH
from lib.config.map_config import MONASTARY_IDENTIFIER
from lib.interact.structure import StructureType
from helper.utils import print_map


class BotState:
    """A class for us to locally the state of the game and what we find relevant"""

    def __init__(self):
        self.last_tile: Tile | None = None
        self.meeples_placed: int = 0


def main():
    game = Game()
    bot_state = BotState()

    while True:
        query = game.get_next_query()

        def choose_move(query: QueryType) -> MoveType:
            match query:
                case QueryPlaceTile() as q:
                    print("placing tile")
                    return handle_place_tile(game, bot_state, q)

                case QueryPlaceMeeple() as q:
                    print("meeple")
                    return handle_place_meeple(game, bot_state, q)
                case _:
                    assert False

        print("sending move")
        game.send_move(choose_move(query))



def handle_place_tile(
    game: Game, bot_state: BotState, query: QueryPlaceTile
) -> MovePlaceTile:
    """
    Find the most recently placed tile and try to place a new tile adjacent to it.
    Tries directions in order: right, bottom, left, top
    """
    grid = game.state.map._grid



    # The direction of placing the tile in reference to the last placed tile
    directions = {
        (
            1,
            0,
        ): "left_edge",  # if we place on the right of the target tile, we will have to consider our left_edge of the tile in our hand
        (
            0,
            1,
        ): "top_edge",  # if we place at the bottom o0f the target tile, we will have to consider the top_edge of
        (-1, 0): "right_edge",  # left
        (0, -1): "bottom_edge",  # top
    }
    # Will either be the latest tile
    latest_tile = game.state.map.placed_tiles[-1]
    latest_pos = latest_tile.placed_pos

    print("Round:", game.state.round)
    print(game.state.event_history)
    assert latest_pos

    # Try to place a tile adjacent to the latest tile
    for tile_hand_index, tile_in_hand in enumerate(game.state.my_tiles):
        river_flag = False
        for find_edge in directions.values():
            if tile_in_hand.internal_edges[find_edge] == StructureType.RIVER:
                river_flag = True
                print("river on tile")
                break

        # Looking at each edge of the target tile and seeing if we can match it
        for (dx, dy), edge in directions.items():
            target_x = latest_pos[0] + dx
            target_y = latest_pos[1] + dy

            # Check bounds
            if not (0 <= target_x < MAX_MAP_LENGTH and 0 <= target_y < MAX_MAP_LENGTH):
                continue

            # Check if position is empty
            if grid[target_y][target_x] is not None:
                continue

            # Cheeck if this is a river tile
            # Try placing the tile at this position by rotating it

            # print_map(game.state.map._grid, range(75, 96))

            if game.can_place_tile_at(tile_in_hand, target_x, target_y):
                if river_flag:
                    print(tile_in_hand.internal_edges[edge])
                    if tile_in_hand.internal_edges[edge] != StructureType.RIVER:
                        continue

                bot_state.last_tile = tile_in_hand
                bot_state.last_tile.placed_pos = (target_x, target_y)
                print(
                    bot_state.last_tile.placed_pos,
                    tile_hand_index,
                    tile_in_hand.rotation,
                    tile_in_hand.tile_type,
                    flush=True,
                )

                return game.move_place_tile(
                    query, tile_in_hand._to_model(), tile_hand_index
                )
    print("could not find with heuristic")
    return brute_force_tile(game, bot_state, query)


def handle_place_meeple(
    game: Game, bot_state: BotState, query: QueryPlaceMeeple
) -> MovePlaceMeeplePass | MovePlaceMeeple:
    """
    Try to place a meeple on the most recently placed tile.
    Priority order: monastery -> Anything else
    """
    recent_tile = bot_state.last_tile
    if not recent_tile:
        return game.move_place_meeple_pass(query)

    # Priority order for meeple placement
    placement_priorities = [
        MONASTARY_IDENTIFIER,  # monastery
        "top_edge",
        "right_edge",
        "bottom_edge",
        "left_edge",  # edges
    ]

    if bot_state.meeples_placed == 7:
        print("no meeple :(")
        return game.move_place_meeple_pass(query)

    for edge in placement_priorities:
        # Check if this edge has a valid structure and is unclaimed
        if edge == MONASTARY_IDENTIFIER:
            print("looking for monastary")
            # Check if tile has monastery and it's unclaimed
            if (
                hasattr(recent_tile, "modifiers")
                and any(mod.name == "MONESTARY" for mod in recent_tile.modifiers)
                # and recent_tile.internal_claims.get(MONASTARY_IDENTIFIER) is None
            ):
                print("found monastary")
                # assert bot_state.last_tile
                print(
                    "[ Placed meeple ] M ",
                    recent_tile,
                    edge,
                    # bot_state.last_tile.internal_edges[edge],
                    flush=True,
                )
                bot_state.meeples_placed += 1
                return game.move_place_meeple(
                    query, recent_tile._to_model(), MONASTARY_IDENTIFIER
                )
        else:
            # Check if edge has a claimable structure
            assert bot_state.last_tile
            structures = list(
                game.state.get_placeable_structures(
                    bot_state.last_tile._to_model()
                ).items()
            )
            print("structurees: ", structures)
            # e, structure = structures[0] if structures else None, None
            # print("e: ", e)
            # print("structure:", structure)

            if recent_tile.internal_claims.get(edge) is None:
                print("Edge bot is looking at:", edge)
                # print("structure type, ", bot_state.last_tile.internal_edges[edge])
                # # Check if the structure is actually unclaimed (not connected to claimed structures)
                # print("from game state get claims: ", game.state._get_claims(recent_tile, edge))
                # print("river check: ", bot_state.last_tile.internal_edges[edge] != StructureType.RIVER)
                # print("grass check: ", )

                if (
                    not game.state._get_claims(recent_tile, edge)
                    and not game.state._check_completed_component(recent_tile, edge)
                    and bot_state.last_tile.internal_edges[edge] != StructureType.RIVER
                    and bot_state.last_tile.internal_edges[edge] != StructureType.GRASS
                ):
                    print(
                        "[ Placed meeple ] ",
                        recent_tile,
                        edge,
                        bot_state.last_tile.internal_edges[edge],
                        flush=True,
                    )
                    bot_state.meeples_placed += 1
                    return game.move_place_meeple(query, recent_tile._to_model(), edge)

    # No valid placement found, pass
    print("[ ERROR ] ", flush=True)
    print_map(game.state.map._grid, range(75, 96))

    return game.move_place_meeple_pass(query)


# """Copied from simple.py: brute force a valid tile"""


def brute_force_tile(
    game: Game, bot_state: BotState, query: QueryPlaceTile
) -> MovePlaceTile:
    grid = game.state.map._grid
    directions = {
        (0, 1): "top",
        (1, 0): "right",
        (0, -1): "bottom",
        (-1, 0): "left",
    }

    # print(game.state.event_history)

    print("Cards", game.state.my_tiles)

    # assert bot_state.last_tile

    for y in range(70, 90):
        for x in range(70, 90):
            if grid[y][x] is not None:
                print(f"Checking if tile can be placed near tile - {grid[y][x]}")
                for tile_index, tile in enumerate(game.state.my_tiles):
                    for direction in directions:
                        dx, dy = direction
                        x1, y1 = (x + dx, y + dy)

                        if game.can_place_tile_at(tile, x1, y1):
                            bot_state.last_tile = tile
                            bot_state.last_tile.placed_pos = x1, y1
                            return game.move_place_tile(
                                query, tile._to_model(), tile_index
                            )


if __name__ == "__main__":
    main()
