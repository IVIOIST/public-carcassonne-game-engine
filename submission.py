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

    print(game.state.my_tiles)

    # check if in river phase

    for tile_hand_index, tile_in_hand in enumerate(game.state.my_tiles):
        river_flag = False
        for find_edge in directions.values():
            if tile_in_hand.internal_edges[find_edge] == StructureType.RIVER:
                river_flag = True
                print("river on tile")
                break
    


    if river_flag:
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
            
            print_map(game.state.map._grid, range(75, 96))

            if game.can_place_tile_at(tile_in_hand, target_x, target_y):

                uturn_check = False
                print(tile_in_hand.internal_edges[edge])
                if tile_in_hand.internal_edges[edge] != StructureType.RIVER:
                    continue

                for tile_edge in tile_in_hand.get_edges():
                    if (
                        tile_edge == edge
                        or tile_in_hand.internal_edges[tile_edge]
                        != StructureType.RIVER
                    ):
                        continue
                    forcast_coordinates_one = {
                        "top_edge": (0, -1),
                        "right_edge": (1, 0),
                        "bottom_edge": (0, 1),
                        "left_edge": (-1, 0),
                    }

                    extension = forcast_coordinates_one[tile_edge]
                    forecast_x = target_x + extension[0]
                    forecast_y = target_y + extension[1]
                    print(forecast_x, forecast_y)
                    for coords in forcast_coordinates_one.values():
                        checking_x = forecast_x + coords[0]
                        checking_y = forecast_y + coords[1]
                        if checking_x != target_x or checking_y != target_y:
                            if grid[checking_y][checking_x] is not None:
                                print("direct uturn")
                                uturn_check = True

                    forcast_coordinates_two = {
                        "top_edge": (0, -2),
                        "right_edge": (2, 0),
                        "bottom_edge": (0, 2),
                        "left_edge": (-2, 0),
                    }
                    extension = forcast_coordinates_two[tile_edge]

                    forecast_x = target_x + extension[0]
                    forecast_y = target_y + extension[1]
                    for coords in forcast_coordinates_one.values():
                        checking_x = forecast_x + coords[0]
                        checking_y = forecast_y + coords[1]
                        if grid[checking_y][checking_x] is not None:
                            print("future uturn")
                            uturn_check = True

                if uturn_check:
                    tile_in_hand.rotate_clockwise(1)
                    if tile_in_hand.internal_edges[edge] != StructureType.RIVER:
                        tile_in_hand.rotate_clockwise(2)
                
                bot_state.last_tile = tile_in_hand
                bot_state.last_tile.placed_pos = (target_x, target_y)
                print(
                    bot_state.last_tile.placed_pos,
                    tile_hand_index,
                    tile_in_hand.rotation,
                    tile_in_hand.tile_type,
                    flush=True,
                )

                print("reached")

                return game.move_place_tile(
                    query, tile_in_hand._to_model(), tile_hand_index
                )

def value_monastaries(game: Game):
    
    monasteries = []
    
    grid = game.state.map._grid

    for x in range(0,22): # 21 columns
        for y in range(0,21): # 20 rows
            tile = grid[y][x]

            if tile is not None and tile.internal_claims[MONASTARY_IDENTIFIER] is not None: 
                    
                    # check if tile is a monastery tile
                    if (tile.tile_id == "A" or tile.tile_id == "B"):
                        
                        # check all adjacent tiles
                            
                        top_left = grid[y-1][x-1]
                        top = grid[y-1][x]
                        top_right = grid[y-1][x+1]
                        middle_left = grid[y][x-1]
                        middle_right = grid[y][x+1]
                        bottom_left = grid[y+1][x-1]
                        bottom = grid[y+1][x]
                        bottom_right = grid[y+1][x+1]

                        score = 0

                        if top_left is not None:
                            score+=1
                        if top is not None: 
                            score+=1
                        if top_right is not None:
                            score+=1
                        if middle_left is not None: 
                            score+=1
                        if middle_right is not None: 
                            score+=1
                        if bottom_left is not None: 
                            score+=1
                        if bottom is not None: 
                            score+=1
                        if bottom_right is not None: 
                            score+=1


                        # create tuple for this monastery

                        monastery = (score,x,y)
                        monasteries.append(monastery)

    return monasteries         

def value_cities():
    pass

def value_roads():
    pass

def value_fields():
    pass

def handle_place_meeple(
    game: Game, bot_state: BotState, query: QueryPlaceMeeple
) -> MovePlaceMeeplePass | MovePlaceMeeple:
    """
    Try to place a meeple on the most recently placed tile.
    Priority order: monastery -> Anything else
    """

    print("tiles are\n", flush = True)

    value_monastaries(game)

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

    for edge in placement_priorities:
        # Check if this edge has a valid structure and is unclaimed
        if edge == MONASTARY_IDENTIFIER:
            # Check if tile has monastery and it's unclaimed
            if (
                hasattr(recent_tile, "modifiers")
                and any(mod.name == "MONESTERY" for mod in recent_tile.modifiers)
                and recent_tile.internal_claims.get(MONASTARY_IDENTIFIER) is None
            ):
                assert bot_state.last_tile
                print(
                    "[ ERROR ] M ",
                    recent_tile,
                    edge,
                    bot_state.last_tile.internal_edges[edge],
                    flush=True,
                )
                return game.move_place_meeple(
                    query, recent_tile._to_model(), MONASTARY_IDENTIFIER
                )
        else:
            # Check if edge has a claimable structure
            assert bot_state.last_tile
            structures = list(
                game.state.get_placeable_structures(bot_state.last_tile._to_model()).items()
            )

            e, structure = structures[0] if structures else None, None

            if structure and e and recent_tile.internal_claims.get(edge) is None:
                # Check if the structure is actually unclaimed (not connected to claimed structures)
                if not game.state._get_claims(recent_tile, e):
                    print(
                        "[ ERROR ] ",
                        recent_tile,
                        edge,
                        bot_state.last_tile.internal_edges[edge],
                        flush=True,
                    )
                    return game.move_place_meeple(query, recent_tile._to_model(), edge)

    # No valid placement found, pass
    print("[ ERROR ] ", flush=True)
    return game.move_place_meeple_pass(query)


if __name__ == "__main__":
    main()
