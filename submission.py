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
    else:
        # Non-river phase strategy
        best_score = float("-inf")
        best_move = None
        best_tile_index = None
        best_rotation = None

        # Define a scoring function to evaluate placements
        def evaluate_placement(game, tile, x, y):
            score = 0
            grid = game.state.map._grid
            
            # Higher score for monastery placements with more surrounding tiles
            if MONASTARY_IDENTIFIER in tile.tile_type:
                adjacent_count = 0
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        nx, ny = x + dx, y + dy
                        if (0 <= nx < MAX_MAP_LENGTH and 0 <= ny < MAX_MAP_LENGTH and 
                            grid[ny][nx] is not None and (dx != 0 or dy != 0)):
                            adjacent_count += 1
                score += adjacent_count * 3
            
            # Check edge connections
            adjacency_map = {
                "top_edge": (0, -1, "bottom_edge"),
                "right_edge": (1, 0, "left_edge"),
                "bottom_edge": (0, 1, "top_edge"),
                "left_edge": (-1, 0, "right_edge"),
            }
            
            for my_edge, (dx, dy, their_edge) in adjacency_map.items():
                nx, ny = x + dx, y + dy
                if (0 <= nx < MAX_MAP_LENGTH and 0 <= ny < MAX_MAP_LENGTH and grid[ny][nx] is not None):
                    adjacent_tile = grid[ny][nx]
                    if tile.internal_edges[my_edge] == adjacent_tile.internal_edges[their_edge]:
                        if tile.internal_edges[my_edge] == StructureType.CITY:
                            score += 10  # Cities are high value
                        elif tile.internal_edges[my_edge] == StructureType.ROAD:
                            score += 5   # Roads are medium value
            
            # Count city and road segments on the tile
            city_count = sum(1 for edge in tile.internal_edges.values() if edge == StructureType.CITY)
            road_count = sum(1 for edge in tile.internal_edges.values() if edge == StructureType.ROAD)
            
            score += city_count * 2
            score += road_count
            
            return score

        # Try each tile in hand
        for tile_hand_index, tile_in_hand in enumerate(game.state.my_tiles):
            original_rotation = tile_in_hand.rotation
            
            # Try different rotations
            for _ in range(4):
                tile_in_hand.rotate_clockwise(1)
                
                # Check positions adjacent to existing tiles
                for placed_tile in game.state.map.placed_tiles:
                    px, py = placed_tile.placed_pos
                    
                    # Check all adjacent positions
                    for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                        x, y = px + dx, py + dy
                        
                        if not (0 <= x < MAX_MAP_LENGTH and 0 <= y < MAX_MAP_LENGTH) or grid[y][x] is not None:
                            continue
                        
                        if game.can_place_tile_at(tile_in_hand, x, y):
                            score = evaluate_placement(game, tile_in_hand, x, y)
                            
                            if score > best_score:
                                best_score = score
                                best_move = (x, y)
                                best_tile_index = tile_hand_index
                                best_rotation = tile_in_hand.rotation
            
            # Reset rotation to original
            while tile_in_hand.rotation != original_rotation:
                tile_in_hand.rotate_clockwise(1)

        # Use the best move found
        if best_move:
            x, y = best_move
            chosen_tile = game.state.my_tiles[best_tile_index]
            
            # Set rotation to match best found
            while chosen_tile.rotation != best_rotation:
                chosen_tile.rotate_clockwise(1)
            
            bot_state.last_tile = chosen_tile
            bot_state.last_tile.placed_pos = (x, y)
            
            return game.move_place_tile(query, chosen_tile._to_model(), best_tile_index)
        else:
            # Fallback to first valid move
            for tile_hand_index, tile_in_hand in enumerate(game.state.my_tiles):
                for _ in range(4):
                    tile_in_hand.rotate_clockwise(1)
                    for y in range(MAX_MAP_LENGTH):
                        for x in range(MAX_MAP_LENGTH):
                            if grid[y][x] is None and game.can_place_tile_at(tile_in_hand, x, y):
                                bot_state.last_tile = tile_in_hand
                                bot_state.last_tile.placed_pos = (x, y)
                                return game.move_place_tile(query, tile_in_hand._to_model(), tile_hand_index)

def value_monastaries(game: Game):
    
    monasteries = []
    
    grid = game.state.map._grid

    for y in range(len(grid)):
        for x in range(len(grid[0])):
            tile = grid[y][x]

            if tile is not None: 
                    
                    # check if tile is a monastery tile

                    # removed R8 cause it seems invalid?

                    if ((tile.tile_type == "A" or tile.tile_type == "B")and tile.internal_claims[MONASTARY_IDENTIFIER] is None):

                        print("monastary found",flush = True)

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

    monasteries.sort(key=lambda score: score[0], reverse=True) # get monasteries from highest score to lowest

    return monasteries

def value_cities(game: Game):

    grid = game.state.map._grid
    seen_edges = set()
    cities = []

    # Edge mapping: where to look and what the neighbor’s edge is
    edge_directions = {
        "top_edge": (0, -1, "bottom_edge"),
        "bottom_edge": (0, 1, "top_edge"),
        "left_edge": (-1, 0, "right_edge"),
        "right_edge": (1, 0, "left_edge"),
    }

    def is_city(structure):
        return structure == StructureType.CITY

    for y in range(len(grid)):
        for x in range(len(grid[0])):
            tile = grid[y][x]
            if tile is None:
                continue

            for edge_name in Tile.get_edges():
                if not is_city(tile.internal_edges[edge_name]):

                    print(tile.tile_type,flush = True)
                    print("is not a city",flush = True)

                    continue
                
                print(tile.tile_type,flush = True)
                print("is a city",flush = True)

                if (x, y, edge_name) in seen_edges:
                    continue

                # Begin city flood-fill from (x, y, edge_name)
                stack = [(x, y, edge_name)]
                city_tiles = set()
                is_claimed = False

                while stack:
                    cx, cy, edge = stack.pop()
                    if (cx, cy, edge) in seen_edges:
                        continue

                    seen_edges.add((cx, cy, edge))
                    current_tile = grid[cy][cx]
                    city_tiles.add((cx, cy))

                    if current_tile.internal_claims[edge] is not None:
                        is_claimed = True
                        break  # Don't include claimed cities

                    # Explore across this edge to adjacent tile
                    if edge in edge_directions:
                        dx, dy, opp_edge = edge_directions[edge]
                        nx, ny = cx + dx, cy + dy

                        if 0 <= nx < MAX_MAP_LENGTH and 0 <= ny < MAX_MAP_LENGTH:
                            neighbor = grid[ny][nx]
                            if neighbor is not None:
                                s1 = current_tile.internal_edges[edge]
                                s2 = neighbor.internal_edges[opp_edge]
                                if StructureType.is_compatible(s1, s2):
                                    stack.append((nx, ny, opp_edge))

                if not is_claimed:
                    estimated_score = 1.5 * len(city_tiles)
                    cities.append((estimated_score, grid[y][x]))  # use origin tile as anchor

    cities.sort(key=lambda score: score[0], reverse=True)
    
    return cities

def value_roads(game: Game):

    grid = game.state.map._grid
    seen_edges = set()
    roads = []

    edge_directions = {
        "top_edge": (0, -1, "bottom_edge"),
        "bottom_edge": (0, 1, "top_edge"),
        "left_edge": (-1, 0, "right_edge"),
        "right_edge": (1, 0, "left_edge"),
    }

    def is_road(structure):
        return structure in {StructureType.ROAD, StructureType.ROAD_START}

    def get_road_edges(tile):
        return [e for e in Tile.get_edges() if is_road(tile.internal_edges[e])]

    for y in range(len(grid)):
        for x in range(len(grid[0])):
            tile = grid[y][x]
            if tile is None:
                continue

            for edge_name in get_road_edges(tile):
                if (x, y, edge_name) in seen_edges:
                    continue

                stack = [(x, y, edge_name)]
                road_tiles = set()
                is_claimed = False

                while stack:
                    cx, cy, edge = stack.pop()
                    if (cx, cy, edge) in seen_edges:
                        continue

                    seen_edges.add((cx, cy, edge))
                    current_tile = grid[cy][cx]
                    road_tiles.add((cx, cy))

                    if current_tile.internal_claims[edge] is not None:
                        is_claimed = True
                        break

                    road_edges_here = get_road_edges(current_tile)

                    # Internal connection across the tile (if exactly 2 edges)
                    if len(road_edges_here) == 2:
                        for other_edge in road_edges_here:
                            if other_edge != edge and (cx, cy, other_edge) not in seen_edges:
                                stack.append((cx, cy, other_edge))

                    # External connection to neighbor
                    if edge in edge_directions:
                        dx, dy, opp_edge = edge_directions[edge]
                        nx, ny = cx + dx, cy + dy

                        if 0 <= nx < MAX_MAP_LENGTH and 0 <= ny < MAX_MAP_LENGTH:
                            neighbor = grid[ny][nx]
                            if neighbor is not None:
                                s1 = current_tile.internal_edges[edge]
                                s2 = neighbor.internal_edges[opp_edge]
                                if StructureType.is_compatible(s1, s2):
                                    stack.append((nx, ny, opp_edge))

                if not is_claimed:
                    estimated_score = float(len(road_tiles))  # 1 point per tile
                    roads.append((estimated_score, grid[y][x]))

    
    roads.sort(key=lambda score: score[0], reverse=True)

    return roads

def value_fields():
    return None

def handle_place_meeple(
    game: Game, bot_state: BotState, query: QueryPlaceMeeple
) -> MovePlaceMeeplePass | MovePlaceMeeple:
    """
    Try to place a meeple on the most recently placed tile.
    Priority order: monastery -> Anything else
    """

    grid = game.state.map._grid

    monasteries = value_monastaries(game)
    
    print("monastaries are\n",flush = True)
    print(monasteries,flush = True)
    print("\n",flush = True)


    cities = value_cities(game)

    print("cities are\n",flush = True)
    print(cities,flush = True)
    print("\n",flush = True)

    roads = value_roads(game)

    print("roads are\n",flush = True)
    print(roads,flush = True)
    print("\n",flush = True)

    #fields = value_fields(game)

    #highest = max(cities,roads,fields) # get tile with largest value

    if len(monasteries) != 0:
        monastery_highest = monasteries[0]

        print("wanna place a meeple at \n",flush = True)

        print(monastery_highest,flush= True)

        xcoord = monastery_highest[1]
        ycoord = monastery_highest[2]

        print(xcoord,flush= True)
        print(ycoord,flush = True)

        tile = grid[ycoord][xcoord]

        print(tile.tile_type,flush = True)

        return game.move_place_meeple(query, tile._to_model(), MONASTARY_IDENTIFIER)

    # if monastery_highest[0] > highest[0]:
    
    # if highest is None: 
    
    return game.move_place_meeple_pass(query)

    # tile = grid[highest[2], highest[1]]

    # return game.move_place_meeple(query, tile._to_model(), edge)



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
