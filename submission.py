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
        # Non-river phase strategy - FIXED VERSION
        best_score = float("-inf")
        best_move = None
        best_tile_index = None
        best_tile = None

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
                if (0 <= nx < MAX_MAP_LENGTH and 0 <= ny < MAX_MAP_LENGTH and 
                    grid[ny][nx] is not None):
                    adjacent_tile = grid[ny][nx]
                    if tile.internal_edges[my_edge] == adjacent_tile.internal_edges[their_edge]:
                        if tile.internal_edges[my_edge] == StructureType.CITY:
                            score += 10
                        elif tile.internal_edges[my_edge] == StructureType.ROAD:
                            score += 5
            
            # Count city and road segments on the tile
            city_count = sum(1 for edge in tile.internal_edges.values() 
                           if edge == StructureType.CITY)
            road_count = sum(1 for edge in tile.internal_edges.values() 
                           if edge == StructureType.ROAD)
            
            score += city_count * 2
            score += road_count
            
            return score

        # Try each tile in hand
        for tile_hand_index, tile_in_hand in enumerate(game.state.my_tiles):
            original_rotation = tile_in_hand.rotation
            
            # Try different rotations
            for rotation in range(4):
                # Check positions adjacent to existing tiles
                for placed_tile in game.state.map.placed_tiles:
                    px, py = placed_tile.placed_pos
                    
                    # Check all adjacent positions
                    for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                        x, y = px + dx, py + dy
                        
                        # Check bounds and if position is empty
                        if (not (0 <= x < MAX_MAP_LENGTH and 0 <= y < MAX_MAP_LENGTH) or 
                            grid[y][x] is not None):
                            continue
                        
                        # CRITICAL: Check if this tile placement is actually allowed
                        if not game.can_place_tile_at(tile_in_hand, x, y):
                            continue
                            
                        # Additional validation: make sure tile edges match adjacent tiles
                        placement_valid = True
                        adjacency_map = {
                            "top_edge": (0, -1, "bottom_edge"),
                            "right_edge": (1, 0, "left_edge"), 
                            "bottom_edge": (0, 1, "top_edge"),
                            "left_edge": (-1, 0, "right_edge"),
                        }
                        
                        for my_edge, (dx, dy, their_edge) in adjacency_map.items():
                            nx, ny = x + dx, y + dy
                            if (0 <= nx < MAX_MAP_LENGTH and 0 <= ny < MAX_MAP_LENGTH and 
                                grid[ny][nx] is not None):
                                adjacent_tile = grid[ny][nx]
                                if (tile_in_hand.internal_edges[my_edge] != 
                                    adjacent_tile.internal_edges[their_edge]):
                                    placement_valid = False
                                    break
                        
                        if not placement_valid:
                            continue
                            
                        score = evaluate_placement(game, tile_in_hand, x, y)
                        
                        if score > best_score:
                            best_score = score
                            best_move = (x, y)
                            best_tile_index = tile_hand_index
                            best_tile = tile_in_hand
                
                # Rotate for next iteration
                tile_in_hand.rotate_clockwise(1)
            
            # Reset rotation to original
            tile_in_hand.rotation = original_rotation

        # Use the best move found
        if best_move and best_tile is not None:
            x, y = best_move
            
            bot_state.last_tile = best_tile
            bot_state.last_tile.placed_pos = (x, y)
            
            print(f"Placing tile at ({x}, {y}) with score {best_score}")
            return game.move_place_tile(query, best_tile._to_model(), best_tile_index)
        
        else:
            # Fallback: find any valid placement
            print("No optimal move found, using fallback strategy")
            for tile_hand_index, tile_in_hand in enumerate(game.state.my_tiles):
                original_rotation = tile_in_hand.rotation
                
                for rotation in range(4):
                    # Check all positions on the map
                    for y in range(MAX_MAP_LENGTH):
                        for x in range(MAX_MAP_LENGTH):
                            if (grid[y][x] is None and 
                                game.can_place_tile_at(tile_in_hand, x, y)):
                                
                                bot_state.last_tile = tile_in_hand
                                bot_state.last_tile.placed_pos = (x, y)
                                
                                print(f"Fallback: placing tile at ({x}, {y})")
                                return game.move_place_tile(
                                    query, tile_in_hand._to_model(), tile_hand_index
                                )
                    
                    tile_in_hand.rotate_clockwise(1)
                
                # Reset rotation
                tile_in_hand.rotation = original_rotation
            
            # Raise exception because its fucked
            raise Exception("No valid tile placement found!")
 

def value_monastaries(game: Game):
    
    monasteries = []
    
    grid = game.state.map._grid

    for y in range(len(grid)):
        for x in range(len(grid[0])):
            tile = grid[y][x]


            if tile is not None: 
                    

                    # check if tile is a monastery tile
                    if ((tile.tile_id == "A" or tile.tile_id == "B") and tile.internal_claims[MONASTARY_IDENTIFIER] is not None):
                        
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

    # Mapping of direction: (dx, dy, opposite_edge)
    edge_directions = {
        "top_edge": (0, -1, "bottom_edge"),
        "bottom_edge": (0, 1, "top_edge"),
        "left_edge": (-1, 0, "right_edge"),
        "right_edge": (1, 0, "left_edge"),
    }

    for y in range(len(grid)):
        for x in range(len(grid[0])):
            tile = grid[y][x]
            if tile is None:
                continue

            for edge_name in Tile.get_edges():
                if tile.internal_edges[edge_name] != StructureType.CITY:
                    continue

                if (x, y, edge_name) in seen_edges:
                    continue

                # Begin DFS to find all connected city edges
                stack = [(tile, edge_name)]
                city_edges = set()
                city_tiles = set()
                is_claimed = False

                while stack:
                    current_tile, current_edge = stack.pop()
                    cx, cy = current_tile.placed_pos

                    if (cx, cy, current_edge) in seen_edges:
                        continue

                    seen_edges.add((cx, cy, current_edge))
                    city_edges.add((cx, cy, current_edge))
                    city_tiles.add(current_tile)

                    # Skip if any edge has a meeple
                    if current_tile.internal_claims[current_edge] is not None:
                        is_claimed = True
                        break  # no need to continue traversing

                    # Check neighbor in the direction of the edge
                    dx, dy, opposite_edge = edge_directions[current_edge]
                    nx, ny = cx + dx, cy + dy

                    if not (0 <= nx < MAX_MAP_LENGTH and 0 <= ny < MAX_MAP_LENGTH):
                        continue

                    neighbor_tile = grid[ny][nx]
                    if neighbor_tile is None:
                        continue

                    if neighbor_tile.internal_edges[opposite_edge] == StructureType.CITY:
                        stack.append((neighbor_tile, opposite_edge))

                if not is_claimed:
                    estimated_score = 1.5 * len(city_tiles)
                    cities.append((estimated_score, tile))

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

    def is_road_edge(edge_type):
        return edge_type in {StructureType.ROAD, StructureType.ROAD_START}

    def get_road_edges(tile):
        return [edge for edge in Tile.get_edges() if is_road_edge(tile.internal_edges[edge])]

    for y in range(len(grid)):
        for x in range(len(grid[0])):
            tile = grid[y][x]
            if tile is None:
                continue

            road_edges = get_road_edges(tile)
            num_roads = len(road_edges)

            for edge_name in road_edges:
                if (x, y, edge_name) in seen_edges:
                    continue

                # Start a new road segment
                stack = [(tile, edge_name)]
                road_edges_in_segment = set()
                road_tiles = set()
                is_claimed = False

                while stack:
                    current_tile, current_edge = stack.pop()
                    cx, cy = current_tile.placed_pos

                    if (cx, cy, current_edge) in seen_edges:
                        continue

                    seen_edges.add((cx, cy, current_edge))
                    road_edges_in_segment.add((cx, cy, current_edge))
                    road_tiles.add(current_tile)

                    if current_tile.internal_claims[current_edge] is not None:
                        is_claimed = True
                        break

                    connected_edges = get_road_edges(current_tile)

                    # If tile has exactly 2 road edges, treat them as internally connected
                    if len(connected_edges) == 2:
                        for other_edge in connected_edges:
                            if other_edge != current_edge and (cx, cy, other_edge) not in seen_edges:
                                stack.append((current_tile, other_edge))

                    # Check external connection
                    if current_edge in edge_directions:
                        dx, dy, opp_edge = edge_directions[current_edge]
                        nx, ny = cx + dx, cy + dy

                        if 0 <= nx < MAX_MAP_LENGTH and 0 <= ny < MAX_MAP_LENGTH:
                            neighbor_tile = grid[ny][nx]
                            if (
                                neighbor_tile is not None and
                                is_road_edge(neighbor_tile.internal_edges[opp_edge])
                            ):
                                stack.append((neighbor_tile, opp_edge))

                if not is_claimed:
                    estimated_score = float(len(road_tiles))  # 1 point per tile
                    roads.append((estimated_score, tile))

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

    print("monasteries are\n",flush = True)
    print(monasteries,flush = True)
    print("\n",flush = True)

    #cities = value_cities(game)



    #roads = value_roads(game)
    #fields = value_fields(game)

    #highest = max(cities,roads,fields) # get tile with largest value

    # Get the monastery with the highest score (most surrounded)
    # monastery_highest = monasteries[0]

    # If the best monastery is better than the highest-valued other structure, prioritize it
    # if monastery_highest[0] > highest[0]:
    #     # Retrieve the tile object for the best monastery
    #     tile = grid[monastery_highest[2], monastery_highest[1]]

    #     # Place a meeple on the monastery
    #     return game.move_place_meeple(query, tile._to_model(), MONASTARY_IDENTIFIER)
    
    # Otherwise, try to place a meeple on the highest-valued structure found
    # if highest is None: 
    #     # If there is no valid structure, pass the turn
    #     return game.move_place_meeple_pass(query)

    # Retrieve the tile object for the highest-valued structure
    # tile = grid[highest[2], highest[1]]

    # Place a meeple on the appropriate edge/structure
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
