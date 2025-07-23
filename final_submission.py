import random  # <-- Required for evaluate_* functions

from typing import List, Tuple, Dict

from helper.game import Game
from helper.utils import print_map

from lib.interact.tile import Tile, TileModifier
from lib.models.tile_model import TileModel
from lib.interact.structure import StructureType
from copy import deepcopy

from lib.interface.queries.typing import QueryType
from lib.interface.queries.query_place_tile import QueryPlaceTile
from lib.interface.queries.query_place_meeple import QueryPlaceMeeple

from lib.interface.events.moves.typing import MoveType
from lib.interface.events.moves.move_place_tile import MovePlaceTile
from lib.interface.events.moves.move_place_meeple import (
	MovePlaceMeeple,
	MovePlaceMeeplePass,
)

from collections import defaultdict, deque
from typing import Callable, Iterator, Protocol

from lib.config.map_config import MAX_MAP_LENGTH, MONASTARY_IDENTIFIER



class BotState:
    """A class for us to locally the state of the game and what we find relevant"""

    def __init__(self):
        self.last_tile: Tile | None = None
        self.meeples_placed: int = 0
        self.river_phase = True
        self.best_tile_placement: dict | None = None
        self.edge = None
        self.round_number = 0


def main():
    game = Game()
    bot_state = BotState()

    while True:
        query = game.get_next_query()

        def choose_move(query: QueryType) -> MoveType:
            match query:
                case QueryPlaceTile() as q:
                    return handle_place_tile(game, bot_state, q)

                case QueryPlaceMeeple() as q:
                    return handle_place_meeple(game, bot_state, q)
                case _:
                    assert False

        game.send_move(choose_move(query))


def handle_place_tile(game: Game, bot_state: BotState, query: QueryPlaceTile) -> MovePlaceTile:
    # 1. End river phase if needed
    placed_tiles_count = len(game.state.map.placed_tiles)
    if placed_tiles_count >= 12 and bot_state.river_phase:
        bot_state.river_phase = False

    # 2. Get all valid placements
    possible_placements = get_all_valid_placements(game, bot_state)
    if not possible_placements:
        raise ValueError("No valid tile placement found")

    # 3. Analyze each placement (BFS/structure analysis)
    analyzed = []
    for tile_idx, tile, x, y, rotation in possible_placements:
        analysis = analyze_tile_placement_structures(game, tile, x, y, rotation)
        points = analysis['total_potential_points']
        analyzed.append((points, tile_idx, tile, x, y, rotation, analysis))

    # 4. Sort placements by points
    analyzed.sort(reverse=True, key=lambda tup: tup[0])
    best = analyzed[0]
    _, tile_idx, tile, x, y, rotation, analysis = best

    # 5. Store meeple info for next phase
    bot_state.last_tile = tile


    claimable = analysis['claimable_structures']
    details   = analysis['structure_details']

    if bot_state.round_number < 10:
        high_value = [e for e in claimable if details[e]['points'] >= 1]
    else:
        high_value = [e for e in claimable if details[e]['points'] >= 0]

    if high_value:
        # pick the edge with the highest points
        best_edge = max(high_value, key=lambda e: details[e]['points'])
        best_value = details[best_edge]['points']
    else:
        best_edge = None
        best_value = 0

    bot_state.edge = best_edge

    # After picking best_edge
    if best_value < 3 and bot_state.meeples_placed >= 6:
        bot_state.edge = None  # don't place meeple for tiny value late game

    # 6. Place the tile
    original_tile = game.state.my_tiles[tile_idx]
    while original_tile.rotation != 0:
        original_tile.rotate_clockwise(1)
    original_tile.rotate_clockwise(rotation)
    original_tile.placed_pos = (x, y)


    bot_state.round_number += 1

    return game.move_place_tile(query, original_tile._to_model(), tile_idx)


def handle_place_meeple(
    game: Game, bot_state: BotState, query: QueryPlaceMeeple
) -> MovePlaceMeeplePass | MovePlaceMeeple:

    
    recent_tile = bot_state.last_tile
    edge = bot_state.edge

    # update number of meeples

    bot_state.meeples_placed = 7 - game.state.me.num_meeples
    

    if not recent_tile:
        return game.move_place_meeple_pass(query)

    if bot_state.meeples_placed == 7:
        return game.move_place_meeple_pass(query)

    if TileModifier.MONASTARY in recent_tile.modifiers:
        bot_state.meeples_placed += 1

        return game.move_place_meeple(query, recent_tile._to_model(), MONASTARY_IDENTIFIER)

    if not edge:
        return game.move_place_meeple_pass(query)

    bot_state.meeples_placed += 1

    return game.move_place_meeple(query, recent_tile._to_model(), edge)
    

def get_all_valid_placements(game: Game, bot_state: BotState = None) -> List[Tuple[int, object, int, int, int]]:

    # Check if we're in river phase
    if bot_state and bot_state.river_phase:
        return get_valid_river_placement(game, bot_state)
    
    valid_placements = []

    # Define the 4 directions: right, down, left, up
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

    # For each available tile
    for tile_index, current_tile in enumerate(game.state.my_tiles):
        
        # Reset cache for each new tile in hand
        checked_positions = set()  # Cache to avoid checking same position multiple times for this tile
        
        # Keep original tile rotation at 0, use separate variable for test rotation
        # Reset tile to rotation 0 if needed
        while current_tile.rotation != 0:
            current_tile.rotate_clockwise(1)

        # For each placed tile on the board
        for placed_tile in game.state.map.placed_tiles:
            # Get the position of this placed tile
            x, y = placed_tile.placed_pos

            # Check each of the 4 adjacent positions around this tile
            for dx, dy in directions:
                # Calculate the adjacent position coordinates
                adj_x, adj_y = x + dx, y + dy

                # Skip if this position has already been checked for this tile
                position_key = (adj_x, adj_y)
                if position_key in checked_positions:
                    continue
                
                # Mark this position as checked for this tile
                checked_positions.add(position_key)

                # Check if this adjacent position is valid and empty
                # Valid: within board boundaries (0 to 169)
                # Empty: no tile already placed there
                if (0 <= adj_x < 170 and
                    0 <= adj_y < 170 and
                    game.state.map._grid[adj_y][adj_x] is None):

                    # For each rotation of the tile
                    for test_rotation in range(4):
                        # Create a temporary copy of the tile for testing this rotation
                        test_tile = deepcopy(current_tile)
                        
                        # Rotate the test tile to the target rotation
                        test_tile.rotate_clockwise(test_rotation)

                        # Check if this tile can be placed here
                        if local_can_place_tile_at(game, test_tile, adj_x, adj_y, bot_state):
                            placement = (tile_index, current_tile, adj_x, adj_y, test_rotation)
                            valid_placements.append(placement)

    return valid_placements


def local_can_place_tile_at(game: Game, tile: Tile, x: int, y: int, bot_state: BotState = None) -> bool:


    directions = {
        (0, -1): "top_edge",
        (1, 0): "right_edge", 
        (0, 1): "bottom_edge",
        (-1, 0): "left_edge",
    }

    edge_opposite = {
        "top_edge": "bottom_edge",
        "bottom_edge": "top_edge",
        "left_edge": "right_edge",
        "right_edge": "left_edge",
    }

    
    has_any_neighbour = False
    
    # Check each direction for neighbours
    for (dx, dy), edge in directions.items():
        nx, ny = x + dx, y + dy
        
        # Check boundaries
        if not (0 <= ny < len(game.state.map._grid) and 0 <= nx < len(game.state.map._grid[0])):
            continue
        
        neighbour_tile = game.state.map._grid[ny][nx]
        
        if neighbour_tile is None:
            continue
        
        has_any_neighbour = True
        
        # Check structure compatibility
        tile_edge_structure = tile.internal_edges[edge]
        neighbour_edge_structure = neighbour_tile.internal_edges[edge_opposite[edge]]
        
        # Use engine's compatibility check for more robust validation
        try:
            is_compatible = StructureType.is_compatible(tile_edge_structure, neighbour_edge_structure)
            
            if not is_compatible:
                return False
        except Exception as e:
            return False

    if has_any_neighbour:
        return True
    else:
        return False

def get_valid_river_placement(game: Game, bot_state: BotState) -> List[Tuple[int, object, int, int, int]]:

    valid_placements = []
    
    if not game.state.map.placed_tiles:
        return valid_placements
    
    # Get the latest placed tile
    latest_tile = game.state.map.placed_tiles[-1]
    latest_x, latest_y = latest_tile.placed_pos
    
    # Define the 4 directions: right, down, left, up
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    direction_names = ["DOWN", "RIGHT", "UP", "LEFT"]
    
    # Check each edge of the latest tile for river connections
    for i, (dx, dy) in enumerate(directions):
        nx, ny = latest_x + dx, latest_y + dy
        
        # Check if this position is empty and within bounds
        if (0 <= nx < 170 and 
            0 <= ny < 170 and 
            game.state.map._grid[ny][nx] is None):
            
            # Determine which edge of the latest tile faces this direction
            if dx == 1:  # RIGHT
                edge = "right_edge"
            elif dx == -1:  # LEFT
                edge = "left_edge"
            elif dy == 1:  # DOWN
                edge = "bottom_edge"
            else:  # UP
                edge = "top_edge"
            
            # Check if the latest tile has a river edge facing this direction
            if latest_tile.internal_edges[edge] == StructureType.RIVER:
                
                # This is a potential river head position
                # Check if any of our tiles can be placed here
                for tile_index, current_tile in enumerate(game.state.my_tiles):
                    # Reset tile to rotation 0
                    while current_tile.rotation != 0:
                        current_tile.rotate_clockwise(1)
                    
                    # Test all rotations
                    for test_rotation in range(4):
                        test_tile = deepcopy(current_tile)
                        test_tile.rotate_clockwise(test_rotation)
                        
                        # Check if this tile can be placed at the river head
                        if river_can_place_tile_at(game, test_tile, latest_tile, nx, ny, bot_state):
                            placement = (tile_index, current_tile, nx, ny, test_rotation)
                            valid_placements.append(placement)
                
                # If we found valid placements for this river head, we can stop checking other directions
                if valid_placements:
                    break
    
    return valid_placements




def local_structure_compatible(structure1: StructureType, structure2: StructureType) -> bool:

    if structure1 == StructureType.RIVER:
        return structure2 == StructureType.RIVER
    elif structure1 == StructureType.CITY:
        return structure2 == StructureType.CITY
    elif structure1 == StructureType.ROAD:
        return structure2 in [StructureType.ROAD, StructureType.ROAD_START]
    elif structure1 == StructureType.ROAD_START:
        return structure2 in [StructureType.ROAD, StructureType.ROAD_START]
    elif structure1 == StructureType.GRASS:
        return structure2 == StructureType.GRASS
    else:
        return False

def river_can_place_tile_at(game: Game, test_tile: Tile, latest_tile: Tile, x: int, y: int, bot_state: BotState = None) -> bool:

    # Use engine's river validation for robust checking
    try:
        river_result = game.state.map.river_validation(test_tile, x, y)
        
        if river_result == "pass":
            return True
        elif river_result == "uturn":
            return False
        elif river_result == "disjoint":
            return False
        else:
            return False
    except Exception as e:
        return False


def evaluate_monastary(grid, tile: Tile):
    
    position = tile.placed_pos
    
    x = position[0]
    y = position[1]


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

    if score == 8:
        #11 because we have to account for value from taking back the meeple
        score = 11

    return score


# taken from engine    
def traverse_connected_component(
    grid,
    start_tile: "Tile",
    edge: str,
    visited=None,
    yield_cond: Callable[[Tile, str], bool] = lambda _1, _2: True,
    modify: Callable[[Tile, str], None] = lambda _1, _2: None,
) -> Iterator[tuple["Tile", str]]:
    if visited is None:
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
                ce, tile.placed_pos, grid
            )

            if external_tile is None:
                continue

            if not StructureType.is_compatible(
                structure_type, external_tile.internal_edges[ce_neighbour]
            ):
                continue

            if (external_tile, ce_neighbour) not in visited:
                queue.append((external_tile, ce_neighbour))



def analyze_tile_placement_structures(
    game: Game, 
    tile: Tile, 
    x: int, 
    y: int,
    rotation: int
) -> dict:

    
    # Create a temporary copy of the tile for simulation
    test_tile = deepcopy(tile)
    
    # Set the tile to the target rotation
    while test_tile.rotation != 0:
        test_tile.rotate_clockwise(1)
    test_tile.rotate_clockwise(rotation)
    
    # Simulate placing the tile at the position
    test_tile.placed_pos = (x, y)
    
    # Create a deep copy of the grid for safe analysis
    grid_copy = []
    for row in game.state.map._grid:
        grid_copy.append(row.copy())  # Copy each row
    
    # Temporarily place the tile in the copied grid for analysis
    grid_copy[y][x] = test_tile
    
    claimable_structures = []
    total_potential_points = 0
    structure_details = {}
    # Use a shared visited set to avoid double-counting
    visited = set()
    
    # Handle monastery as a special case (not per edge)
    if TileModifier.MONASTARY in test_tile.modifiers:
        points = evaluate_monastary(grid_copy, test_tile)
        meeple = test_tile.internal_claims.get(MONASTARY_IDENTIFIER)
        is_claimed = meeple is not None
        structure_details[MONASTARY_IDENTIFIER] = {
            'type': 'MONASTARY',
            'points': points,
            'claimed': is_claimed
        }
        total_potential_points += points
        claimable_structures.append(MONASTARY_IDENTIFIER)
        
    # Now check all edges for other structures (not monastery)
    for edge in ["top_edge", "right_edge", "bottom_edge", "left_edge"]:
        if (test_tile, edge) in visited:
            continue
        structure_type = test_tile.internal_edges[edge]
        # Skip grass and river edges (not claimable)
        if structure_type in [StructureType.GRASS, StructureType.RIVER]:
            structure_details[edge] = {
                'type': structure_type,
                'points': 0,
                'claimed': False
            }
            continue

        # Use shared visited set
        connected_components = list(traverse_connected_component(grid_copy, test_tile, edge, visited=visited))
        
        # find amount of opponents who owns this structure and whether we own it too
        our_player_id = game.state.me.player_id
        our_meeples = 0
        opponent_player_ids = set()
        
        for t, e in connected_components:
            meeple = t.internal_claims[e]
            if meeple is not None:
                if meeple.player_id == our_player_id:
                    our_meeples = 1
                else:
                    opponent_player_ids.add(meeple.player_id)
                    
                    
        num_opponents_meeples = len(opponent_player_ids)
        
        # set claim boolean
        if our_meeples + num_opponents_meeples >0:
            is_claimed = True
        else:
            is_claimed = False
            
            
        # count number of tiles in structure
        unique_tiles = {tile for tile, edge in connected_components}
        points = 0
        if structure_type in [StructureType.CITY]:
            points = 2 * len(unique_tiles)
            for tile in unique_tiles:
                if TileModifier.EMBLEM in tile.modifiers:
                    points += 2
        else:
            points = len(unique_tiles)
            
        # adjust points by opponent vs our meeples
        adjust_points = (our_meeples - 0.34* num_opponents_meeples) # 0.34 is correct
        if our_meeples == 0 and num_opponents_meeples == 0:
            adjust_points = 1.1 # need tweaking

            
            
        points = adjust_points * points
        
        # find number of open edges
        open_edges = 0
        for tile_in_component, edge_in_component in connected_components:
            external_tile = Tile.get_external_tile(edge_in_component, tile_in_component.placed_pos, grid_copy)
            if external_tile is None:
                open_edges += 1
        
        
        completion_factor = 1.0
        highest_score = max(player.points for player in game.state.players.values())
        if highest_score > 30:
            phase_1 = False
            phase_2 = True
        else:
            phase_1 = True
            phase_2 = False
        if phase_1:
            if structure_type == StructureType.CITY:
                completion_factor = 1.0
            elif structure_type == StructureType.ROAD and structure_type == StructureType.ROAD_START:
                if open_edges == 0:
                    completion_factor = 1.0
                elif open_edges == 1:
                    completion_factor = 0.9
                else:
                    completion_factor = 0.8
        elif phase_2:
            if structure_type == StructureType.CITY:
                if open_edges == 0:
                    completion_factor = 1.0
                elif open_edges == 1:
                    completion_factor = 0.8
                elif open_edges == 2:
                    completion_factor = 0.6
                elif open_edges == 3:
                    completion_factor = 0.3
                else:
                    completion_factor = 0.3
            elif structure_type == StructureType.ROAD and structure_type == StructureType.ROAD_START:
                if open_edges == 0:
                    completion_factor = 1.0
                elif open_edges == 1:
                    completion_factor = 0.7
                else:
                    completion_factor = 0.4
                    
                    
        points = points * completion_factor
        if not is_claimed:
            claimable_structures.append(edge)
            
            
            
        total_potential_points += points
        structure_details[edge] = {
            'type': structure_type,
            'points': points,
            'claimed': is_claimed
        }
    return {
        'claimable_structures': claimable_structures,
        'total_potential_points': total_potential_points,
        'structure_details': structure_details
    }


def count_surrounding_monasteries(game, x, y):

    my_player_id = game.state.me.player_id
    grid = game.state.map._grid
    my_monasteries = 0
    opponent_monasteries = 0

    # Offsets for the 8 surrounding tiles
    offsets = [(-1, -1), (0, -1), (1, -1),
               (-1,  0),          (1,  0),
               (-1,  1), (0,  1), (1,  1)]

    for dx, dy in offsets:
        nx, ny = x + dx, y + dy
        if 0 <= nx < len(grid[0]) and 0 <= ny < len(grid):
            tile = grid[ny][nx]
            if tile is not None and TileModifier.MONASTARY in tile.modifiers:
                # Monastery is always claimed on the MONASTARY_IDENTIFIER
                meeple = tile.internal_claims.get(MONASTARY_IDENTIFIER)
                if meeple is not None:
                    if meeple.player_id == my_player_id:
                        my_monasteries += 1
                    else:
                        opponent_monasteries += 1

    return my_monasteries, opponent_monasteries





if __name__ == "__main__":
    main()