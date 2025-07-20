import random  # <-- Required for evaluate_* functions

from typing import List, Tuple, Dict

from helper.game import Game
from helper.utils import print_map

from lib.interact.tile import Tile
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

from lib.config.map_config import MAX_MAP_LENGTH, MONASTARY_IDENTIFIER



class BotState:
    """A class for us to locally the state of the game and what we find relevant"""

    def __init__(self):
        self.last_tile: Tile | None = None
        self.meeples_placed: int = 0
        self.river_phase = True
        self.best_tile_placement: dict | None = None


def main():
    game = Game()
    bot_state = BotState()

    while True:
        query = game.get_next_query()

        def choose_move(query: QueryType) -> MoveType:
            match query:
                case QueryPlaceTile() as q:
                    print("Running handle_place_tile")
                    return handle_place_tile(game, bot_state, q)

                case QueryPlaceMeeple() as q:
                    print("Running handle_place_meeple")
                    return handle_place_meeple(game, bot_state, q)
                case _:
                    assert False

        print("Player's move")
        game.send_move(choose_move(query))


def handle_place_tile(
    game: Game, bot_state: BotState, query: QueryPlaceTile
) -> MovePlaceTile:
    """
    Strategic tile placement considering:
    1. Completing our own structures for maximum points
    2. Blocking opponent structures when beneficial
    3. Creating opportunities for future placements
    4. Maintaining board connectivity
    """

    # Check tile count and end river phase if needed
    placed_tiles_count = len(game.state.map.placed_tiles)
    print(f"📊 [TILE] Total placed tiles: {placed_tiles_count}")
    
    if placed_tiles_count >= 12 and bot_state and bot_state.river_phase:
        print(f"🏁 [RIVER] 12+ tiles placed - ending river phase")
        bot_state.river_phase = False

    # Print current map layout for debugging
    print(f"📊 [TILE] Placed tiles: {len(game.state.map.placed_tiles)}", flush=True)
    print(f"📊 [TILE] My tiles: {[t.tile_type for t in game.state.my_tiles]}", flush=True)

    # Try to find the best placement for maximum points first
    print(f"🎯 [TILE] Looking for best point-maximizing placement...")
    best_point_placement = find_best_tile_placement_for_points(game, bot_state)
    
    if best_point_placement and best_point_placement[5] > 0:  # If we found a placement with positive points
        tile_idx, tile, x, y, rotation, points = best_point_placement
        print(f"🏆 [TILE] Found high-point placement: {points} points")
        
        # Get the original tile from our hand and set its rotation and position
        original_tile = game.state.my_tiles[tile_idx]
        
        # Reset to rotation 0 first, then rotate to target rotation
        while original_tile.rotation != 0:
            original_tile.rotate_clockwise(1)
        
        # Rotate to the target rotation
        original_tile.rotate_clockwise(rotation)
        original_tile.placed_pos = (x, y)
        
        # Remember which tile we placed and where for meeple placement
        bot_state.last_tile = original_tile

        print(
            f"AI Carcassonne placing tile {original_tile.tile_type} at ({x}, {y}) with rotation {rotation} for {points} points", flush=True
        )

        # Tell the game engine to place this tile
        return game.move_place_tile(query, original_tile._to_model(), tile_idx)

    # Fallback to regular evaluation if no high-point placement found
    print(f"📊 [TILE] No high-point placement found, using regular evaluation...")
    
    # Get all possible valid placements for our tiles
    possible_placements = get_all_valid_placements(game, bot_state)

    # Check if we found any valid placements
    if not possible_placements:
        print("❌ No valid placements found!", flush=True)
        print("Using brute force fallback...", flush=True)
        return brute_force_tile(game, bot_state, query)

    print(f"✅ Found {len(possible_placements)} valid placements", flush=True)

    # Score each placement option based on our strategy
    scored_placements = []
    for tile_idx, tile, x, y, rotation in possible_placements:
        # Calculate how good this placement is
        score = evaluate_tile_placement(game, bot_state, tile, x, y, rotation)
        scored_placements.append((score, tile_idx, tile, x, y, rotation))

    # Sort by score (highest first) and pick the best placement
    scored_placements.sort(reverse=True)
    best_score, best_tile_idx, best_tile, best_x, best_y, best_rotation = (
        scored_placements[0]
    )

    # Get the original tile from our hand and set its rotation and position
    original_tile = game.state.my_tiles[best_tile_idx]
    
    # Reset to rotation 0 first, then rotate to target rotation
    while original_tile.rotation != 0:
        original_tile.rotate_clockwise(1)
    
    # Rotate to the target rotation
    original_tile.rotate_clockwise(best_rotation)
    original_tile.placed_pos = (best_x, best_y)
    
    # Remember which tile we placed and where for meeple placement
    bot_state.last_tile = original_tile

    print(
        f"AI Carcassonne placing tile {original_tile.tile_type} at ({best_x}, {best_y}) with rotation {best_rotation}, score: {best_score}", flush=True
    )

    # Tell the game engine to place this tile
    return game.move_place_tile(query, original_tile._to_model(), best_tile_idx)

def brute_force_tile(
    game: Game, bot_state: BotState, query: QueryPlaceTile
) -> MovePlaceTile:
    """Fallback strategy: brute force search for any valid placement"""
    grid = game.state.map._grid
    height = len(grid)
    width = len(grid[0]) if height > 0 else 0

    directions = {
        (0, 1): "top",
        (1, 0): "right",
        (0, -1): "bottom",
        (-1, 0): "left",
    }

    print("🔍 BRUTE FORCE: Searching for any valid placement...", flush=True)
    print(f"Grid size: {width}x{height}", flush=True)

    for y in range(height):
        for x in range(width):
            if grid[y][x] is not None:
                print(f"🔍 Checking near tile at ({x}, {y}): {grid[y][x].tile_type}", flush=True)
                for tile_index, tile in enumerate(game.state.my_tiles):
                    for direction in directions:
                        dx, dy = direction
                        x1, y1 = (x + dx, y + dy)

                        if local_can_place_tile_at(game, tile, x1, y1, bot_state):
                            print(f"✅ BRUTE FORCE: Found valid placement at ({x1}, {y1})", flush=True)
                            bot_state.last_tile = tile
                            bot_state.last_tile.placed_pos = (x1, y1)
                            return game.move_place_tile(
                                query, tile._to_model(), tile_index
                            )

    print("❌ BRUTE FORCE: No valid placement found!", flush=True)
    raise ValueError("No valid tile placement found")


def handle_place_meeple(
    game: Game, bot_state: BotState, query: QueryPlaceMeeple
) -> MovePlaceMeeplePass | MovePlaceMeeple:
    """
    Place meeples strategically to maximize points.
    Uses the same point-maximization logic as tile placement.
    """
    print(f"🎯 [MEEPLE] Starting meeple placement", flush=True)
    
    recent_tile = bot_state.last_tile
    if not recent_tile:
        print(f"❌ [MEEPLE] No recent tile - passing", flush=True)
        return game.move_place_meeple_pass(query)

    print(f"🎯 [MEEPLE] Recent tile: {recent_tile.tile_type} at {recent_tile.placed_pos}", flush=True)
    print(f"🎯 [MEEPLE] Meeples placed: {bot_state.meeples_placed}/7", flush=True)

    if bot_state.meeples_placed == 7:
        print(f"❌ [MEEPLE] No meeples left - passing", flush=True)
        return game.move_place_meeple_pass(query)

    # Find all unclaimed structures
    available_structures = []
    print(f"🎯 [MEEPLE] Checking edges for available structures...", flush=True)
    
    # Check edges
    for edge in ["top_edge", "right_edge", "bottom_edge", "left_edge"]:
        print(f"🎯 [MEEPLE] Checking {edge}:", flush=True)
        print(f"  - Internal claims: {recent_tile.internal_claims.get(edge)}", flush=True)
        print(f"  - Edge type: {recent_tile.internal_edges[edge]}", flush=True)
        
        if (recent_tile.internal_claims.get(edge) is None
            and recent_tile.internal_edges[edge] != StructureType.RIVER
            and recent_tile.internal_edges[edge] != StructureType.GRASS):
            print(f"  ✅ {edge} is unclaimed and valid structure type", flush=True)
            
            # Be more conservative - only place on truly isolated structures
            # Check if this edge connects to any existing structures
            structure_type = recent_tile.internal_edges[edge]
            print(f"  - Structure type: {structure_type}", flush=True)
            
            # Check if this edge faces any neighbor (ADDITIONAL CONDITION)
            edge_faces_neighbor = False
            tile_x, tile_y = recent_tile.placed_pos
            
            # Check the direction this edge faces
            if edge == "top_edge":
                check_x, check_y = tile_x, tile_y - 1
            elif edge == "right_edge":
                check_x, check_y = tile_x + 1, tile_y
            elif edge == "bottom_edge":
                check_x, check_y = tile_x, tile_y + 1
            elif edge == "left_edge":
                check_x, check_y = tile_x - 1, tile_y
            
            # Check if there's a neighbor in that direction
            if (0 <= check_x < 170 and 0 <= check_y < 170 and 
                game.state.map._grid[check_y][check_x] is not None):
                print(f"  ❌ {edge} faces a neighbor at ({check_x}, {check_y})", flush=True)
                edge_faces_neighbor = True
            else:
                print(f"  ✅ {edge} doesn't face any neighbor", flush=True)
            
            # For roads and cities, be extra careful about connections
            if structure_type in [StructureType.ROAD, StructureType.CITY]:
                print(f"  ⚠️  {edge} is {structure_type} - checking for connections", flush=True)
                
                # Check if this structure connects to any existing structures
                # This is a simplified check - we'll be conservative
                connected_to_existing = False
                
                # Check all adjacent tiles to see if they have the same structure type
                for dx, dy in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
                    nx, ny = recent_tile.placed_pos[0] + dx, recent_tile.placed_pos[1] + dy
                    
                    if (0 <= nx < 170 and 0 <= ny < 170 and 
                        game.state.map._grid[ny][nx] is not None):
                        
                        neighbor = game.state.map._grid[ny][nx]
                        # Check if neighbor has the same structure type on the connecting edge
                        if dx == 0 and dy == -1:  # top neighbor
                            if neighbor.internal_edges["bottom_edge"] == structure_type:
                                connected_to_existing = True
                                print(f"    ❌ Connected to top neighbor {neighbor.tile_type}", flush=True)
                        elif dx == 1 and dy == 0:  # right neighbor
                            if neighbor.internal_edges["left_edge"] == structure_type:
                                connected_to_existing = True
                                print(f"    ❌ Connected to right neighbor {neighbor.tile_type}", flush=True)
                        elif dx == 0 and dy == 1:  # bottom neighbor
                            if neighbor.internal_edges["top_edge"] == structure_type:
                                connected_to_existing = True
                                print(f"    ❌ Connected to bottom neighbor {neighbor.tile_type}", flush=True)
                        elif dx == -1 and dy == 0:  # left neighbor
                            if neighbor.internal_edges["right_edge"] == structure_type:
                                connected_to_existing = True
                                print(f"    ❌ Connected to left neighbor {neighbor.tile_type}", flush=True)
                
                if connected_to_existing:
                    print(f"  ❌ {edge} connects to existing structure - skipping", flush=True)
                    continue
                else:
                    print(f"  ✅ {edge} is isolated - safe to place", flush=True)
            
            # For other structure types, use the original check
            existing_claims = game.state._get_claims(recent_tile, edge)
            print(f"  - Existing claims: {existing_claims}", flush=True)
            
            # FINAL CHECK: Only add if no existing claims AND doesn't face neighbor
            if not existing_claims and not edge_faces_neighbor:
                print(f"  ✅ {edge} has no existing claims and doesn't face neighbor - adding to available", flush=True)
                available_structures.append(edge)
            else:
                if existing_claims:
                    print(f"  ❌ {edge} has existing claims - skipping", flush=True)
                if edge_faces_neighbor:
                    print(f"  ❌ {edge} faces neighbor - skipping", flush=True)
        else:
            print(f"  ❌ {edge} is not available (claimed/river/grass)", flush=True)
    
    print(f"🎯 [MEEPLE] Available structures: {available_structures}", flush=True)
    
    if not available_structures:
        print(f"❌ [MEEPLE] No available structures - passing", flush=True)
        return game.move_place_meeple_pass(query)
    
    # Evaluate each available structure for point potential
    best_structure = None
    best_points = -999999.0
    
    print(f"🎯 [MEEPLE] Evaluating structures for maximum points...")
    
    for edge in available_structures:
        # Simulate placing a meeple on this edge
        structure_type = recent_tile.internal_edges[edge]
        
        # Calculate potential points for this structure
        if structure_type == StructureType.CITY:
            points = 2  # 2 points per tile for cities
        elif structure_type == StructureType.ROAD:
            points = 1  # 1 point per tile for roads
        elif structure_type == StructureType.ROAD_START:
            points = 1  # 1 point per tile for roads
        else:
            points = 0
        
        # Check if this structure is completed or uncompleted
        # For now, we'll use a simple heuristic - if it's isolated, it's likely uncompleted
        # In a full implementation, you'd check the actual completion status
        
        # If it's isolated (no connections), it's likely uncompleted
        is_isolated = True
        tile_x, tile_y = recent_tile.placed_pos
        
        # Check if this edge connects to any existing structures
        for dx, dy in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
            nx, ny = tile_x + dx, tile_y + dy
            
            if (0 <= nx < 170 and 0 <= ny < 170 and 
                game.state.map._grid[ny][nx] is not None):
                
                neighbor = game.state.map._grid[ny][nx]
                # Check if neighbor has the same structure type on the connecting edge
                if dx == 0 and dy == -1:  # top neighbor
                    if neighbor.internal_edges["bottom_edge"] == structure_type:
                        is_isolated = False
                elif dx == 1 and dy == 0:  # right neighbor
                    if neighbor.internal_edges["left_edge"] == structure_type:
                        is_isolated = False
                elif dx == 0 and dy == 1:  # bottom neighbor
                    if neighbor.internal_edges["top_edge"] == structure_type:
                        is_isolated = False
                elif dx == -1 and dy == 0:  # left neighbor
                    if neighbor.internal_edges["right_edge"] == structure_type:
                        is_isolated = False
        
        # Calculate points based on completion status
        if is_isolated:
            # Uncompleted structure - half points
            calculated_points = points * 0.5
            print(f"  📈 [MEEPLE] {edge} (uncompleted): {calculated_points} points")
        else:
            # Completed structure - full points
            calculated_points = points * 10.0
            print(f"  🏆 [MEEPLE] {edge} (completed): {calculated_points} points")
        
        if calculated_points > best_points:
            best_points = calculated_points
            best_structure = edge
            print(f"    🏆 [MEEPLE] New best! {calculated_points} points")
    
    if best_structure:
        print(f"🎯 [MEEPLE] Best structure: {best_structure} for {best_points} points")
        print(f"🎯 [MEEPLE] Placing meeple on {best_structure}", flush=True)
        bot_state.meeples_placed += 1
        return game.move_place_meeple(query, recent_tile._to_model(), best_structure)
    else:
        print(f"❌ [MEEPLE] No good structure found - passing", flush=True)
        return game.move_place_meeple_pass(query)




def get_all_valid_placements(game: Game, bot_state: BotState = None) -> List[Tuple[int, object, int, int, int]]:
    """
    Find all valid positions where any of the current tiles can be placed.
    Uses river-specific placement during river phase, regular placement otherwise.

    Returns:
        List of tuples: (tile_index, tile, x, y, rotation)
        Each tuple represents a valid placement option
    """
    # Check if we're in river phase
    if bot_state and bot_state.river_phase:
        print(f"🌊 [RIVER] River phase active - using river placement logic", flush=True)
        return get_valid_river_placement(game, bot_state)
    
    print(f"🎯 [REGULAR] Regular phase - using standard placement logic", flush=True)
    
    valid_placements = []

    # Define the 4 directions: right, down, left, up
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

    print(f"🔍 [SEARCH] Checking {len(game.state.map.placed_tiles)} placed tiles", flush=True)

    # For each available tile
    for tile_index, current_tile in enumerate(game.state.my_tiles):
        print(f"🔍 [SEARCH] Testing tile {current_tile.tile_type} (index {tile_index})", flush=True)
        
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
                            print(f"        ✅ [SEARCH] Tile {test_tile.tile_type} can be placed at ({adj_x}, {adj_y}) with rotation {test_rotation}", flush=True)

                            placement = (tile_index, current_tile, adj_x, adj_y, test_rotation)
                            valid_placements.append(placement)

    print(f"✅ [SEARCH] Found {len(valid_placements)} valid placement options", flush=True)
    return valid_placements


def local_can_place_tile_at(game: Game, tile: Tile, x: int, y: int, bot_state: BotState = None) -> bool:
    """
    Local implementation of the engine's can_place_tile_at function.
    This checks if a tile can be placed at the given position with its current rotation.
    Rotation testing is handled by get_all_valid_placements.
    """

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

    # print(f"🔍 [LOCAL] Checking tile {tile.tile_type} at ({x}, {y}) rotation {tile.rotation * 90}°")
    
    
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
    """
    Find valid river placements by tracing the river from the latest placed tile.
    Returns river head positions where new river tiles can be placed.
    """
    valid_placements = []
    
    if not game.state.map.placed_tiles:
        print("🌊 [RIVER] No placed tiles found")
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
                print(f"  🌊 [RIVER] Found river head at ({nx}, {ny}) - {direction_names[i]} of latest tile")
                
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
                            print(f"    ✅ [RIVER] Tile {test_tile.tile_type} can be placed at river head ({nx}, {ny}) with rotation {test_rotation}")
                            placement = (tile_index, current_tile, nx, ny, test_rotation)
                            valid_placements.append(placement)
                
                # If we found valid placements for this river head, we can stop checking other directions
                if valid_placements:
                    break
    
    print(f"🌊 [RIVER] Found {len(valid_placements)} valid river placements")
    return valid_placements




def local_structure_compatible(structure1: StructureType, structure2: StructureType) -> bool:
    """
    Our own structure compatibility validator - faster than the engine's version
    """
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
    """
    Robust river validation using engine's river_validation function.
    """
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
    
    
    
    
    

def analyze_tile_placement_structures(
    game: Game, 
    tile: Tile, 
    x: int, 
    y: int, 
    rotation: int
) -> dict:
    """
    Simulates placing a tile at the given position and analyzes all structures.
    
    Returns a dictionary with:
    - 'claimable_structures': List of edges that can be claimed (not already claimed)
    - 'total_potential_points': Total points from all structures on this tile
    - 'structure_details': Dict mapping edge -> {'type': StructureType, 'points': int, 'claimed': bool}
    """
    
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
    
    # Check each edge of the tile
    for edge in ["top_edge", "right_edge", "bottom_edge", "left_edge"]:
        structure_type = test_tile.internal_edges[edge]
        
        # Skip grass and river edges (not claimable)
        if structure_type in [StructureType.GRASS, StructureType.RIVER]:
            structure_details[edge] = {
                'type': structure_type,
                'points': 0,
                'claimed': False
            }
            continue
        
        # Check if this structure is already claimed
        # We need to temporarily update the game's grid for proper analysis
        original_grid = game.state.map._grid
        game.state.map._grid = grid_copy
        existing_claims = game.state._get_claims(test_tile, edge)
        game.state.map._grid = original_grid  # Restore immediately
        is_claimed = len(existing_claims) > 0
        
        # Calculate potential points for this structure
        if structure_type == StructureType.CITY:
            points = 2  # 2 points per tile for cities
        elif structure_type == StructureType.ROAD:
            points = 1  # 1 point per tile for roads
        elif structure_type == StructureType.ROAD_START:
            points = 1  # 1 point per tile for roads
        elif structure_type == StructureType.MONASTARY:
            points = 9  # 9 points for completed monastery
        else:
            points = 0
        
        # Add points to total if not claimed
        if not is_claimed:
            total_potential_points += points
            claimable_structures.append(edge)
        
        # Store structure details
        structure_details[edge] = {
            'type': structure_type,
            'points': points,
            'claimed': is_claimed
        }
    

    
    # No need to restore grid state since we used a copy
    
    return {
        'claimable_structures': claimable_structures,
        'total_potential_points': total_potential_points,
        'structure_details': structure_details
    }


def evaluate_tile_placement_for_points(
    game: Game, 
    bot_state: BotState, 
    tile: Tile, 
    x: int, 
    y: int, 
    rotation: int
) -> float:
    """
    Evaluate tile placement specifically for point maximization.
    Returns the net points we would gain from this placement.
    """
    # Check if this placement would complete any structures
    completion_result = check_tile_placement_completion(game, tile, x, y, rotation)
    
    net_points = 0.0
    
    # Add points from completed structures we own (full points)
    if completion_result['our_points_gained'] > 0:
        net_points += completion_result['our_points_gained'] * 10.0  # Big bonus for completed structures
        print(f"🏆 [POINTS] Completed our structures: +{completion_result['our_points_gained'] * 10.0}")
    
    # Add points from uncompleted structures we own (half points)
    if completion_result['our_uncompleted_points'] > 0:
        net_points += completion_result['our_uncompleted_points'] * 0.5  # Half points for uncompleted
        print(f"📈 [POINTS] Uncompleted our structures: +{completion_result['our_uncompleted_points'] * 0.5}")
    
    # Subtract points opponents would gain from completed structures (negative for us)
    if completion_result['opponent_points_gained'] > 0:
        # Calculate penalty for helping opponents
        opponent_penalty = completion_result['opponent_points_gained'] * 5.0
        
        # If we also gain points, reduce the penalty by 1/3
        if completion_result['our_points_gained'] > 0:
            opponent_penalty *= (2.0 / 3.0)  # Reduce penalty by 1/3
            print(f"💀 [POINTS] Opponents completed structures: -{opponent_penalty} (reduced penalty - we also gain points)")
        else:
            print(f"💀 [POINTS] Opponents completed structures: -{opponent_penalty}")
        
        net_points -= opponent_penalty
    
    # Add bonus for city modifier emblems
    city_modifier_bonus = calculate_city_modifier_bonus(tile, x, y, rotation)
    if city_modifier_bonus > 0:
        net_points += city_modifier_bonus
        print(f"🏰 [POINTS] City modifier emblem bonus: +{city_modifier_bonus}")
    
    print(f"📊 [POINTS] Net points: {net_points}")
    
    return net_points


def calculate_city_modifier_bonus(
    tile: Tile, 
    x: int, 
    y: int, 
    rotation: int
) -> float:
    """
    Calculate bonus points for cities with modifier emblems.
    """
    bonus = 0.0
    
    # Create a temporary copy of the tile for simulation
    test_tile = deepcopy(tile)
    
    # Set the tile to the target rotation
    while test_tile.rotation != 0:
        test_tile.rotate_clockwise(1)
    test_tile.rotate_clockwise(rotation)
    
    # Check each edge for city structures with modifier emblems
    for edge in ["top_edge", "right_edge", "bottom_edge", "left_edge"]:
        structure_type = test_tile.internal_edges[edge]
        
        if structure_type == StructureType.CITY:
            # Check if this city has a modifier emblem
            if has_city_modifier_emblem(test_tile, edge):
                bonus += 2.0
                print(f"    🏰 [MODIFIER] {edge} city has modifier emblem: +2.0")
    
    return bonus


def has_city_modifier_emblem(tile: Tile, edge: str) -> bool:
    """
    Check if a city edge has a modifier emblem.
    This is a simplified check - in practice you'd check the actual tile properties.
    """
    # For now, we'll use a simple heuristic based on tile type
    # In a real implementation, you'd check the actual tile's modifier properties
    
    # Check if the tile type suggests it has modifier emblems
    tile_type = tile.tile_type.lower()
    
    # Common modifier emblem indicators in tile names
    modifier_indicators = ['cathedral', 'shield', 'banner', 'flag', 'emblem', 'coat']
    
    for indicator in modifier_indicators:
        if indicator in tile_type:
            return True
    
    return False


def find_best_tile_placement_for_points(
    game: Game, 
    bot_state: BotState
) -> tuple[int, Tile, int, int, int, float]:
    """
    Find the tile placement that would give us the most points.
    
    Returns:
        (tile_index, tile, x, y, rotation, points)
    """
    best_placement = None
    best_points = -999999.0
    
    # Get all possible valid placements
    possible_placements = get_all_valid_placements(game, bot_state)
    
    if not possible_placements:
        print("❌ [POINTS] No valid placements found!")
        return None
    
    print(f"🔍 [POINTS] Evaluating {len(possible_placements)} possible placements for maximum points...")
    
    for tile_idx, tile, x, y, rotation in possible_placements:
        points = evaluate_tile_placement_for_points(game, bot_state, tile, x, y, rotation)
        
        print(f"  📍 [POINTS] Tile {tile.tile_type} at ({x}, {y}) rotation {rotation}: {points} points")
        
        if points > best_points:
            best_points = points
            best_placement = (tile_idx, tile, x, y, rotation, points)
            print(f"    🏆 [POINTS] New best! {points} points")
    
    if best_placement:
        tile_idx, tile, x, y, rotation, points = best_placement
        print(f"🎯 [POINTS] Best placement: Tile {tile.tile_type} at ({x}, {y}) rotation {rotation} for {points} points")
        
        # Store the best placement in bot_state for meeple placement
        bot_state.best_tile_placement = {
            'tile_idx': tile_idx,
            'tile': tile,
            'x': x,
            'y': y,
            'rotation': rotation,
            'points': points
        }
    
    return best_placement


def evaluate_tile_placement(
    game: Game, bot_state: BotState, tile, x: int, y: int, rotation: int
) -> float:
    """
    Evaluate the strategic value of a tile placement.
    Returns a score where higher is better.
    """
    score = 0.0

    # Start with a base score for any valid placement
    score += 1.0

    # Check if this placement would complete any structures
    completion_result = check_tile_placement_completion(game, tile, x, y, rotation)
    
    if completion_result['completed_structures']:
        print(f"🎯 [EVAL] Tile placement would complete {len(completion_result['completed_structures'])} structures")
        
        # Bonus for completing our own structures
        if completion_result['our_points_gained'] > 0:
            score += completion_result['our_points_gained'] * 10.0  # Big bonus for our points
            print(f"🏆 [EVAL] We would gain {completion_result['our_points_gained']} points")
        
        # Penalty for completing opponent structures (but smaller than our bonus)
        if completion_result['opponent_points_gained'] > 0:
            # Calculate penalty for helping opponents
            opponent_penalty = completion_result['opponent_points_gained'] * 5.0
            
            # If we also gain points, reduce the penalty by 1/3
            if completion_result['our_points_gained'] > 0:
                opponent_penalty *= (2.0 / 3.0)  # Reduce penalty by 1/3
                print(f"💀 [EVAL] Opponents would gain {opponent_penalty} points (reduced penalty - we also gain points)")
            else:
                print(f"💀 [EVAL] Opponents would gain {opponent_penalty} points")
            
            score -= opponent_penalty
        
        # Additional strategic considerations
        for edge in completion_result['completed_structures']:
            players = completion_result['claiming_players'][edge]
            details = completion_result['structure_details'][edge]
            
            # If we're completing an opponent's structure, consider if it frees up their meeples
            our_player_id = game.state.me.player_id
            if our_player_id not in players and players:
                # We're completing an opponent's structure - this frees up their meeples
                # This could be bad if they can immediately place more meeples
                score -= 2.0  # Small penalty for freeing opponent meeples
                print(f"⚠️ [EVAL] Would free opponent meeples from {edge}")

    # Small random factor to break ties
    score += random.random() * 0.1

    return score


def evaluate_structure_completion(game: Game, tile, x: int, y: int) -> float:
    """Evaluate if this placement completes any structures."""
    return random.uniform(0, 2)


def evaluate_structure_extension(game: Game, tile, x: int, y: int) -> float:
    """Evaluate if this placement extends our existing structures."""
    my_meeples = game.state.get_meeples_placed_by(game.state.me.player_id)

    bonus = 0.0
    for meeple in my_meeples:
        if meeple.placed and meeple.placed.placed_pos:
            meeple_x, meeple_y = meeple.placed.placed_pos
            distance = abs(meeple_x - x) + abs(meeple_y - y)
            if distance <= 2:  # Close to our meeples
                bonus += 1.0 / (distance + 1)

    return bonus


def evaluate_future_potential(game: Game, tile, x: int, y: int) -> float:
    """Evaluate the future potential of this placement."""
    grid = game.state.map._grid
    height = len(grid)
    width = len(grid[0]) if height > 0 else 0

    open_edges = 0
    for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
        adj_x, adj_y = x + dx, y + dy
        if 0 <= adj_x < width and 0 <= adj_y < height and grid[adj_y][adj_x] is None:
            open_edges += 1

    return open_edges * 0.3


def evaluate_meeple_placement(
    game: Game, bot_state: BotState, edge: str, structure_type: StructureType
) -> float:
    """
    Evaluate the value of placing a meeple on a specific structure.
    Returns a score where higher is better.
    """
    score = 0.0

    # Base score varies by structure type
    if structure_type == StructureType.CITY:
        score += 3.0  # Cities are generally more valuable
    elif structure_type == StructureType.ROAD:
        score += 2.0  # Roads are medium value
    elif structure_type == StructureType.FIELD:
        score += 1.0  # Fields are lower immediate value
    else:
        score += 2.5  # Monasteries and others

    # Bonus for defensive strategy on cities
    if bot_state.strategy_preference == "defensive" and structure_type == StructureType.CITY:
        score += 2.0

    return score


def get_unclaimed_structures(
    game: Game, tile_model: TileModel, structures: Dict[str, StructureType]
) -> Dict[str, StructureType]:
    """
    Filter structures to only include those that are not already claimed.
    """
    unclaimed = {}
    for edge, structure_type in structures.items():
        if is_structure_unclaimed(game, tile_model, edge):
            unclaimed[edge] = structure_type
    return unclaimed


def is_structure_unclaimed(game: Game, tile, edge: str) -> bool:
    """
    Check if a structure on a tile is already claimed by a meeple.
    """
    # This is a simplified check - in a real implementation you'd need to
    # check the actual structure ownership logic
    return True


def check_tile_placement_completion(
    game: Game, 
    tile: Tile, 
    x: int, 
    y: int, 
    rotation: int
) -> dict:
    """
    Check if placing a tile at the given position would complete any structures.
    
    Returns a dictionary with:
    - 'completed_structures': List of edges that would be completed
    - 'uncompleted_structures': List of edges that would be uncompleted (we have meeples on)
    - 'claiming_players': Dict mapping edge -> list of player IDs who have meeples on this structure
    - 'structure_details': Dict mapping edge -> {'type': StructureType, 'tiles_count': int, 'points': int, 'completed': bool}
    - 'our_points_gained': Total points we would gain from completing our own structures
    - 'opponent_points_gained': Total points opponents would gain from completing their structures
    - 'our_uncompleted_points': Points from uncompleted structures we own
    """
    
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
    
    completed_structures = []
    uncompleted_structures = []
    claiming_players = {}
    structure_details = {}
    our_points_gained = 0
    opponent_points_gained = 0
    our_uncompleted_points = 0
    
    # Check each edge of the tile for completion
    for edge in ["top_edge", "right_edge", "bottom_edge", "left_edge"]:
        structure_type = test_tile.internal_edges[edge]
        
        # Skip grass and river edges (not claimable structures)
        if structure_type in [StructureType.GRASS, StructureType.RIVER]:
            continue
        
        # Get all players who have meeples on this structure
        players_on_structure = get_players_on_structure(test_tile, edge, grid_copy)
        claiming_players[edge] = players_on_structure
        
        # Calculate structure details
        tiles_in_component = count_tiles_in_component(test_tile, edge, grid_copy)
        
        # Calculate points based on structure type
        if structure_type == StructureType.CITY:
            points = 2 * tiles_in_component  # 2 points per tile for cities
        elif structure_type == StructureType.ROAD:
            points = 1 * tiles_in_component  # 1 point per tile for roads
        elif structure_type == StructureType.ROAD_START:
            points = 1 * tiles_in_component  # 1 point per tile for road starts (same as roads)
        elif structure_type == StructureType.MONASTARY:
            points = 9  # 9 points for completed monastery
        else:
            points = 0
        
        our_player_id = game.state.me.player_id
        
        # Check if this structure would be completed
        if is_structure_completed(test_tile, edge, grid_copy):
            completed_structures.append(edge)
            
            structure_details[edge] = {
                'type': structure_type,
                'tiles_count': tiles_in_component,
                'points': points,
                'completed': True
            }
            
            # Calculate who gets the points
            if our_player_id in players_on_structure:
                # We have a meeple on this completed structure
                our_points_gained += points
            else:
                # Opponents have meeples on this completed structure
                opponent_points_gained += points
        else:
            # Structure is not completed
            structure_details[edge] = {
                'type': structure_type,
                'tiles_count': tiles_in_component,
                'points': points,
                'completed': False
            }
            
            # Only count uncompleted points if we own the structure
            if our_player_id in players_on_structure:
                our_uncompleted_points += points
    
    return {
        'completed_structures': completed_structures,
        'uncompleted_structures': uncompleted_structures,
        'claiming_players': claiming_players,
        'structure_details': structure_details,
        'our_points_gained': our_points_gained,
        'opponent_points_gained': opponent_points_gained,
        'our_uncompleted_points': our_uncompleted_points
    }


def is_structure_completed(start_tile: Tile, edge: str, grid: list[list[Tile | None]]) -> bool:
    """
    Check if a structure is completed by checking if ALL edges in the component have external tiles.
    This matches the game engine's get_completed_components logic exactly.
    Special handling for roads: only complete when there are exactly two start points.
    """
    
    # Get all tiles and edges in this connected component
    component = list(traverse_connected_component(start_tile, edge, grid))
    
    # Check if EVERY tile in the component has an external tile for its edge
    for tile, component_edge in component:
        assert tile.placed_pos is not None
        
        # Get the external tile in the direction of this edge
        external_tile = get_external_tile(component_edge, tile.placed_pos, grid)
        
        # If any edge doesn't have an external tile, the structure is not completed
        if external_tile is None:
            return False
    
    # Special handling for roads: check if there are exactly two start points
    structure_type = start_tile.internal_edges[edge]
    if structure_type in [StructureType.ROAD, StructureType.ROAD_START]:
        return is_road_properly_completed(component)
    
    # If all edges have external tiles, the structure is completed
    return True


def is_road_properly_completed(component: list[tuple[Tile, str]]) -> bool:
    """
    Check if a road is properly completed by having exactly two start points.
    """
    start_points = 0
    
    for tile, edge in component:
        structure_type = tile.internal_edges[edge]
        
        # Count ROAD_START structures
        if structure_type == StructureType.ROAD_START:
            start_points += 1
    
    # Road is only completed if it has exactly two start points
    return start_points == 2


def traverse_connected_component(
    start_tile: Tile, 
    edge: str, 
    grid: list[list[Tile | None]]
) -> list[tuple[Tile, str]]:
    """
    Traverse a connected component starting from a tile and edge.
    Based on the game engine's _traverse_connected_component logic.
    """
    from collections import deque
    
    visited = set()
    component = []
    
    # Not a traversable edge - ie monastery etc
    if edge not in start_tile.internal_edges.keys():
        return component
    
    structure_type = start_tile.internal_edges[edge]
    
    queue = deque([(start_tile, edge)])
    
    while queue:
        tile, edge = queue.popleft()
        
        if (tile, edge) in visited:
            continue
        
        # Visiting portion of traversal
        visited.add((tile, edge))
        component.append((tile, edge))
        
        connected_internal_edges = [edge]
        opposite_edge = get_opposite_edge(edge)
        
        # Check directly adjacent edges
        for adjacent_edge in get_adjacent_edges(edge):
            if tile.internal_edges[adjacent_edge] == structure_type:
                connected_internal_edges.append(adjacent_edge)
        
        # Opposite edge if adjacent connection
        if (len(connected_internal_edges) > 1 and 
            tile.internal_edges[opposite_edge] == structure_type):
            connected_internal_edges.append(opposite_edge)
        
        # Handle ROAD_START -> ROAD conversion
        if structure_type == StructureType.ROAD_START:
            structure_type = StructureType.ROAD
        
        # Add all connected internal edges to visited
        for adjacent_edge in connected_internal_edges[1:]:
            visited.add((tile, adjacent_edge))
            component.append((tile, adjacent_edge))
        
        # External Tiles
        for ce in connected_internal_edges:
            assert tile.placed_pos
            ce_neighbour = get_opposite_edge(ce)
            
            external_tile = get_external_tile(ce, tile.placed_pos, grid)
            
            if external_tile is None:
                continue
            
            # Check compatibility - roads and road starts are compatible
            external_structure_type = external_tile.internal_edges[ce_neighbour]
            is_compatible = False
            
            if structure_type == StructureType.ROAD:
                # Roads are compatible with both ROAD and ROAD_START
                is_compatible = (external_structure_type in [StructureType.ROAD, StructureType.ROAD_START])
            else:
                # Use engine's compatibility check for other structures
                is_compatible = StructureType.is_compatible(structure_type, external_structure_type)
            
            if not is_compatible:
                continue
            
            if (external_tile, ce_neighbour) not in visited:
                queue.append((external_tile, ce_neighbour))
    
    return component


def get_players_on_structure(start_tile: Tile, edge: str, grid: list[list[Tile | None]]) -> list[int]:
    """
    Get all player IDs who have meeples on a structure.
    """
    players = set()
    
    # Traverse the connected component
    for tile, component_edge in traverse_connected_component(start_tile, edge, grid):
        meeple = tile.internal_claims[component_edge]
        if meeple is not None:
            players.add(meeple.player_id)
    
    return list(players)


def count_tiles_in_component(start_tile: Tile, edge: str, grid: list[list[Tile | None]]) -> int:
    """
    Count unique tiles in a connected component.
    """
    unique_tiles = set()
    
    for tile, _ in traverse_connected_component(start_tile, edge, grid):
        unique_tiles.add(tile)
    
    return len(unique_tiles)


def get_external_tile(edge: str, pos: tuple[int, int], grid: list[list[Tile | None]]) -> Tile | None:
    """
    Get the external tile in the direction of an edge.
    """
    match edge:
        case "left_edge":
            return grid[pos[1]][pos[0] - 1]
        case "right_edge":
            return grid[pos[1]][pos[0] + 1]
        case "top_edge":
            return grid[pos[1] - 1][pos[0]]
        case "bottom_edge":
            return grid[pos[1] + 1][pos[0]]
        case _:
            return None


def get_opposite_edge(edge: str) -> str:
    """
    Get the opposite edge.
    """
    return {
        "left_edge": "right_edge",
        "right_edge": "left_edge",
        "top_edge": "bottom_edge",
        "bottom_edge": "top_edge",
    }[edge]


def get_adjacent_edges(edge: str) -> list[str]:
    """
    Get adjacent edges.
    """
    return {
        "left_edge": ["top_edge", "bottom_edge"],
        "right_edge": ["top_edge", "bottom_edge"],
        "top_edge": ["left_edge", "right_edge"],
        "bottom_edge": ["left_edge", "right_edge"],
    }[edge]


### NOT USED RN


def test_completion_checking(game: Game, tile: Tile, x: int, y: int, rotation: int) -> None:
    """
    Test function to demonstrate how to use the completion checking.
    """
    print(f"🔍 [TEST] Checking completion for tile {tile.tile_type} at ({x}, {y}) rotation {rotation}")
    
    result = check_tile_placement_completion(game, tile, x, y, rotation)
    
    if result['completed_structures']:
        print(f"✅ [TEST] Would complete structures: {result['completed_structures']}")
        print(f"🏆 [TEST] Our points gained: {result['our_points_gained']}")
        print(f"💀 [TEST] Opponent points gained: {result['opponent_points_gained']}")
        
        for edge in result['completed_structures']:
            players = result['claiming_players'][edge]
            details = result['structure_details'][edge]
            
            print(f"  📋 [TEST] {edge}:")
            print(f"    🎯 Type: {details['type']}")
            print(f"    🧩 Tiles: {details['tiles_count']}")
            print(f"    🏆 Points: {details['points']}")
            print(f"    👥 Claiming players: {players}")
    else:
        print(f"❌ [TEST] Would not complete any structures")



if __name__ == "__main__":
    main()
