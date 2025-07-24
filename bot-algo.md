This bot uses a rule-based heuristic algorithm to evaluate all valid tile placements each turn. For every candidate position and orientation, it simulates placing the tile and estimates the total potential score by analyzing what connected structures could be created. Factors such as number of open edges (completion factor), structure size, emblems, and existing meeples (ours and opponents') are all considered

There are 2 phases. Phase 1 occurs when the highest scoring player is below 30 points. The analysis doesn't care how close a city is to completion in this stage, and roads have minimal penalties if placing the tile there wouldn't complete it. In phase 2 (> 30 points), a penalty is imposed (the completion factor) if a city or road would not be completed by placing the tile there (so the points are yet to be realized, and the meeple wont be immediately freed). The severity is based on the number of open edges in the structure (a city with 3 open ends has completion factor 0.3, a 70% reduction)

For monastaries we check the current scoring potential (how many surrounding tiles). To normalize the monastary points with cities and roads (which get a bunch of heuristic modifiers), if placing the monastary immediately completes it, it gets an additional 2 points when considering this placement. 

Pretty much all of this processing is done in analyze_tile_placement_structures, which analyzes all possible tile placements and returns information pertaining to the value of the placement (as discussed above). The tile with the highest total_potential_points is placed. Then a few more minor heuristics are applied in handle_place_tile (have we placed too many meeples, is the round number less than 10 and we can be more conservative for meeple placement) and the best_edge for the meeple is added to bot_state. handle_place_meeple pretty much places the meeple from recent_tile and best_edge stored in bot_state by handle_place_tile. sometimes best_edge is None if handle_place_tile doesn't think we should place a meeple so then handle_place_meeple passes

It's score efficient cause the analysis function greedily gets the best tile based on our heuristics (the top sorted element of the list returned in analyze_tile_placement_structures) and the best edge as well. It's also time efficient because we aren't doing crazy quadruple nested for loops. lots of if statements and calculations are done which arent time consuming. 

unikeys for pep: 
kmum0850
esun0349
ccas2710

jwan0839 (Tim not doing pep)
