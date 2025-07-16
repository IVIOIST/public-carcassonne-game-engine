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
    pass

def handle_place_meeple(
    game: Game, bot_state: BotState, query: QueryPlaceMeeple
) -> MovePlaceMeeplePass | MovePlaceMeeple:
    pass

if __name__ == "__main__":
    main()
