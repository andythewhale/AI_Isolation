"""Finish all TODO items in this file to complete the isolation project, then
test your agent's strength against a set of known agents using tournament.py
and include the results in your report.
"""
import random


class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    pass


# This is for my be agressive / run away scoring. I like this as a sub strategy for always targeting the enemy.
# I want custom score 2 and 3 to be aggressive and move towards the enemy and run away.
# custom score 1 will be the flagship
# This is just a really easy way to have some basic strategies.
def distance_to_enemy(game, me, enemy):
    my_locale = game.get_player_location(me)
    enemy_locale = game.get_player_location(enemy)
    x_dist = (my_locale[0] - enemy_locale[0]) ** 2
    y_dist = (my_locale[1] - enemy_locale[1]) ** 2
    total_distance = x_dist + y_dist
    return float(total_distance)


# This blocks the enemy player when they only have one move remaining.
# Pretty sure if we have the time to calculate this we should always stomp out the enemy
# So this can be incorporated into any strategy so long as there is any computing time.
def ko_move(game, me, enemy_moves, my_moves):
    value = 0
    spot = enemy_moves[0]
    for move in my_moves:
        if move == spot:
            value = float('inf')
    return value


def ko_match(my_moves, enemy_moves):
    check = False
    spot = enemy_moves
    for move in my_moves:
        if move == spot:
            check = True
        else:
            check = False
    return check


# This function figures out the percent of moves used for the board space.
# Most games start getting harder when there is about 35-45% of the board taken up.
# Our strategy needs to change if this is so.
# We're going to assume 49 spaces. But this can be generalized to any board size.
# As the number of spaces increases, the limiting factor becomes the square perimeters of the board.
# 25% of the board taken up, means there's about 2-3 perimeters remaining.
# The only reason I don't generalize is because it's more efficient.
# You should always be more efficient when you have the resources to be.
# If the enemy player is using the flee strategy than this is advantageous as well.
def percent_moves_used(blank_space):
    percent_blank = (blank_space) / 49
    return float(percent_blank)


def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.
    This should be the best heuristic function for your project submission.
    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.
    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).
    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)
    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    # TODO: finish this function!
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    # This will be our beginning strategy.
    # Stay as close to the center as possible
    # Do this as long as 75% of the spaces remain open on the board.
    # I copied this code from the center sample player. It's a good strategy for the beginning.
    a = game.get_blank_spaces()
    if percent_moves_used(float(len(a))) < float(0.62):
        w, h = game.width / 2., game.height / 2.
        y, x = game.get_player_location(player)
        return float((h - y) ** 2 + (w - x) ** 2)

        # Do we have any KO's?
        # enemy_moves = game.get_legal_moves(game.get_opponent(player))
        # num_enemy_moves = len(enemy_moves)
        # if num_enemy_moves == 1 & ko_match(game.get_legal_moves(player), enemy_moves) == True:
        # return ko_move(game, player, enemy_moves, game.get_legal_moves(player))

        # If I can't kill you, I need to get some space and be alone.
    return float(len(game.get_legal_moves(player)))


def custom_score_2(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.
    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.
    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).
    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)
    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    # TODO: finish this function!
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    # Let's run away.
    return distance_to_enemy(game, player, game.get_opponent(player))


def custom_score_3(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.
    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.
    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).
    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)
    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    # TODO: finish this function!
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    # Let's be really aggressive
    return 1 / distance_to_enemy(game, player, game.get_opponent(player))


class IsolationPlayer:
    """Base class for minimax and alphabeta agents -- this class is never
    constructed or tested directly.
    ********************  DO NOT MODIFY THIS CLASS  ********************
    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)
    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.
    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """

    def __init__(self, search_depth=3, score_fn=custom_score, timeout=10.):
        self.search_depth = search_depth
        self.score = score_fn
        self.time_left = None
        self.TIMER_THRESHOLD = timeout


class MinimaxPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using depth-limited minimax
    search. You must finish and test this player to make sure it properly uses
    minimax to return a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.
        **************  YOU DO NOT NEED TO MODIFY THIS FUNCTION  *************
        For fixed-depth search, this function simply wraps the call to the
        minimax method, but this method provides a common interface for all
        Isolation agents, and you will replace it in the AlphaBetaPlayer with
        iterative deepening search.
        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).
        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.
        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = (-1, -1)

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            return self.minimax(game, self.search_depth)

        except SearchTimeout:
            pass  # Handle any actions required after timeout as needed

        # Return the best move from the last completed search iteration
        return best_move

    def minimax(self, game, depth):
        """Implement depth-limited minimax search algorithm as described in
        the lectures.
        This should be a modified version of MINIMAX-DECISION in the AIMA text.
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Minimax-Decision.md
        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************
        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state
        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting
        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves
        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.
            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """

        # OK so first we need to check to make sure we haven't timed out the function.
        # I didn't make this. This was given.
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        # First let's define our default values for the score and the best possible move.
        # If we get a better score or a better move this will change.
        # We need the score here to determine what the max/min score is.
        best_move = (-1, -1)
        highest_saved_score = float("-inf")

        # Now let's get the actual moves available, if we have any:
        legal_moves = game.get_legal_moves(game.active_player)

        # My intuition here was to first make an if statement asking for the length of our legal moves.
        # However, I got help, I can see now, if there are no legal moves, then there's nothing to do.
        # I was overthinking the problem.
        # So for each move in our legal_moves...
        for move in legal_moves:
            # We get a copy of the move for the forecasted game board if we make that move.
            forcasted_move = game.forecast_move(move)

            # We get the score of this move, and the depth is a step below and it is also at a minimizing edge.
            temporary_score = self.minum(forcasted_move, depth - 1)

            # If this score is greater than our highest saved score then our highest saved score is this score.
            if temporary_score > highest_saved_score:
                highest_saved_score = temporary_score

                # Our best_move is the move with the highest score.
                best_move = move

        # Now we return best_move to complete the interface.
        return best_move

    # This is the maximizing function for the minimax algorithm
    # If we're on a maximizing edge we should maximize the node at the next level.
    def maxim(self, game, depth):
        # This function takes self, the game state, and the depth as an argument.
        # This function outputs the score of the best move.

        # First let's make sure we have time to do this:
        # This if statement is given and I did not make it.
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        # Now let's check to make sure we're not at the root node:
        # NOTE: active_player took me forever to figure out.
        # I kept implementing it as 'self, game', kind of embarrassing.
        if depth == 0:
            return self.score(game, game.active_player)

        # Let's get that max score:
        # First we set it to negative infinity as a baseline
        opt_score = float("-inf")

        # Next we get the children of the current node our active player is standing at.
        children = game.get_legal_moves(game.active_player)

        # Now for each child of the set of children we find out what they're really worth.
        # We set the new child as equal to the best child if they're worth more.
        # This is recursion on our base case.
        # Check out the move score variable and the minum function. This is confusing and still makes my head spin.
        # At the next level down, we're taking the minimum score, that's why it looks kind of confusing.
        for child in children:
            move = game.forecast_move(child)  # Copy of gamestate with new move kind of applied, no commitment.
            move_score = self.minum(move, depth - 1)
            if move_score > opt_score:
                opt_score = move_score

        # For the longest time I was getting this wrong.
        # For some reason I thought the function must need to return the move and the score.
        # However this simplifies everything.
        return opt_score

    # Now we do the opposite of maxim:
    def minum(self, game, depth):
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        if depth == 0:
            # I was stuck on this part for a very long time.
            # I am a fool.
            return self.score(game, game.inactive_player)

        opt_score = float("inf")
        children = game.get_legal_moves(game.active_player)
        for child in children:
            move = game.forecast_move(child)
            move_score = self.maxim(move, depth - 1)
            if move_score <= opt_score:
                opt_score = move_score

        return opt_score


class AlphaBetaPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using iterative deepening minimax
    search with alpha-beta pruning. You must finish and test this player to
    make sure it returns a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.
        Modify the get_move() method from the MinimaxPlayer class to implement
        iterative deepening search instead of fixed-depth search.
        **********************************************************************
        NOTE: If time_left() < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************
        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).
        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.
        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # TODO: finish this function!
        # Not sure why but it was really scary for me to try and start this part of the project.
        # It took me about 2 days to actually sit down and look at alphabeta.
        # But to get started let's look at what needs to happen.
        # Most of this was easy by looking at the MiniMaxPlayer and copying it.
        # We need a default move if there is no default move.
        best_move = (-1, -1)

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            # The first thing I tried was just to copy the MiniMaxPlayer, it did not work.
            # So we need to start at the root and go down the tree using iterative deepening.
            # This is similar to depth first search but with pruning I think.
            depth = 1

            # The implementation of this while loop doesn't really matter, this is just a good way of doing it.
            # Keep in mind, without the except statement, this runs forever
            while True:
                best_move = self.alphabeta(game, depth)
                depth = depth + 1

                # This will run until we get a SearchTimeOut:
        except SearchTimeout:

            pass

        return best_move

    def alphabeta(self, game: object, depth: object, a: object = float("-inf"), b: object = float("inf")) -> object:
        """Implement depth-limited minimax search with alpha-beta pruning as
        described in the lectures.
        This should be a modified version of ALPHA-BETA-SEARCH in the AIMA text
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Alpha-Beta-Search.md
        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************
        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state
        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting
        alpha : float
            Alpha limits the lower bound of search on minimizing layers
        beta : float
            Beta limits the upper bound of search on maximizing layers
        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves
        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.
            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        # TODO: finish this function!
        # The additions here were hardest for me, not sure why:

        # First let's define our default values for the score and the best possible move.
        # If we get a better score or a better move this will change.
        # We need the score here to determine what the max/min score is.
        best_move = (-1, -1)
        highest_saved_score = float("-inf")

        # Now let's get the actual moves available, if we have any:
        legal_moves = game.get_legal_moves(game.active_player)

        # My intuition here was to first make an if statement asking for the length of our legal moves.
        # However, I got help, I can see now, if there are no legal moves, then there's nothing to do.
        # I was overthinking the problem.
        # So for each move in our legal_moves...
        for move in legal_moves:
            # We get a copy of the move for the forecasted game board if we make that move.
            forcasted_move = game.forecast_move(move)

            # We get the score of this move, and the depth is a step below and it is also at a minimizing edge.
            temporary_score = self.abminum(forcasted_move, depth - 1, a, b)

            # If this score is greater than our highest saved score then our highest saved score is this score.
            if temporary_score > highest_saved_score:
                highest_saved_score = temporary_score

                # Our best_move is the move with the highest score.
                best_move = move

            # The difference between alphabeta and minimax is here:
            if temporary_score >= b:
                return move

            # This just makes a equal to the higher value, for now.
            a = max(a, temporary_score)

        # Now we return best_move to complete the interface.
        return best_move

    # AB Maxim is very similar to normal Maxim but with AB.

    def abmaxim(self, game, depth, a, b):
        # This function takes self, the game state, the depth, a and b as an argument.
        # This function outputs the score of the best move.
        # This function also prunes nodes that could not possibly possess an optimal strategy for us.

        # First let's make sure we have time to do this:
        # This if statement is given and I did not make it.
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        # Now let's check to make sure we're not at the root node:
        # NOTE: active_player took me forever to figure out.
        # I kept implementing it as 'self, game', kind of embarrassing.
        if depth == 0:
            return self.score(game, game.active_player)

        # Let's get that max score:
        # First we set it to negative infinity as a baseline
        opt_score = float("-inf")

        # Next we get the children of the current node our active player is standing at.
        children = game.get_legal_moves(game.active_player)

        # Now for each child of the set of children we find out what they're really worth.
        # We set the new child as equal to the best child if they're worth more.
        # This is recursion on our base case.
        # Check out the move score variable and the minum function. This is confusing and still makes my head spin.
        # At the next level down, we're taking the minimum score, that's why it looks kind of confusing.
        for child in children:
            move = game.forecast_move(child)  # Copy of gamestate with new move kind of applied, no commitment.
            move_score = self.abminum(move, depth - 1, a, b)
            if move_score > opt_score:
                opt_score = move_score

                # Here's where things get way different:
            if opt_score >= b:
                return opt_score

            a = max(a, opt_score)

        # For the longest time I was getting this wrong.
        # For some reason I thought the function must need to return the move and the score.
        # However this simplifies everything.
        return opt_score

    # Now we do the opposite of abmaxim, which is the same as minim with beta:
    def abminum(self, game, depth, a, b):
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        if depth == 0:
            # I was stuck on this part for a very long time.
            # I couldn't understand to call this on the inactive player.
            return self.score(game, game.inactive_player)

        opt_score = float("inf")
        children = game.get_legal_moves(game.active_player)
        for child in children:
            move = game.forecast_move(child)
            move_score = self.abmaxim(move, depth - 1, a, b)
            if move_score < opt_score:
                opt_score = move_score

            if opt_score <= a:
                return opt_score

            b = min(b, opt_score)

        return opt_score
