"""
Tic-Tac-Toe with LMS Weight Learning
Mitchell (1997) - Machine Learning, Chapter 1, Exercise 1.5

The learning agent (X) uses a linear value function:
    V(b) = w0 + w1*f1 + w2*f2 + w3*f3 + w4*f4 + w5*f5

Features:
    f1 - agent's 2-in-a-row count (with empty third cell)
    f2 - opponent's 2-in-a-row count (threats)
    f3 - agent controls center (1 or 0)
    f4 - opponent corner count
    f5 - agent corner count

Opponent uses a fixed hand-crafted strategy.
Weights are updated using the LMS rule after each game.
"""

import random
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Board utilities
# ---------------------------------------------------------------------------

LINES = [
    [0, 1, 2], [3, 4, 5], [6, 7, 8],  # rows
    [0, 3, 6], [1, 4, 7], [2, 5, 8],  # columns
    [0, 4, 8], [2, 4, 6]               # diagonals
]
CORNERS = [0, 2, 6, 8]


def check_winner(board):
    """Return 'X', 'O', 'draw', or None."""
    for line in LINES:
        vals = [board[i] for i in line]
        if vals[0] and vals[0] == vals[1] == vals[2]:
            return vals[0]
    if all(board):
        return 'draw'
    return None


def legal_moves(board):
    return [i for i, v in enumerate(board) if not v]


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_features(board, agent):
    """
    Extract 6 features from board state for the given agent.
    Returns list: [bias, f1, f2, f3, f4, f5]
    """
    opp = 'O' if agent == 'X' else 'X'
    f1 = f2 = f3 = f4 = f5 = 0

    for line in LINES:
        vals = [board[i] for i in line]
        agent_count = vals.count(agent)
        opp_count   = vals.count(opp)
        empty       = vals.count(None)
        if agent_count == 2 and empty == 1:
            f1 += 1
        if opp_count == 2 and empty == 1:
            f2 += 1

    if board[4] == agent:
        f3 = 1
    f4 = sum(1 for c in CORNERS if board[c] == opp)
    f5 = sum(1 for c in CORNERS if board[c] == agent)

    return [1, f1, f2, f3, f4, f5]   # 1 is bias term


# ---------------------------------------------------------------------------
# Value function
# ---------------------------------------------------------------------------

def evaluate(board, agent, weights):
    """Linear value function V(b) = w . f(b)"""
    features = extract_features(board, agent)
    return sum(w * f for w, f in zip(weights, features))


# ---------------------------------------------------------------------------
# Agent move (greedy w.r.t. learned V)
# ---------------------------------------------------------------------------

def agent_move(board, agent, weights):
    moves = legal_moves(board)
    if not moves:
        return None

    best_score = float('-inf')
    best_move  = moves[0]

    for m in moves:
        nb = board[:]
        nb[m] = agent
        winner = check_winner(nb)
        if winner == agent:
            return m                          # immediate win
        score = evaluate(nb, agent, weights)
        if score > best_score:
            best_score = score
            best_move  = m

    return best_move


# ---------------------------------------------------------------------------
# Fixed hand-crafted opponent
# ---------------------------------------------------------------------------

def handcrafted_move(board, opp, agent):
    moves = legal_moves(board)
    if not moves:
        return None

    # 1. Win if possible
    for m in moves:
        nb = board[:]; nb[m] = opp
        if check_winner(nb) == opp:
            return m

    # 2. Block agent from winning
    for m in moves:
        nb = board[:]; nb[m] = agent
        if check_winner(nb) == agent:
            return m

    # 3. Take center
    if not board[4]:
        return 4

    # 4. Take a corner
    corners = [c for c in CORNERS if not board[c]]
    if corners:
        return random.choice(corners)

    # 5. Any move
    return random.choice(moves)


# ---------------------------------------------------------------------------
# LMS weight update
# ---------------------------------------------------------------------------

def update_weights(history, agent, outcome, weights, lr):
    """
    LMS rule: wi <- wi + lr * (V_target - V_pred) * fi
    Walk backwards through move history, using next state's value as target.
    Terminal outcome: win=1.0, draw=0.0, loss=-1.0
    """
    terminal = 1.0 if outcome == agent else (-1.0 if outcome != 'draw' else 0.0)

    for i in range(len(history) - 1, -1, -1):
        board, move = history[i]
        nb = board[:]
        nb[move] = agent
        features = extract_features(nb, agent)
        pred = sum(w * f for w, f in zip(weights, features))

        # Target: terminal value for last move, else next board's evaluation
        if i == len(history) - 1:
            target = terminal
        else:
            next_board, next_move = history[i + 1]
            nb2 = next_board[:]
            nb2[next_move] = agent
            target = evaluate(nb2, agent, weights)

        error = target - pred
        for j in range(len(weights)):
            weights[j] += lr * error * features[j]


# ---------------------------------------------------------------------------
# Play a single game
# ---------------------------------------------------------------------------

def play_game(weights, lr, agent='X', opp='O'):
    board = [None] * 9
    history = []
    turn = 0

    while True:
        winner = check_winner(board)
        if winner:
            update_weights(history, agent, winner, weights, lr)
            return winner

        if turn % 2 == 0:
            m = agent_move(board, agent, weights)
            history.append((board[:], m))
            board[m] = agent
        else:
            m = handcrafted_move(board, opp, agent)
            board[m] = opp

        turn += 1


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(num_games=2000, lr=0.1, window=50):
    weights = [0.0] * 6
    results = []
    win_rates = []
    game_nums = []

    for g in range(1, num_games + 1):
        result = play_game(weights, lr)
        results.append(1 if result == 'X' else 0)

        if g % window == 0:
            wr = sum(results[-window:]) / window * 100
            win_rates.append(wr)
            game_nums.append(g)
            print(f"Game {g:>5} | win rate (last {window}): {wr:.1f}% | "
                  f"weights: [{', '.join(f'{w:.3f}' for w in weights)}]")

    return weights, game_nums, win_rates


# ---------------------------------------------------------------------------
# Plot results
# ---------------------------------------------------------------------------

def plot_results(game_nums, win_rates, lr):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(game_nums, win_rates, color='#2a78d6', linewidth=2)
    ax.fill_between(game_nums, win_rates, alpha=0.1, color='#2a78d6')
    ax.axhline(y=50, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='50% baseline')
    ax.set_xlabel('Training games played')
    ax.set_ylabel('Win rate % (last 50 games)')
    ax.set_title(f'LMS Tic-Tac-Toe Learning Agent (η = {lr})')
    ax.set_ylim(0, 100)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('win_rate_plot.png', dpi=150)
    plt.show()
    print("Plot saved as win_rate_plot.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    LEARNING_RATE = 0.1
    NUM_GAMES     = 2000

    print("=" * 60)
    print("  Tic-Tac-Toe LMS Learning Agent — Mitchell Chapter 1")
    print("=" * 60)
    print(f"  Learning rate η = {LEARNING_RATE}")
    print(f"  Training games  = {NUM_GAMES}")
    print("=" * 60 + "\n")

    final_weights, game_nums, win_rates = train(
        num_games=NUM_GAMES,
        lr=LEARNING_RATE
    )

    print("\nFinal learned weights:")
    labels = ['w0 (bias)', 'w1 (2-in-row)', 'w2 (opp 2-in-row)',
              'w3 (center)', 'w4 (opp corners)', 'w5 (agent corners)']
    for label, w in zip(labels, final_weights):
        print(f"  {label:<22} = {w:.4f}")

    plot_results(game_nums, win_rates, LEARNING_RATE)