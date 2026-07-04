# Tic-Tac-Toe — LMS Learning Agent

Implements a tic-tac-toe playing agent that learns a linear value function
using the LMS (Least Mean Squares) weight update rule.

Based on Exercise 1.5 from Tom Mitchell's _Machine Learning_ (1997).

## How it works

The agent evaluates board positions using a linear combination of features:

V(b) = w0 + w1·f1 + w2·f2 + w3·f3 + w4·f4 + w5·f5

| Feature | Description                 |
| ------- | --------------------------- |
| f1      | Agent's 2-in-a-row count    |
| f2      | Opponent's 2-in-a-row count |
| f3      | Agent controls center       |
| f4      | Opponent corner count       |
| f5      | Agent corner count          |

Weights are updated after each game using the LMS rule. The agent trains
against a fixed hand-crafted opponent that plays near-optimally.

## Running it

```bash
pip install matplotlib numpy
python tictactoe_lms.py
```

## Output

- Live win rate printed every 50 games
- Final learned weights printed at the end
- Win rate plot saved as `win_rate_plot.png`
