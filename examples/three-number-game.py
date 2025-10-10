# I thought of a simple game two player game.each player get a random number 1, 2 or 3 
# they can decide to keep it or redraw the number
# if a player want to redraw, the other player can ask to check the currently held number, if it is a 2 or a 3, then the player who wanted to redraw loses 10$ and the game ends. If it is a 1 the checking player loses 10$ and the game ends.
# if both want to redraw the games ends in a draw.
# if both keep the numbers are compared and the person with the highest number gets 3$ (if equal it's a draw and no money is exchanged.

# What is the optimal strategy?


def P1_winnings(p1, c1, n1, p2, c2, n2, W=4, R=10, G=2.6): #2.6 gives 50%
    # p1 is the prob that player 1 keeps the number
    # p2 is the prob that player 2 keeps the number
    # c1 is the probability that player 1 keeps the number
    # c2 is the probability that player 2 keeps the number
    # n1 is the current number of player 1
    # n2 is the current number of player 2
    # W is the amount of money exchanged if the numbers are compared and someone wins.
    # R is the amount of money exchanged if checking player wins

    S = 0 #expected value for player 1
    def sign(x):
        return 1 if x > 0 else -1 if x < 0 else 0

    S += p1 * p2 * sign(n1-n2) * W  # both keep
    S += (1 - p1) * p2 * ( # player 2 can check
        c2 * (R/G if n1 == 1 else -R) + #if they check and the nuber is 1 player 1 wins R, else p1 if it is 2 or 3 they lose R
        (1 - c2) * sign(2 - n2) * W * 2/3 # if they don't check, player 1 wins with prob 2/3 if n2 is 2 or 3
    )
    S += p1 * (1 - p2) * ( # player 2 redraws and player 1 can check
        c1 * (-R/G if n2 == 1 else R) + #if player 1 checks and the number is 1 they lose R, else p2 if it is 2 or 3 they lose R
        (1 - c1) * sign(n1 - 2) * W * 2/3 # if they don't check, player 2 wins with prob 2/3 if n1 is 2 or 3
    )
    S += (1 - p1) * (1 - p2) * 0
    return S

def compute_expected_winning_P1(p1, c1, p2, c2):
    S = 0
    for n1 in [1, 2, 3]:
        for n2 in [1, 2, 3]:
            S += P1_winnings(p1[n1-1], c1[n1-1], n1, p2[n2-1], c2[n2-1], n2)/9
    return S

p2_list = [[0.5, 0.5, 0.5]]
c2_list = [[0.5, 0.5, 0.5]]


#maximize_p1_c1 compute_expected_winning_P1
# s.t. 0 <= p1 <= 1
#      0 <= c1 <= 1
#The Support of Mixed Strategies

from scipy.optimize import minimize
def objective(x):
    p1 = [x[0], x[1], x[2]]
    c1 = [x[3], x[4], x[5]]
    return max(-compute_expected_winning_P1(p1, c1, p2, c2) for p2, c2 in zip(p2_list, c2_list))

for i in range(20):  # run the optimization multiple times to find a good solution
    print(f"Iteration {i+1}")
    x0 = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]  # initial guess
    bounds = [(0, 1)] * 6  # bounds for p1 and c1
    result = minimize(objective, x0, bounds=bounds, method='Powell')
    if result.success:
        optimal_p1 = result.x[:3]
        optimal_c1 = result.x[3:]
        print(f"\tOptimal p1: p[0] = {optimal_p1[0]:.2%} \tp[1] = {optimal_p1[1]:.2%} \tp[2] = {optimal_p1[2]:.2%}")
        print(f"\tOptimal c1: c[0] = {optimal_c1[0]:.2%} \tc[1] = {optimal_c1[1]:.2%} \tc[2] = {optimal_c1[2]:.2%}")
        p2_list.append(optimal_p1)
        c2_list.append(optimal_c1)
        print("\tExpected winnings for player 1:", -result.fun)
        if abs(result.fun) < 1e-4:
            break
    else:
        print("\tOptimization failed:", result.message)
