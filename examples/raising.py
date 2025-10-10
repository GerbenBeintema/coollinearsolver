from scipy.optimize import minimize


def run_example(verbose=1):
    def expected_winning_P1(p1, f1, n1, p2, f2, n2, W = 1, G = 100):
        S = 0  # expected value for player 1
        def sign(x):
            return 1 if x > 0 else -1 if x < 0 else 0
        s = sign(n1 - n2)
        S += (1 - p1) * (1 - p2) * s * W  # both call
        S += (1 - p1) * p2 *( #player 1 calls and player 2 raises
            f1 * -W + # player 1 folds and loses W
            (1 - f1) * s * G # player 1 calls
        ) #p2 raises
        S += p1 * (1-p2) *(
            f2 * W + # player 2 folds and loses W
            (1 - f2) * s * G # player 2 calls
        ) #p1 raises
        S += p1 * p2 * G * s # both raise
        return S

    N = 5 # pull 0 to N-1 randomly

    def compute_expected_winning_P1(p1, f1, p2, f2):
        S = 0
        for n1 in range(N):
            for n2 in range(N):
                S += expected_winning_P1(p1[n1], f1[n1], n1, p2[n2], f2[n2], n2)
        return S / (N * N)

    p2_list = [[0.5] * N]
    f2_list = [[0.5] * N]

    def objective(x):
        p1 = [x[i] for i in range(N)]
        f1 = [x[i + N] for i in range(N)]
        return max(-compute_expected_winning_P1(p1, f1, p2, f2) for p2, f2 in zip(p2_list, f2_list))

    for i in range(20):  # run the optimization multiple times to find a good solution
        if verbose:
            print(f"Iteration {i+1}")
        x0 = [0.5] * (2 * N)  # initial guess
        bounds = [(0, 1)] * (2 * N)
        result = minimize(objective, x0, bounds=bounds, method='Powell')
        if result.success:
            optimal_p1 = result.x[:N]
            optimal_f1 = result.x[N:]
            if verbose:
                print(f"\tOptimal p1:", ' '.join(f'{optimal_p1[i]:.2%}' for i in range(N)))
                print(f"\tOptimal f1:", ' '.join(f'{optimal_f1[i]:.2%}' for i in range(N)))
            p2_list.append(optimal_p1)
            f2_list.append(optimal_f1)
            if verbose:
                print("\tExpected winnings for player 1:", -result.fun)
            if abs(result.fun) < 1e-10:
                break
        else:
            if verbose:
                print("\tOptimization failed:", result.message)
    return (p2_list, f2_list)


if __name__ == '__main__':
    run_example()

