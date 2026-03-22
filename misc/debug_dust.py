import numpy as np

from cerberus.complexity import calculate_dust_score


def random_seq(n):
    return "".join(np.random.choice(["A", "C", "G", "T"], n))


def repeat_seq(n):
    return "A" * n


print(f"Random (Raw): {calculate_dust_score(random_seq(1000), normalize=False)}")
print(f"Random (Norm): {calculate_dust_score(random_seq(1000), normalize=True)}")
print(f"Repeat (Raw): {calculate_dust_score(repeat_seq(1000), normalize=False)}")
print(f"Repeat (Norm): {calculate_dust_score(repeat_seq(1000), normalize=True)}")
