import argparse

import numpy as np

import torch_ref
from data import get_data
from model import init_params, loss_and_grads
from numeric_check import CHECKED, TOL, label, numeric_grad

LOSS_TOL = 1e-12
GRAD_TOL = 1e-12
NAMES = ["W1", "b1", "W2", "b2"]


def compare_with_torch(loss_np, grads_np, loss_pt, grads_pt):
    rows = [("Втрата", abs(loss_np - loss_pt), LOSS_TOL)]
    for name in NAMES:
        diff = np.max(np.abs(grads_np[name] - grads_pt[name]))
        rows.append((f"Градієнт {name}", float(diff), GRAD_TOL))
    return rows


def print_torch_table(rows):
    print(f"{'Величина':<16}{'Макс. абс. різниця':>22}   Перевірку пройдено")
    for name, diff, tol in rows:
        print(f"{name:<16}{diff:>22.3e}   {'так' if diff <= tol else 'ні'}")


def print_numeric_table(params, X, y, grads):
    header = (
        f"{'Параметр':<12}{'backward()':>16}{'Чисельна похідна':>20}"
        f"{'Абс. різниця':>16}   Перевірку пройдено"
    )
    print(header)
    for name, index in CHECKED:
        manual = float(grads[name][index])
        numeric = numeric_grad(params, X, y, name, index)
        diff = abs(numeric - manual)
        print(
            f"{label(name, index):<12}{manual:>16.9f}{numeric:>20.9f}"
            f"{diff:>16.3e}   {'так' if diff <= TOL else 'ні'}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bug",
        action="store_true",
        help="прибрати ділення на кількість обʼєктів у градієнті за логітами",
    )
    args = parser.parse_args()
    divide_by_n = not args.bug

    X_train, y_train, X_test, y_test = get_data()
    params = init_params()

    mode = "з навмисною помилкою" if args.bug else "правильна"
    print(f"Реалізація: {mode}")
    print(f"Навчальна вибірка: {X_train.shape}, тестова: {X_test.shape}")
    print(f"Обʼєктів у класах (навчальна): {np.bincount(y_train)}")
    print()

    loss_np, grads_np = loss_and_grads(
        params, X_train, y_train, divide_by_n=divide_by_n
    )
    loss_pt, grads_pt = torch_ref.loss_and_grads(params, X_train, y_train)

    print(f"Втрата NumPy:   {loss_np:.15f}")
    print(f"Втрата PyTorch: {loss_pt:.15f}")
    print()

    print("Звірка з PyTorch")
    print_torch_table(compare_with_torch(loss_np, grads_np, loss_pt, grads_pt))
    print()

    print("Чисельна перевірка градієнтів")
    print_numeric_table(params, X_train, y_train, grads_np)


if __name__ == "__main__":
    main()
