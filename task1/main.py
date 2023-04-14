import argparse
import typing as tp


def parse():
    parser = argparse.ArgumentParser('CLI parser for task1')
    parser.add_argument('A',  nargs="+", type=int, help='Array A')
    return parser.parse_args()


def multiplicate(A: tp.List[int]) -> tp.List[int]:
    if len(A) == 1:
        return A
    elif len(A) == 2:
        return [A[1], A[0]]
    else:
        num_zeros: int = 0
        total_mul: int = 1
        zero_index: int = -1
        A_transformed = []
        len_a = len(A)

        for i in range(len_a):
            if A[i] == 0:
                num_zeros += 1
                zero_index = i
            else:
                total_mul = total_mul * A[i]

        if num_zeros == 1:
            A_transformed = [0] * len_a
            A_transformed[zero_index] = total_mul
            return A_transformed

        elif num_zeros >= 2:
            return [0] * len_a

        else:
            for i in range(len_a):
                if i == zero_index:
                    A_transformed.append(total_mul)
                else:
                    A_transformed.append(total_mul // A[i])
            return A_transformed


if __name__ == '__main__':
    args = parse()
    print(f'Initial array: {args.A}')
    transformed_array = multiplicate(args.A)
    print(f'Transformed array: {transformed_array}')
