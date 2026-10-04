# Intentional logic error: return after just the first item.
def broken_mae(actual, predicted):
    total = 0
    for i in range(len(actual)):
        total = total + abs(predicted[i] - actual[i])
        return total / len(actual)

print(broken_mae([3, 5, 8, 9, 11], [3, 5, 7, 9, 11]))
