# Intentional logic error: reset the accumulated value in every iteration.
actual = [3, 5, 8, 9, 11]
predicted = [3, 5, 7, 9, 11]
for i in range(len(actual)):
    total = 0
    total = total + abs(predicted[i] - actual[i])
print(total / len(actual))
