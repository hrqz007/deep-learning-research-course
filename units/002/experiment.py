# Unit 002: original synthetic data from 001; standard library only.
# All inputs are finite, aligned numbers; training inputs are unique.
# Public teaching labels illustrate ordering, not a secure blind test.
# 1. Record the interpreter version without exposing a personal file path.
import sys
print("Python", sys.version)

# 2. The same five calibration records as Unit 001.
train_x = [1, 2, 3, 4, 5]
train_y = [3, 5, 8, 9, 11]
candidates = [0, 1, 2]

# 3. Define the two prediction rules and the average absolute error.
def predict_a(x, bias):
    return 2 * x + bias


def mae(actual, predicted):
    n = len(actual)
    if n == 0 or len(predicted) != n:
        raise ValueError("Need nonempty equal-length lists")
    total = 0
    for i in range(n):
        total = total + abs(predicted[i] - actual[i])
    return total / n


def predict_b(known_x, known_y, x):
    for i in range(len(known_x)):
        if x == known_x[i]:
            return known_y[i]
    return 0

# 4. Select the bias using training records only.
# Candidates are increasing, and a strict comparison keeps the smaller tie.
best_bias = candidates[0]
initial_predictions = []
for x in train_x:
    initial_predictions.append(predict_a(x, best_bias))
best_score = mae(train_y, initial_predictions)
candidate_scores = []
for bias in candidates:
    predicted = []
    for x in train_x:
        predicted.append(predict_a(x, bias))
    score = mae(train_y, predicted)
    candidate_scores.append(score)
    print("candidate", bias, "train MAE", score)
    if score < best_score:
        best_score = score
        best_bias = bias

# 5. Record training errors and freeze the test predictions.
a_train = []
b_train = []
for x in train_x:
    a_train.append(predict_a(x, best_bias))
    b_train.append(predict_b(train_x, train_y, x))
a_train_mae = mae(train_y, a_train)
b_train_mae = mae(train_y, b_train)
print("A train MAE", a_train_mae)
print("B train MAE", b_train_mae)

test_x = [1.5, 2.5, 3.5, 4.5]
a_test = []
b_test = []
for x in test_x:
    a_test.append(predict_a(x, best_bias))
    b_test.append(predict_b(train_x, train_y, x))
print("frozen bias", best_bias)
print("A predictions", a_test)
print("B predictions", b_test)

# 6. Only now reveal the public teaching answers. Do not retune the bias.
test_y = [4, 6, 8, 10]
a_test_mae = mae(test_y, a_test)
b_test_mae = mae(test_y, b_test)
print("A test MAE", a_test_mae)
print("B test MAE", b_test_mae)

# 7. The changed relation is 3*x + 1, as in Unit 001.
stress_y = [5.5, 8.5, 11.5, 14.5]
a_stress_mae = mae(stress_y, a_test)
print("A stress MAE", a_stress_mae)
print("test x | label | A | B | stress label | A stress error")
for i in range(len(test_x)):
    print(test_x[i], test_y[i], a_test[i], b_test[i],
          stress_y[i], abs(a_test[i] - stress_y[i]))
