# Accuracy
#             Expose Antrum Facial_recess ALL
# fold 1 5      [0.93, 0.62, 0.93, 0.83]
# fold 2 4      [0.62, 0.91, 0.59, 0.71]
# fold 6 7      [0.89, 0.64, 0.85, 0.80]
# fold 8, 10    [0.82, 0.55, 0.96, 0.78]
# fold 11, 12   [0.92, 0.57, 0.97, 0.82]

# Precision
#             Expose Antrum Facial_recess ALL
# fold 1 5      [0.68, 0.90, 0.99, 0.86]
# fold 2 4      [0.64, 0.64, 0.98, 0.75]
# fold 6 7      [0.64, 0.93, 0.93, 0.83]
# fold 8, 10    [0.64, 0.93, 0.86, 0.81]
# fold 11, 12   [0.69, 0.95, 0.92, 0.85]

import numpy as np

acc = np.array([[0.93, 0.62, 0.93, 0.83],
                [0.62, 0.91, 0.59, 0.71],
                [0.89, 0.64, 0.85, 0.80],
                [0.82, 0.55, 0.96, 0.78],
                [0.92, 0.57, 0.97, 0.82]])

acc_avg = np.mean(acc, axis=0)
acc_std = np.std(acc, axis=0)
print(acc_avg, acc_std)

prec = np.array([[0.68, 0.90, 0.99, 0.86],
                 [0.64, 0.64, 0.98, 0.75],
                 [0.64, 0.93, 0.93, 0.83],
                 [0.64, 0.93, 0.86, 0.81],
                 [0.69, 0.95, 0.92, 0.85]])

prec_avg = np.mean(prec, axis=0)
prec_std = np.std(prec, axis=0)
print(prec_avg, prec_std)
