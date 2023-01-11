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

# Accuracy
#             Expose Antrum Facial_reces ALL
# fold 1 5      [0.85, 0.68, 0.96, 0.83]
# fold 2 12     [0.93, 0.50, 0.91, 0.78]
# fold 4 11     [0.61, 0.78, 0.80, 0.73]
# fold 6 7      [0.82, 0.55, 0.96, 0.78]
# fold 8, 10    [0.92, 0.57, 0.97, 0.82]

# Precision
#             Expose Antrum Facial_reces ALL
# fold 1 5      [0.70, 0.82, 0.99, 0.84]
# fold 2 12     [0.64, 0.85, 0.96, 0.82]
# fold 4 11     [0.61, 0.69, 0.93, 0.74]
# fold 6 7      [0.82, 0.55, 0.96, 0.78]
# fold 8, 10    [0.92, 0.57, 0.97, 0.82]


import numpy as np

acc = np.array([[0.85, 0.68, 0.96, 0.83],
                [0.93, 0.50, 0.91, 0.78],
                [0.61, 0.78, 0.80, 0.73],
                [0.82, 0.55, 0.96, 0.78],
                [0.92, 0.57, 0.97, 0.82]])


prec = np.array([[0.70, 0.82, 0.99, 0.84],
                 [0.64, 0.85, 0.96, 0.82],
                 [0.61, 0.69, 0.93, 0.74],
                 [0.82, 0.55, 0.96, 0.78],
                 [0.92, 0.57, 0.97, 0.82]])


f1 = 2 / (1 / acc + 1 / prec)
f1_avg = np.mean(f1, axis=0)
f1_std = np.std(f1, axis=0)
print(f1_avg, f1_std)

acc_avg = np.mean(acc, axis=0)
acc_std = np.std(acc, axis=0)
print(acc_avg, acc_std)

prec_avg = np.mean(prec, axis=0)
prec_std = np.std(prec, axis=0)
print(prec_avg, prec_std)

result = {'Accuracy' : (acc_avg, f1_std),
          'Precision': (prec_avg, prec_std),
          'F1 Score' : (f1_avg, f1_std)}
 
for metric, (avg, std) in result.items():
    line = f"{metric} "
    for i in range(4):
        line += f'& ${avg[i]:.3f} \\pm {std[i]:.3f}$ '
    line += '\\\\'
    print(line)