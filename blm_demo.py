import blm
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

x = np.linspace(0, 1, 100)
y = 1 + 2*x + np.random.normal(0, 0.2, 100)
df = pd.DataFrame({'x': x, 'y': y})
plt.scatter(df['x'], df['y'])

m = blm.LinearRegression.from_formula('y ~ x', df)
m.sample(500, 470)
y_pred = m.predict_sample(df)
for y_sample in y_pred:
	plt.plot(x, y_sample, color='blue', alpha=.1)
plt.show()
