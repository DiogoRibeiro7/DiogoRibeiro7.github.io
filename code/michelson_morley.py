import pandas as pd
import matplotlib.pyplot as plt

# Hypothetical data
data = {
    'Trial': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Measurement (µs)': [0.01, 0.02, -0.01, 0.00, 0.03, -0.02, 0.01, 0.02, 0.00, 0.03]
}

# Create DataFrame
df = pd.DataFrame(data)

# Create Box Plot
plt.figure(figsize=(10, 6))
plt.boxplot(df['Measurement (µs)'])
plt.title('Michelson-Morley Experiment Measurements')
plt.ylabel('Measurement (µs)')
plt.xticks([1], ['Michelson-Morley Data'])
plt.show()
