import matplotlib.pyplot as plt
import pandas as pd

file_path = './training_metrics.csv'

data = pd.read_csv(file_path)

# Dados fornecidos
epochs = data['epoch'].tolist()
running_loss = data['running_loss'].tolist()

# Criar o gráfico
plt.figure(figsize=(8,6))
plt.plot(epochs, running_loss, marker='o', linestyle='-', color='b')

# Adicionar título e rótulos aos eixos
plt.title("Training Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Running Loss")

# Mostrar o gráfico
plt.grid(True)
plt.show()