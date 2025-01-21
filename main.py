import numpy as np
from scipy.linalg import lu, qr
import matplotlib.pyplot as plt
import pandas as pd
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Função para exibir matriz com formato legível
def print_matrix(name, matrix):
    print(f"\n{name}:")
    print(matrix)

# Definição do sistema linear Ax = b
A = np.array([[4, -2, 1],
              [3, 6, -4],
              [2, 1, 8]])
b = np.array([12, -25, 32])

# Verificando a determinante de A
det_A = np.linalg.det(A)

# Resolução com decomposição LU
P, L, U = lu(A)
Pb = np.dot(P, b)
y = np.linalg.solve(L, Pb)
x_lu = np.linalg.solve(U, y)

# Resolução com decomposição QR
Q, R = qr(A)
Qt_b = np.dot(Q.T, b)
x_qr = np.linalg.solve(R, Qt_b)

diff = np.abs(x_lu - x_qr)

# Criando uma tabela para exibição dos dados e valores
data = {
    "Descrição": [
        "Determinante de A", 
        "Pb (Permutação de b)", 
        "y (L * y = Pb)", 
        "x (LU)", 
        "x (QR)", 
        "Diferença (LU vs QR)"
    ],
    "Valores": [
        det_A,
        Pb.tolist(),
        y.tolist(),
        x_lu.tolist(),
        x_qr.tolist(),
        diff.tolist()
    ]
}
df = pd.DataFrame(data)

# Função para exibir tabela e gráfico em janelas interativas
def create_interactive_interface():
    def show_table():
        clear_window()
        menu_frame.grid(row=0, column=0, sticky=(tk.W, tk.E))

        frame = ttk.Frame(window, padding="10")
        frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Treeview",
                        background="#F9F9F9",
                        foreground="black",
                        rowheight=30,
                        fieldbackground="#F9F9F9",
                        font=("Arial", 12))
        style.configure("Treeview.Heading",
                        font=("Arial", 14, "bold"),
                        background="#4CAF50",
                        foreground="white",
                        padding=5)

        columns = ("Descrição", "Valores")
        tree = ttk.Treeview(frame, columns=columns, show="headings")
        tree.heading("Descrição", text="Descrição", anchor="center")
        tree.heading("Valores", text="Valores", anchor="center")
        tree.column("Descrição", width=400, anchor="center")
        tree.column("Valores", width=600, anchor="center")

        for index, row in df.iterrows():
            tree.insert("", tk.END, values=(row["Descrição"], row["Valores"]))

        tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscroll=scrollbar.set)
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))

    def show_plot():
        clear_window()
        menu_frame.grid(row=0, column=0, sticky=(tk.W, tk.E))

        frame = ttk.Frame(window, padding="10")
        frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Ajustando o tamanho e margens do gráfico
        fig, ax = plt.subplots(figsize=(10, 6))  # Reduzido o tamanho do gráfico

        # Adicionando barras para LU e QR com cores diferentes
        indices = np.arange(len(diff))
        width = 0.4  # Largura das barras

        bars_lu = ax.bar(indices - width / 2, x_lu, width, label="Solução LU", color='#4CAF50', edgecolor='black')
        bars_qr = ax.bar(indices + width / 2, x_qr, width, label="Solução QR", color='#2196F3', edgecolor='black')

        # Adicionando valores acima das barras
        for bar in bars_lu:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height + 0.01, f'{height:.4f}', ha='center', va='bottom', fontsize=10, color='black')

        for bar in bars_qr:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height + 0.01, f'{height:.4f}', ha='center', va='bottom', fontsize=10, color='black')

        # Configurações do gráfico
        ax.set_title("Comparação entre soluções LU e QR", fontsize=16, fontweight='bold', color='#333333')
        ax.set_ylabel("Valores de x", fontsize=12, color='#333333')
        ax.set_xlabel("Componentes de x", fontsize=12, color='#333333')
        ax.set_xticks(indices)
        ax.set_xticklabels(["x1", "x2", "x3"], fontsize=10, color='#333333')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(visible=True, axis='y', linestyle='--', alpha=0.6)
        ax.tick_params(axis='y', labelsize=10, colors='#333333')
        ax.tick_params(axis='x', labelsize=10, colors='#333333')

        # Adicionando a legenda
        ax.legend(loc="upper left", fontsize=10, frameon=False)

        # Ajustando manualmente as margens
        plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.2)

        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.get_tk_widget().grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        canvas.draw()

    def clear_window():
        for widget in window.winfo_children():
            if widget != menu_frame:
                widget.destroy()

    window = tk.Tk()
    window.title("Interface Interativa")
    window.state('zoomed')

    menu_frame = ttk.Frame(window, padding="10")
    menu_frame.grid(row=0, column=0, sticky=(tk.W, tk.E))

    table_button = ttk.Button(menu_frame, text="Mostrar Tabela", command=show_table)
    table_button.grid(row=0, column=0, padx=10, pady=5)

    plot_button = ttk.Button(menu_frame, text="Mostrar Gráfico", command=show_plot)
    plot_button.grid(row=0, column=1, padx=10, pady=5)

    show_table()

    window.mainloop()

# Iniciar interface interativa
create_interactive_interface()
