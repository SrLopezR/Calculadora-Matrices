import threading
import time
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from fraccion import Fraccion
from gauss import GaussJordanEngine,GaussEngine
from matrices import (
    sumar_matrices, multiplicar_matrices,
    multiplicar_escalar_matriz, formatear_matriz, Transpuesta, determinante_matriz,
    determinante_cofactores)
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import re


TEXT_BG = "#1E1E1E"
TEXT_FG = "#FFFFFF"
TEXT_FONT = ("Cascadia Code", 10)

def make_text(parent, height=12, wrap="word", font=TEXT_FONT):
    return tk.Text(parent,
                   background=TEXT_BG,
                   foreground=TEXT_FG,
                   insertbackground=TEXT_FG,
                   font=font,
                   height=height,
                   wrap=wrap)
def configurar_estilo_oscuro(root):
    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except:
        pass

    # Colores base
    fondo = "#1e1e1e"
    texto = "#ffffff"
    acento = "#4fc3f7"
    borde = "#5c5f61"

    # Fuente global
    fuente = ("Times New Roman", 10)

    root.configure(bg=fondo)

    # General
    style.configure(".", background=fondo, foreground=texto, font=fuente)
    style.configure("TLabel", background=fondo, foreground=texto)
    style.configure("TFrame", background=fondo)
    style.configure("TLabelframe", background=fondo, foreground=texto, bordercolor=borde)
    style.configure("TLabelframe.Label", background=fondo, foreground=acento)
    style.configure("TButton", background=fondo, foreground=texto, padding=6, font=("Times New Roman", 10), borderwidth=1 , bordercolor = borde)
    style.map("TButton",
              background=[("active", acento), ("pressed", texto)],
              foreground=[("active", "#000000")])
    # --- Treeview (tablas de matrices) ---
    style.configure("Treeview",
                    background="#1E1E1E",      # fondo normal oscuro
                    fieldbackground="#1E1E1E",
                    foreground="#FFFFFF",       # texto blanco normal
                    font=("Cascadia Code", 10))
    style.map("Treeview",
              background=[("selected", "#2563EB")],  # azul de selección
              foreground=[("selected", "#000000")])  # texto negro al seleccionar

    # --- Card.TLabelframe (secciones de resultados y pasos) ---
    style.configure("Card.TLabelframe",
                    background="#1E1E1E",  # fondo oscuro
                    bordercolor="#3C3C3C",  # borde gris oscuro
                    relief="solid")

    style.configure("Card.TLabelframe.Label",
                    background="#1E1E1E",  # fondo oscuro del título
                    foreground="#FFFFFF",  # texto blanco
                    font=("Cascadia Code", 10, "bold"))

    # Campos de entrada
    style.configure("TEntry", fieldbackground="#1e1e1e", foreground=texto, insertcolor=texto)
    style.configure("TSpinbox", fieldbackground="#1e1e1e", foreground=texto, insertcolor=texto)
    style.configure("TNotebook", background=fondo, tabmargins=[2, 5, 2, 0])
    style.configure("TNotebook.Tab", background=fondo, foreground=texto, padding=[8, 4])
    style.map("TNotebook.Tab", background=[("selected", acento)], foreground=[("selected", "#000000")])

    # Checkbuttons y radiobuttons
    style.configure("TCheckbutton", background=fondo, foreground=texto, font=fuente)
    style.configure("TRadiobutton", background=fondo, foreground=texto, font=fuente)

# ------------------ Widgets reutilizables ------------------

class MatrixInput(ttk.Frame):
    """Cuadrícula de entradas para matriz (con columna b opcional)."""
    def __init__(self, master, rows=3, cols=3, allow_b=True, **kw):
        super().__init__(master, **kw)
        self.rows, self.cols = rows, cols
        self.allow_b = allow_b
        self.entries = []
        self._build()

    def _build(self):
        # Encabezados
        for j in range(self.cols):
            ttk.Label(self, text=f"x{j+1}", anchor="center").grid(row=0, column=j, padx=4, pady=(0,6))
        if self.allow_b:
            ttk.Label(self, text="|", width=2).grid(row=0, column=self.cols)
            ttk.Label(self, text="b", anchor="center").grid(row=0, column=self.cols+1, padx=4, pady=(0,6))

        for i in range(self.rows):
            fila = []
            for j in range(self.cols + (1 if self.allow_b else 0)):
                e = ttk.Entry(self, width=9, justify="center")
                col = j if j < self.cols else j + 1
                e.grid(row=i+1, column=col, padx=3, pady=3)
                fila.append(e)
            self.entries.append(fila)

            if self.allow_b:
                ttk.Label(self, text="|", width=2).grid(row=i+1, column=self.cols)

    def set_size(self, rows, cols):
        for w in list(self.winfo_children()):
            w.destroy()
        self.rows, self.cols = rows, cols
        self.entries = []
        self._build()

    def clear(self):
        for fila in self.entries:
            for e in fila:
                e.delete(0, tk.END)

    def get_matrix(self):
        M = []
        for i in range(self.rows):
            fila = []
            for j in range(len(self.entries[i])):
                t = self.entries[i][j].get().strip() or "0"
                try:
                    fila.append(Fraccion(t))
                except Exception:
                    raise ValueError(f"Valor inválido en fila {i+1}, columna {j+1}: '{t}'")
            M.append(fila)
        return M


class MatrixView(ttk.Frame):
    """Vista de matriz tipo tabla con resaltado de fila pivote."""
    def __init__(self, master, **kw):
        super().__init__(master, **kw)
        self.tree = ttk.Treeview(self, show="headings", height=10)
        self.tree.pack(fill="both", expand=True)
        self._cols = 0

    def set_matrix(self, M):
        if not M: return
        rows, cols = len(M), len(M[0])
        if cols != self._cols:
            self.tree.delete(*self.tree.get_children())
            self.tree["columns"] = [f"c{j}" for j in range(cols)]
            for j in range(cols):
                txt = f"x{j+1}" if j < cols-1 else "b"
                self.tree.heading(f"c{j}", text=txt)
                self.tree.column(f"c{j}", width=80, anchor="center")
            self._cols = cols
        self.tree.delete(*self.tree.get_children())
        for i in range(rows):
            self.tree.insert("", "end", values=[str(x) for x in M[i]])

    def highlight(self, row=None, col=None):
        style = ttk.Style(self)
        style.map("Treeview", background=[("selected", "#0094f7")])
        if row is not None:
            try:
                iid = self.tree.get_children()[row]
                self.tree.selection_set(iid)
                self.tree.see(iid)
            except IndexError:
                pass


# ------------------ App con pestañas (controles locales) ------------------

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.root = None
        self.title("Soluciones de Álgebra Lineal")
        self.geometry("1100x720")
        self.minsize(1000, 640)
        configurar_estilo_oscuro(self)

        self.engine = None
        self.auto_running = False
        self.auto_thread = None
        self.btn_auto = None  # se crea dentro de la pestaña Gauss
         # Motor exclusivo para la pestaña de Gauss (el método sencillo)
        self.engine_gauss = None
        self.auto_running_gauss = False
        self.auto_thread_gauss = None
        self.btn_auto_gauss = None

        self._build_ui()

    # -------- Estilos (solo UI) --------
    def _setup_style(self):
        style = ttk.Style(self)
        try: style.theme_use("equilux")
        except: pass
        primary = "#2563EB"
        surface = "#F5F7FB"
        border  = "#E3E8F0"

        style.configure(".", font=("Times New Roman", 10))
        style.configure("TFrame", background=surface)
        style.configure("Toolbar.TFrame")
        style.configure("Card.TLabelframe", bordercolor=border)
        style.configure("Card.TLabelframe.Label", foreground="#334155",
                        font=("Times New Roman", 10))
        style.configure("Title.TLabel", font=("Times New Roman", 12))
        style.configure("Status.TLabel", background="#2563EB", anchor="w")
        style.configure("Accent.TButton", padding=6)
        style.map("Accent.TButton",
                  background=[("!disabled", primary), ("pressed", "#2563EB")],
                  foreground=[("!disabled", "#ffffff")])

    # -------- Construcción general --------
    def _build_ui(self):
        # Encabezado minimal (sin controles globales de Gauss)
        header = ttk.Frame(self, style="Toolbar.TFrame")
        header.pack(fill="x", padx=10, pady=8)
        ttk.Label(header, text="Soluciones de Álgebra Lineal", font=("Times New Roman", 12)).pack(side="left")

        # Notebook de pestañas
        self.nb = ttk.Notebook(self)
        self.nb.pack(fill="both", expand=True, padx=10, pady=(0,8))

        # Pestañas
        self._tab_gauss()
        self._tab_gauss_simple() 
        self._tab_suma()
        self._tab_mult()
        self._tab_escalar()
        self._tab_transpuesta()
        self._tab_independencia()
        self._tab_inversa()
        self._tab_determinante()
        self._tab_cramer()
        self._tab_sarrus()
        self._tab_metodos_numericos()
        self._tab_falsa_posicion()
        self._tab_newton_raphson()


        # Resultado general + estado
        result_card = ttk.Labelframe(self, text="Resultado", style="Card.TLabelframe", padding=8)
        result_card.pack(fill="x", padx=10, pady=(0,8))
        self.lbl_result = ttk.Label(result_card, text="—", style="Title.TLabel")
        self.lbl_result.pack(anchor="w")

        self.status = ttk.Label(self, text="Listo.", style="Status.TLabel")
        self.status.pack(fill="x", side="bottom")

    def resize_matrix(self):
        try:
            m, n = int(self.spin_m.get()), int(self.spin_n.get())
            if m <= 0 or n <= 0: raise ValueError
        except:
            messagebox.showerror("Error", "Dimensiones inválidas"); return
        self.matrix_input.set_size(m, n)

    def load_example(self):
        ejemplo = [["1","1","1","6"], ["2","-1","1","3"], ["1","2","-1","3"]]
        self.matrix_input.set_size(3,3)
        for i in range(3):
            for j in range(4):
                self.matrix_input.entries[i][j].delete(0, tk.END)
                self.matrix_input.entries[i][j].insert(0, ejemplo[i][j])

    def clear_inputs(self):
        self.matrix_input.clear()
        self.txt_log.delete(1.0, tk.END)
        self.lbl_result.config(text="—")
        if hasattr(self, "txt_solution"):
            self.txt_solution.delete(1.0, tk.END)

    def start_engine(self):
        try:
            A = self.matrix_input.get_matrix()
        except Exception as e:
            messagebox.showerror("Entrada inválida", str(e)); return
        self.engine = GaussJordanEngine(A)
        self._render_last_step()
        self._log("Inicializado. Use 'Siguiente paso' o 'Reproducir'.")
        self._update_status("Listo para ejecutar.")
        self.lbl_result.config(text="—")
        if hasattr(self, "txt_solution"):
            self.txt_solution.delete(1.0, tk.END)

    def next_step(self):
        if not self.engine:
            messagebox.showinfo("Información", "Primero presione 'Resolver (inicializar)'"); return
        step = self.engine.siguiente()
        if step is None:
            self._log("No hay más pasos.")
        self._render_last_step()
        if self.engine.terminado:
            self._show_result()

    def toggle_auto(self):
        if not self.engine:
            messagebox.showinfo("Información", "Primero presione 'Resolver (inicializar)'"); return
        if not self.auto_running:
            self.auto_running = True
            if self.btn_auto: self.btn_auto.config(text="Pausar")
            self.auto_thread = threading.Thread(target=self._auto_run, daemon=True)
            self.auto_thread.start()
        else:
            self.auto_running = False
            if self.btn_auto: self.btn_auto.config(text="Reproducir")

    def _auto_run(self):
        while self.auto_running and self.engine and not self.engine.terminado:
            self.next_step()
            time.sleep(1.0)
        self.auto_running = False
        if self.btn_auto: self.btn_auto.config(text="Reproducir")

    def reset(self):
        self.engine = None
        self.txt_log.delete(1.0, tk.END)
        self.matrix_view.set_matrix([[Fraccion(0)]])
        self.lbl_result.config(text="—")
        if hasattr(self, "txt_solution"):
            self.txt_solution.delete(1.0, tk.END)
        self._update_status("Reiniciado.")

    def export_log(self):
        if not self.engine or not self.engine.log:
            messagebox.showinfo("Información", "No hay pasos para exportar"); return
        fp = filedialog.asksaveasfilename(defaultextension=".txt",
                                          filetypes=[("Texto","*.txt")],
                                          title="Guardar registro de pasos")
        if not fp: return
        with open(fp, "w", encoding="utf-8") as f:
            for i, s in enumerate(self.engine.log, start=1):
                f.write(f"Paso {i}: {s.descripcion}\n")
                for fila in s.matriz:
                    f.write(" [ " + " ".join(str(x) for x in fila[:-1]) + " | " + str(fila[-1]) + " ]\n")
                f.write("\n")
        messagebox.showinfo("Listo", f"Registro exportado a: {fp}")

    def _render_last_step(self):
        if not self.engine or not self.engine.log: return
        step = self.engine.log[-1]
        self.matrix_view.set_matrix(step.matriz)
        self.matrix_view.highlight(step.pivote_row, step.pivote_col)
        self._log(step.descripcion)

    def _log(self, text):
        self.txt_log.insert(tk.END, text + "\n")
        self.txt_log.see(tk.END)

    def _show_result(self):
        if not self.engine:
         return

        # Analizar el sistema con el motor de Gauss-Jordan
        resultado = self.engine.analizar()
        tipo, info, detalles = resultado

        # ---- Barra de resultado (abajo, una sola línea) ----
        if tipo == "inconsistente":
            resumen = "El sistema es inconsistente. No tiene solución."
        elif tipo == "única":
            resumen = "Sistema consistente con solución única."
        else:
            resumen = "Sistema consistente con infinitas soluciones."

        self.lbl_result.config(text=resumen)

        # ---- Cuadro de salida detallada (como la imagen de la derecha) ----
        msg = self.engine.conjunto_solucion(resultado)
        if hasattr(self, "txt_solution"):
            self.txt_solution.delete(1.0, tk.END)
            self.txt_solution.insert(tk.END, msg)

        self._update_status("Cálculo finalizado.")

    def _update_status(self, s):
        self.status.config(text=s)

    # -------- Tab 1: Gauss-Jordan --------
    def _tab_gauss(self):
        tab = ttk.Frame(self.nb)
        self.nb.add(tab, text="Gauss-Jordan")

        # Controles superiores de tamaño y utilidades
        top = ttk.Frame(tab); top.pack(fill="x", padx=8, pady=(8,4))
        ttk.Label(top, text="Ecuaciones (m):").pack(side="left", padx=(2,4))
        self.spin_m = tk.Spinbox(top, from_=1, to=12, width=4)
        self.spin_m.delete(0,"end"); self.spin_m.insert(0,"3"); self.spin_m.pack(side="left")
        ttk.Label(top, text="Incógnitas (n):").pack(side="left", padx=(10,4))
        self.spin_n = tk.Spinbox(top, from_=1, to=12, width=4)
        self.spin_n.delete(0,"end"); self.spin_n.insert(0,"3"); self.spin_n.pack(side="left", padx=(0,8))
        ttk.Button(top, text="Redimensionar", command=self.resize_matrix).pack(side="left", padx=(0,6))
        ttk.Button(top, text="Ejemplo", command=self.load_example).pack(side="left", padx=3)
        ttk.Button(top, text="Limpiar", command=self.clear_inputs).pack(side="left", padx=3)

        # Controles locales de Gauss-Jordan (LOCALES A ESTA PESTAÑA)
        ctrls = ttk.Frame(tab); ctrls.pack(fill="x", padx=8, pady=(0,8))
        ttk.Button(ctrls, text="Resolver (inicializar)", style="Accent.TButton",
                   command=self.start_engine).pack(side="left", padx=(0,6))
        ttk.Button(ctrls, text="Siguiente paso", command=self.next_step).pack(side="left", padx=3)
        self.btn_auto = ttk.Button(ctrls, text="Reproducir", command=self.toggle_auto)
        self.btn_auto.pack(side="left", padx=3)
        ttk.Button(ctrls, text="Reiniciar", command=self.reset).pack(side="left", padx=3)
        ttk.Button(ctrls, text="Exportar pasos", command=self.export_log).pack(side="left", padx=3)

        # Cuerpo: entrada matriz aumentada + vista y pasos
        body = ttk.Panedwindow(tab, orient="horizontal")
        body.pack(fill="both", expand=True, padx=8, pady=8)

        left = ttk.Labelframe(body, text="Matriz aumentada (coeficientes | términos independientes)",
                              style="Card.TLabelframe", padding=8)
        self.matrix_input = MatrixInput(left, rows=3, cols=3, allow_b=True)
        self.matrix_input.pack(fill="x")
        body.add(left, weight=1)

        right = ttk.Frame(body)
        body.add(right, weight=2)

        sub = ttk.Panedwindow(right, orient="horizontal")
        sub.pack(fill="both", expand=True)
        boxA = ttk.Labelframe(sub, text="Matriz y pivotes", style="Card.TLabelframe", padding=6)
        self.matrix_view = MatrixView(boxA); self.matrix_view.pack(fill="both", expand=True)
        sub.add(boxA, weight=1)

        boxB = ttk.Labelframe(sub, text="Pasos / Operaciones", style="Card.TLabelframe", padding=6)
        self.txt_log = make_text(boxB, height=14, wrap="word"); self.txt_log.pack(fill="both", expand=True)
        sub.add(boxB, weight=1)

        # --- Cuadro de salida del sistema (como la imagen de la derecha) ---
        solution_card = ttk.Labelframe(
            right,
            text="Resultados del sistema",
            style="Card.TLabelframe",
            padding=6
        )
        self.txt_solution = make_text(solution_card, height=6, wrap="word")
        self.txt_solution.pack(fill="both", expand=True)
        solution_card.pack(fill="x", pady=(8, 0))
        
            # -------- Tab 2: Gauss (método sencillo, solo hacia adelante) --------
    def _tab_gauss_simple(self):
        tab = ttk.Frame(self.nb)
        self.nb.add(tab, text="Gauss")

        # Controles superiores de tamaño y utilidades
        top = ttk.Frame(tab); top.pack(fill="x", padx=8, pady=(8,4))
        ttk.Label(top, text="Ecuaciones (m):").pack(side="left", padx=(2,4))
        self.spin_m_gauss = tk.Spinbox(top, from_=1, to=12, width=4)
        self.spin_m_gauss.delete(0,"end"); self.spin_m_gauss.insert(0,"3"); self.spin_m_gauss.pack(side="left")
        ttk.Label(top, text="Incógnitas (n):").pack(side="left", padx=(10,4))
        self.spin_n_gauss = tk.Spinbox(top, from_=1, to=12, width=4)
        self.spin_n_gauss.delete(0,"end"); self.spin_n_gauss.insert(0,"3"); self.spin_n_gauss.pack(side="left", padx=(0,8))
        ttk.Button(top, text="Redimensionar", command=self.resize_matrix_gauss).pack(side="left", padx=(0,6))
        ttk.Button(top, text="Ejemplo", command=self.load_example_gauss).pack(side="left", padx=3)
        ttk.Button(top, text="Limpiar", command=self.clear_inputs_gauss).pack(side="left", padx=3)

        # Controles locales de Gauss
        ctrls = ttk.Frame(tab); ctrls.pack(fill="x", padx=8, pady=(0,8))
        ttk.Button(ctrls, text="Resolver (inicializar)", style="Accent.TButton",
                   command=self.start_engine_gauss).pack(side="left", padx=(0,6))
        ttk.Button(ctrls, text="Siguiente paso", command=self.next_step_gauss).pack(side="left", padx=3)
        self.btn_auto_gauss = ttk.Button(ctrls, text="Reproducir", command=self.toggle_auto_gauss)
        self.btn_auto_gauss.pack(side="left", padx=3)
        ttk.Button(ctrls, text="Reiniciar", command=self.reset_gauss).pack(side="left", padx=3)
        ttk.Button(ctrls, text="Exportar pasos", command=self.export_log_gauss).pack(side="left", padx=3)

        # Cuerpo: entrada matriz aumentada + vista y pasos
        body = ttk.Panedwindow(tab, orient="horizontal")
        body.pack(fill="both", expand=True, padx=8, pady=8)

        left = ttk.Labelframe(body, text="Matriz aumentada (coeficientes | términos independientes)",
                              style="Card.TLabelframe", padding=8)
        self.matrix_input_gauss = MatrixInput(left, rows=3, cols=3, allow_b=True)
        self.matrix_input_gauss.pack(fill="x")
        body.add(left, weight=1)

        right = ttk.Frame(body)
        body.add(right, weight=2)

        sub = ttk.Panedwindow(right, orient="horizontal")
        sub.pack(fill="both", expand=True)
        boxA = ttk.Labelframe(sub, text="Matriz y pivotes (Gauss)", style="Card.TLabelframe", padding=6)
        self.matrix_view_gauss = MatrixView(boxA); self.matrix_view_gauss.pack(fill="both", expand=True)
        sub.add(boxA, weight=1)

        boxB = ttk.Labelframe(sub, text="Pasos / Operaciones", style="Card.TLabelframe", padding=6)
        self.txt_log_gauss = make_text(boxB, height=14, wrap="word"); self.txt_log_gauss.pack(fill="both", expand=True)
        sub.add(boxB, weight=1)

        # --- Cuadro de salida del sistema ---
        solution_card = ttk.Labelframe(
            right,
            text="Resultados del sistema",
            style="Card.TLabelframe",
            padding=6
        )
        self.txt_solution_gauss = make_text(solution_card, height=6, wrap="word")
        self.txt_solution_gauss.pack(fill="both", expand=True)
        solution_card.pack(fill="x", pady=(8, 0))

    # ------ Lógica interna de la pestaña Gauss ------

    def resize_matrix_gauss(self):
        try:
            m, n = int(self.spin_m_gauss.get()), int(self.spin_n_gauss.get())
            if m <= 0 or n <= 0:
                raise ValueError
        except:
            messagebox.showerror("Error", "Dimensiones inválidas"); return
        self.matrix_input_gauss.set_size(m, n)

    def load_example_gauss(self):
        ejemplo = [["1","1","1","6"], ["2","-1","1","3"], ["1","2","-1","3"]]
        self.matrix_input_gauss.set_size(3,3)
        for i in range(3):
            for j in range(4):
                self.matrix_input_gauss.entries[i][j].delete(0, tk.END)
                self.matrix_input_gauss.entries[i][j].insert(0, ejemplo[i][j])

    def clear_inputs_gauss(self):
        self.matrix_input_gauss.clear()
        self.txt_log_gauss.delete(1.0, tk.END)
        self.lbl_result.config(text="—")
        self.txt_solution_gauss.delete(1.0, tk.END)

    def start_engine_gauss(self):
        try:
            A = self.matrix_input_gauss.get_matrix()
        except Exception as e:
            messagebox.showerror("Entrada inválida", str(e)); return
        self.engine_gauss = GaussEngine(A)
        self._render_last_step_gauss()
        self._log_gauss("Inicializado. Use 'Siguiente paso' o 'Reproducir'.")
        self._update_status("Gauss listo para ejecutar.")
        self.lbl_result.config(text="—")
        self.txt_solution_gauss.delete(1.0, tk.END)

    def next_step_gauss(self):
        if not self.engine_gauss:
            messagebox.showinfo("Información", "Primero presione 'Resolver (inicializar)'"); return
        step = self.engine_gauss.siguiente()
        if step is None:
            self._log_gauss("No hay más pasos.")
        self._render_last_step_gauss()
        if self.engine_gauss.terminado:
            self._show_result_gauss()

    def toggle_auto_gauss(self):
        if not self.engine_gauss:
            messagebox.showinfo("Información", "Primero presione 'Resolver (inicializar)'"); return
        if not self.auto_running_gauss:
            self.auto_running_gauss = True
            if self.btn_auto_gauss: self.btn_auto_gauss.config(text="Pausar")
            self.auto_thread_gauss = threading.Thread(target=self._auto_run_gauss, daemon=True)
            self.auto_thread_gauss.start()
        else:
            self.auto_running_gauss = False
            if self.btn_auto_gauss: self.btn_auto_gauss.config(text="Reproducir")

    def _auto_run_gauss(self):
        while self.auto_running_gauss and self.engine_gauss and not self.engine_gauss.terminado:
            self.next_step_gauss()
            time.sleep(1.0)
        self.auto_running_gauss = False
        if self.btn_auto_gauss: self.btn_auto_gauss.config(text="Reproducir")

    def reset_gauss(self):
        self.engine_gauss = None
        self.txt_log_gauss.delete(1.0, tk.END)
        self.matrix_view_gauss.set_matrix([[Fraccion(0)]])
        self.lbl_result.config(text="—")
        self.txt_solution_gauss.delete(1.0, tk.END)
        self._update_status("Gauss reiniciado.")

    def export_log_gauss(self):
        if not self.engine_gauss or not self.engine_gauss.log:
            messagebox.showinfo("Información", "No hay pasos para exportar"); return
        fp = filedialog.asksaveasfilename(defaultextension=".txt",
                                          filetypes=[("Texto","*.txt")],
                                          title="Guardar registro de pasos (Gauss)")
        if not fp: return
        with open(fp, "w", encoding="utf-8") as f:
            for i, s in enumerate(self.engine_gauss.log, start=1):
                f.write(f"Paso {i}: {s.descripcion}\n")
                for fila in s.matriz:
                    f.write(" [ " + " ".join(str(x) for x in fila[:-1]) + " | " + str(fila[-1]) + " ]\n")
                f.write("\n")
        messagebox.showinfo("Listo", f"Registro exportado a: {fp}")

    def _render_last_step_gauss(self):
        if not self.engine_gauss or not self.engine_gauss.log: return
        step = self.engine_gauss.log[-1]
        self.matrix_view_gauss.set_matrix(step.matriz)
        self.matrix_view_gauss.highlight(step.pivote_row, step.pivote_col)
        self._log_gauss(step.descripcion)

    def _log_gauss(self, text):
        self.txt_log_gauss.insert(tk.END, text + "\n")
        self.txt_log_gauss.see(tk.END)

    def _show_result_gauss(self):
        if not self.engine_gauss:
            return
        resultado = self.engine_gauss.analizar()
        tipo, info, detalles = resultado

        # Barra inferior: mismo estilo que Gauss-Jordan
        if tipo == "inconsistente":
            resumen = "El sistema es inconsistente. No tiene solución."
        elif tipo == "única":
            resumen = "Sistema consistente con solución única."
        else:
            resumen = "Sistema consistente con infinitas soluciones."
        self.lbl_result.config(text=resumen)

        # Cuadro de resultados de la pestaña Gauss
        msg = self.engine_gauss.conjunto_solucion(resultado)
        self.txt_solution_gauss.delete(1.0, tk.END)
        self.txt_solution_gauss.insert(tk.END, msg)

        self._update_status("Cálculo finalizado (Gauss).")

    # -------- Tab 2: Suma de matrices --------
    def _tab_suma(self):
        tab = ttk.Frame(self.nb)
        self.nb.add(tab, text="Suma")

        frame = ttk.Frame(tab, padding=10); frame.pack(fill="both", expand=True)
        size = ttk.Frame(frame); size.pack(fill="x")

        ttk.Label(size, text="Filas:").pack(side="left")
        self.su_r = tk.Spinbox(size, from_=1,to=12,width=5); self.su_r.delete(0,"end"); self.su_r.insert(0,"2"); self.su_r.pack(side="left", padx=6)
        ttk.Label(size, text="Columnas:").pack(side="left")
        self.su_c = tk.Spinbox(size, from_=1,to=12,width=5); self.su_c.delete(0,"end"); self.su_c.insert(0,"2"); self.su_c.pack(side="left", padx=6)
        ttk.Button(size, text="Redimensionar",
                   command=lambda: [self.suma_A.set_size(int(self.su_r.get()), int(self.su_c.get())),
                                    self.suma_B.set_size(int(self.su_r.get()), int(self.su_c.get()))]
                   ).pack(side="left", padx=10)

        ttk.Label(frame, text="Matriz A", style="Title.TLabel").pack(anchor="w", pady=(8,2))
        self.suma_A = MatrixInput(frame, rows=2, cols=2, allow_b=False); self.suma_A.pack(fill="x")
        ttk.Label(frame, text="Matriz B", style="Title.TLabel").pack(anchor="w", pady=(8,2))
        self.suma_B = MatrixInput(frame, rows=2, cols=2, allow_b=False); self.suma_B.pack(fill="x")

        out = ttk.Panedwindow(frame, orient="horizontal"); out.pack(fill="both", expand=True, pady=8)
        res_box = ttk.Labelframe(out, text="Resultado", style="Card.TLabelframe", padding=6)
        self.suma_out = make_text(res_box, height=12, wrap="word"); self.suma_out.pack(fill="both", expand=True)
        out.add(res_box, weight=1)

        log_box = ttk.Labelframe(out, text="Pasos", style="Card.TLabelframe", padding=6)
        self.suma_log = make_text(log_box, height=12, wrap="word");self.suma_log.pack(fill="both", expand=True)
        out.add(log_box, weight=1)

        btns = ttk.Frame(frame); btns.pack(fill="x")
        ttk.Button(btns, text="Calcular", style="Accent.TButton", command=self._calc_suma).pack(side="left")
        ttk.Button(btns, text="Limpiar",
                   command=lambda: [self.suma_A.clear(), self.suma_B.clear(),
                                    self.suma_out.delete(1.0, tk.END), self.suma_log.delete(1.0, tk.END)]
                   ).pack(side="left", padx=6)

    def _calc_suma(self):
        try:
            A = self.suma_A.get_matrix(); B = self.suma_B.get_matrix()
            R, pasos = sumar_matrices(A, B)
        except Exception as e:
            messagebox.showerror("Error", str(e)); return
        self.suma_out.delete(1.0, tk.END); self.suma_out.insert(tk.END, formatear_matriz(R))
        self.suma_log.delete(1.0, tk.END)
        for p in pasos: self.suma_log.insert(tk.END, p + "\n")

    # -------- Tab 3: Multiplicación --------
    def _tab_mult(self):
        tab = ttk.Frame(self.nb)
        self.nb.add(tab, text="Multiplicación")

        frame = ttk.Frame(tab, padding=10); frame.pack(fill="both", expand=True)
        size = ttk.Frame(frame); size.pack(fill="x")

        ttk.Label(size, text="Filas A:").pack(side="left")
        self.mu_ar = tk.Spinbox(size, from_=1,to=12,width=5); self.mu_ar.delete(0,"end"); self.mu_ar.insert(0,"2"); self.mu_ar.pack(side="left", padx=6)
        ttk.Label(size, text="Columnas A:").pack(side="left")
        self.mu_ac = tk.Spinbox(size, from_=1,to=12,width=5); self.mu_ac.delete(0,"end"); self.mu_ac.insert(0,"2"); self.mu_ac.pack(side="left", padx=6)
        ttk.Label(size, text="Filas B:").pack(side="left", padx=(10,0))
        self.mu_br = tk.Spinbox(size, from_=1,to=12,width=5); self.mu_br.delete(0,"end"); self.mu_br.insert(0,"2"); self.mu_br.pack(side="left", padx=6)
        ttk.Label(size, text="Columnas B:").pack(side="left")
        self.mu_bc = tk.Spinbox(size, from_=1,to=12,width=5); self.mu_bc.delete(0,"end"); self.mu_bc.insert(0,"2"); self.mu_bc.pack(side="left", padx=6)

        def resize_inputs():
            try:
                ra, ca = int(self.mu_ar.get()), int(self.mu_ac.get())
                rb, cb = int(self.mu_br.get()), int(self.mu_bc.get())
            except:
                messagebox.showerror("Error","Dimensiones inválidas"); return
            if ca != rb:
                messagebox.showerror("Error","Para multiplicar: columnas de A deben coincidir con filas de B."); return
            self.mult_A.set_size(ra, ca); self.mult_B.set_size(rb, cb)

        ttk.Button(size, text="Redimensionar", command=resize_inputs).pack(side="left", padx=10)

        ttk.Label(frame, text="Matriz A", style="Title.TLabel").pack(anchor="w", pady=(8,2))
        self.mult_A = MatrixInput(frame, rows=2, cols=2, allow_b=False); self.mult_A.pack(fill="x")
        ttk.Label(frame, text="Matriz B", style="Title.TLabel").pack(anchor="w", pady=(8,2))
        self.mult_B = MatrixInput(frame, rows=2, cols=2, allow_b=False); self.mult_B.pack(fill="x")

        out = ttk.Panedwindow(frame, orient="horizontal"); out.pack(fill="both", expand=True, pady=8)
        res_box = ttk.Labelframe(out, text="Resultado", style="Card.TLabelframe", padding=6)
        self.mult_out = make_text(res_box, height=12, wrap="word"); self.mult_out.pack(fill="both", expand=True)
        out.add(res_box, weight=1)

        log_box = ttk.Labelframe(out, text="Pasos", style="Card.TLabelframe", padding=6)
        self.mult_log = make_text(log_box, height=12, wrap="word"); self.mult_log.pack(fill="both", expand=True)
        out.add(log_box, weight=1)

        btns = ttk.Frame(frame); btns.pack(fill="x")
        ttk.Button(btns, text="Calcular", style="Accent.TButton", command=self._calc_mult).pack(side="left")
        ttk.Button(btns, text="Limpiar",
                   command=lambda: [self.mult_A.clear(), self.mult_B.clear(),
                                    self.mult_out.delete(1.0, tk.END), self.mult_log.delete(1.0, tk.END)]
                   ).pack(side="left", padx=6)

    def _calc_mult(self):
        try:
            A = self.mult_A.get_matrix(); B = self.mult_B.get_matrix()
            R, pasos = multiplicar_matrices(A, B)
        except Exception as e:
            messagebox.showerror("Error", str(e)); return
        self.mult_out.delete(1.0, tk.END); self.mult_out.insert(tk.END, formatear_matriz(R))
        self.mult_log.delete(1.0, tk.END)
        for p in pasos: self.mult_log.insert(tk.END, p + "\n")

    # -------- Tab 4: Escalar × Matriz (paso a paso) --------
    def _tab_escalar(self):
        tab = ttk.Frame(self.nb)
        self.nb.add(tab, text="Escalar × Matriz")

        frame = ttk.Frame(tab, padding=10); frame.pack(fill="both", expand=True)

        top = ttk.Frame(frame); top.pack(fill="x")
        ttk.Label(top, text="Filas:").pack(side="left")
        self.es_r = tk.Spinbox(top, from_=1,to=12,width=5); self.es_r.pack(side="left", padx=4)
        ttk.Label(top, text="Columnas:").pack(side="left")
        self.es_c = tk.Spinbox(top, from_=1,to=12,width=5); self.es_c.pack(side="left", padx=4)
        ttk.Label(top, text="Escalar:").pack(side="left")
        self.es_val = ttk.Entry(top, width=10); self.es_val.pack(side="left", padx=6)
        ttk.Button(top, text="Redimensionar",
                   command=lambda: self.es_A.set_size(int(self.es_r.get()), int(self.es_c.get()))
                   ).pack(side="left", padx=10)

        self.es_A = MatrixInput(frame, rows=2, cols=2, allow_b=False); self.es_A.pack(fill="x", pady=8)

        btns = ttk.Frame(frame); btns.pack(fill="x")
        ttk.Button(btns, text="Calcular", style="Accent.TButton", command=self._escalar_calc).pack(side="left")
        self.es_next = ttk.Button(btns, text="Siguiente paso", command=self._escalar_next, state="disabled")
        self.es_next.pack(side="left", padx=6)
        ttk.Button(btns, text="Exportar log", command=self._escalar_export).pack(side="left")

        self.es_txt = make_text(frame, height=18, wrap="word"); self.es_txt.pack(fill="both", expand=True)
        self._es_pasos = []; self._es_idx = 0; self._es_resultado = []

    def _escalar_calc(self):
        try:
            A = self.es_A.get_matrix()
            esc = self.es_val.get()
            R, pasos = multiplicar_escalar_matriz(esc, A)  # misma lógica
        except Exception as e:
            messagebox.showerror("Error", str(e)); return
        self._es_resultado = R
        self._es_pasos = pasos
        self._es_idx = 0
        self.es_txt.delete(1.0, tk.END)
        self.es_txt.insert(tk.END, "Cálculo inicializado. Presione 'Siguiente paso'.\n")
        self.es_next.config(state="normal")

    def _escalar_next(self):
        if self._es_idx < len(self._es_pasos):
            self.es_txt.insert(tk.END, self._es_pasos[self._es_idx] + "\n\n")
            self._es_idx += 1
            if self._es_idx == len(self._es_pasos):
                self.es_next.config(state="disabled")
        else:
            self.es_next.config(state="disabled")

    def _escalar_export(self):
        if not self._es_pasos:
            messagebox.showinfo("Info", "No hay pasos para exportar."); return
        fp = filedialog.asksaveasfilename(defaultextension=".txt",
                                          filetypes=[("Texto","*.txt")],
                                          title="Guardar registro de pasos")
        if not fp: return
        with open(fp, "w", encoding="utf-8") as f:
            for i, p in enumerate(self._es_pasos, start=1):
                f.write(f"Paso {i}: {p}\n\n")
            f.write("Resultado final:\n")
            for fila in self._es_resultado:
                f.write(" ".join(str(x) for x in fila) + "\n")
        messagebox.showinfo("Listo", f"Registro exportado a: {fp}")

    # -------- Tab 5: Transpuesta --------
    def _tab_transpuesta(self):
        tab = ttk.Frame(self.nb)
        self.nb.add(tab, text="Transpuesta")

        frame = ttk.Frame(tab, padding=10); frame.pack(fill="both", expand=True)
        size = ttk.Frame(frame); size.pack(fill="x")
        ttk.Label(size, text="Filas:").pack(side="left")
        self.tr_r = tk.Spinbox(size, from_=1,to=12,width=5); self.tr_r.delete(0,"end"); self.tr_r.insert(0,"2"); self.tr_r.pack(side="left", padx=6)
        ttk.Label(size, text="Columnas:").pack(side="left")
        self.tr_c = tk.Spinbox(size, from_=1,to=12,width=5); self.tr_c.delete(0,"end"); self.tr_c.insert(0,"2"); self.tr_c.pack(side="left", padx=6)
        ttk.Button(size, text="Redimensionar",
                   command=lambda: self.tr_A.set_size(int(self.tr_r.get()), int(self.tr_c.get()))
                   ).pack(side="left", padx=10)

        ttk.Label(frame, text="Matriz", style="Title.TLabel").pack(anchor="w", pady=(8,2))
        self.tr_A = MatrixInput(frame, rows=2, cols=2, allow_b=False); self.tr_A.pack(fill="x")

        ttk.Label(frame, text="Resultado", style="Title.TLabel").pack(anchor="w", pady=(8,2))
        self.tr_out = make_text(frame, height=14, wrap="word"); self.tr_out.pack(fill="both", expand=True)

        btns = ttk.Frame(frame); btns.pack(fill="x", pady=8)
        ttk.Button(btns, text="Calcular", style="Accent.TButton", command=self._calc_transpuesta).pack(side="left")
        ttk.Button(btns, text="Limpiar",
                   command=lambda: [self.tr_A.clear(), self.tr_out.delete(1.0, tk.END)]
                   ).pack(side="left", padx=6)

    def _calc_transpuesta(self):
        try:
            A = self.tr_A.get_matrix()
            R = Transpuesta(A)
        except Exception as e:
            messagebox.showerror("Error", str(e)); return
        self.tr_out.delete(1.0, tk.END); self.tr_out.insert(tk.END, formatear_matriz(R))

    # -------- Tab 6: Inversa de una matriz --------
    def _tab_inversa(self):
        tab = ttk.Frame(self.nb)
        self.nb.add(tab, text="Inversa")

        frame = ttk.Frame(tab, padding=10)
        frame.pack(fill="both", expand=True)

        # ---- Barra superior: tamaño + redimensionar + modo Gauss (solo comprobar) ----
        size = ttk.Frame(frame)
        size.pack(fill="x")
        ttk.Label(size, text="Tamaño (n×n):").pack(side="left")
        self.inv_n = tk.Spinbox(size, from_=1, to=12, width=5)
        self.inv_n.delete(0, "end"); self.inv_n.insert(0, "3")
        self.inv_n.pack(side="left", padx=6)

        ttk.Button(
            size, text="Redimensionar",
            command=lambda: self.inv_A.set_size(int(self.inv_n.get()), int(self.inv_n.get()))
        ).pack(side="left", padx=10)

        # Toggle: solo comprobar invertibilidad (Gauss)
        self.inv_only_check = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            size,
            text="Solo comprobar invertibilidad (Gauss)",
            variable=self.inv_only_check
        ).pack(side="left", padx=10)

        # ---- Entrada de matriz ----
        ttk.Label(frame, text="Matriz A", style="Title.TLabel").pack(anchor="w", pady=(8, 2))
        self.inv_A = MatrixInput(frame, rows=3, cols=3, allow_b=False)
        self.inv_A.pack(fill="x")

        # ---- Paneles de salida: Resultado + Pasos ----
        out = ttk.Panedwindow(frame, orient="horizontal")
        out.pack(fill="both", expand=True, pady=8)

        # Guardamos la labelframe en self.inv_res_box por si luego queremos cambiarle el título
        self.inv_res_box = ttk.Labelframe(out, text="Matriz inversa", style="Card.TLabelframe", padding=6)
        self.inv_out = make_text(self.inv_res_box, height=14, wrap="word"); self.inv_out.pack(fill="both", expand=True)
        out.add(self.inv_res_box, weight=1)

        log_box = ttk.Labelframe(out, text="Pasos", style="Card.TLabelframe", padding=6)
        self.inv_log = make_text(log_box, height=14, wrap="word"); self.inv_log.pack(fill="both", expand=True)
        out.add(log_box, weight=1)

        # ---- Botones ----
        btns = ttk.Frame(frame)
        btns.pack(fill="x", pady=8)
        ttk.Button(btns, text="Calcular", style="Accent.TButton", command=self._calc_inversa).pack(side="left")
        ttk.Button(
            btns, text="Limpiar",
            command=lambda: [self.inv_A.clear(), self.inv_out.delete(1.0, tk.END), self.inv_log.delete(1.0, tk.END)]
        ).pack(side="left", padx=6)
        ttk.Button(btns, text="Exportar pasos", command=self._inv_export).pack(side="left", padx=6)

        # Buffer para exportar pasos
        self.inv_last_steps = []

    def _calc_inversa(self):
        from matrices import formatear_matriz
        # Validación previa
        try:
            A = self.inv_A.get_matrix()
            if not A or not A[0]:
                messagebox.showinfo("Información", "Ingresa valores en la matriz antes de calcular.")
                return
            n = len(A); m = len(A[0])
            if n != m:
                messagebox.showwarning("Matriz no cuadrada", f"La matriz debe ser cuadrada (n×n). Recibido {n}×{m}.")
                return
        except Exception as e:
            messagebox.showerror("Entrada inválida", str(e))
            return

        # Limpiar salidas
        self.inv_out.delete(1.0, tk.END)
        self.inv_log.delete(1.0, tk.END)
        self.inv_last_steps = []

        if self.inv_only_check.get():
            # --- SOLO COMPROBAR INVERTIBILIDAD (Gauss) ---
            try:
                from matrices import comprobar_invertibilidad
                es_inv, U, pasos, pivs, det = comprobar_invertibilidad(A)
            except Exception as e:
                messagebox.showerror("Error", str(e)); return

            self.inv_last_steps = pasos
            for p in pasos:
                self.inv_log.insert(tk.END, p + "\n\n")

            if es_inv:
                self.inv_out.insert(tk.END, "Matriz escalonada (triangular superior):\n")
                self.inv_out.insert(tk.END, formatear_matriz(U) + "\n")
                self.inv_out.insert(tk.END, f"Pivotes: {pivs}  |  Determinante: {det}\n")
                self.inv_out.insert(tk.END, "Conclusión: A es invertible (rank = n).")
            else:
                self.inv_out.insert(tk.END, "Conclusión: A NO es invertible (det = 0, rank < n).")

            self._update_status("Comprobación de invertibilidad finalizada.")
            return

        # --- CALCULAR INVERSA COMPLETA (Gauss-Jordan con paso a paso) ---
        from matrices import inversa_matriz
        try:
            R, pasos = inversa_matriz(A)
        except ValueError as ve:
            msg = str(ve)
            if "no es invertible" in msg or "determinante = 0" in msg:
                messagebox.showwarning(
                    "Sin inversa",
                    "La matriz no es invertible (determinante = 0).\n"
                    "• Verifica filas/columnas proporcionales o repetidas.\n"
                    "• Evita filas en ceros."
                )
            elif "cuadrada" in msg:
                messagebox.showwarning("Matriz no cuadrada", msg)
            else:
                messagebox.showerror("Error", msg)
            self._update_status("No se pudo calcular la inversa.")
            return
        except Exception as e:
            messagebox.showerror("Error inesperado", str(e))
            self._update_status("Error inesperado al calcular la inversa.")
            return

        self.inv_last_steps = pasos
        self.inv_out.insert(tk.END, formatear_matriz(R))
        for p in pasos:
            self.inv_log.insert(tk.END, p + "\n\n")
        self._update_status("Inversa calculada correctamente.")

    def _inv_export(self):
        if not self.inv_last_steps:
            messagebox.showinfo("Información", "No hay pasos para exportar."); return
        fp = filedialog.asksaveasfilename(defaultextension=".txt",
                                        filetypes=[("Texto","*.txt")],
                                        title="Guardar registro de pasos")
        if not fp: return
        with open(fp, "w", encoding="utf-8") as f:
            for i, p in enumerate(self.inv_last_steps, start=1):
                f.write(f"Paso {i}:\n{p}\n\n")
        messagebox.showinfo("Listo", f"Registro exportado a: {fp}")

    # -------- Tab 7: Independencia Lineal (nueva) --------
    def _tab_independencia(self):
        tab = ttk.Frame(self.nb)
        self.nb.add(tab, text="Independencia Lineal")

        frame = ttk.Frame(tab, padding=10); frame.pack(fill="both", expand=True)

        # Configuración de tamaño: filas (dimensión de vectores), columnas (cantidad de vectores)
        size = ttk.Frame(frame); size.pack(fill="x")
        ttk.Label(size, text="Dimensión (filas):").pack(side="left")
        self.il_r = tk.Spinbox(size, from_=1,to=12,width=5); self.il_r.delete(0,"end"); self.il_r.insert(0,"3"); self.il_r.pack(side="left", padx=6)
        ttk.Label(size, text="Cantidad de vectores (columnas):").pack(side="left", padx=(10,0))
        self.il_c = tk.Spinbox(size, from_=1,to=12,width=5); self.il_c.delete(0,"end"); self.il_c.insert(0,"3"); self.il_c.pack(side="left", padx=6)
        ttk.Button(size, text="Redimensionar",
                   command=lambda: self.il_A.set_size(int(self.il_r.get()), int(self.il_c.get()))
                   ).pack(side="left", padx=10)

        # Entrada de matriz (cada columna es un vector)
        ttk.Label(frame, text="Matriz (cada columna = vector)", style="Title.TLabel").pack(anchor="w", pady=(8,2))
        # NOTA: allow_b=False porque trabajamos con sistema homogéneo A x = 0
        self.il_A = MatrixInput(frame, rows=3, cols=3, allow_b=False); self.il_A.pack(fill="x")

        out = ttk.Panedwindow(frame, orient="horizontal"); out.pack(fill="both", expand=True, pady=8)
        # Vista de matriz reducida
        res_box = ttk.Labelframe(out, text="Matriz reducida (Gauss-Jordan)", style="Card.TLabelframe", padding=6)
        self.il_view = MatrixView(res_box); self.il_view.pack(fill="both", expand=True)
        out.add(res_box, weight=1)

        # Pasos y conclusión
        log_box = ttk.Labelframe(out, text="Pasos y conclusión", style="Card.TLabelframe", padding=6)
        self.il_log = make_text(log_box, height=16, wrap="word"); self.il_log.pack(fill="both", expand=True)
        out.add(log_box, weight=1)

        btns = ttk.Frame(frame); btns.pack(fill="x")
        ttk.Button(btns, text="Analizar independencia", style="Accent.TButton", command=self._calc_independencia).pack(side="left")
        ttk.Button(btns, text="Limpiar",
                   command=lambda: [self.il_A.clear(), self.il_log.delete(1.0, tk.END), self.il_view.set_matrix([[Fraccion(0)]])]
                   ).pack(side="left", padx=6)
        ttk.Button(btns, text="Exportar pasos", command=self._il_export).pack(side="left", padx=6)

        # almacenamiento temporal
        self._il_engine = None

    def _calc_independencia(self):
        """
        Construye el sistema homogéneo A x = 0 (añadiendo columna 0),
        ejecuta Gauss-Jordan con el motor existente y muestra pasos + conclusión.
        """
        try:
            A = self.il_A.get_matrix()  # matriz (dim x p) — columnas son vectores
        except Exception as e:
            messagebox.showerror("Entrada inválida", str(e)); return

        # Si hay más vectores que dimensiones, por teorema es dependiente (p > n)
        n_rows = len(A)
        n_cols = len(A[0]) if A else 0
        if n_cols == 0:
            messagebox.showerror("Error", "La matriz no tiene columnas."); return

        # Construir matriz aumentada para sistema homogéneo A x = 0
        zero = Fraccion(0)
        aug = []
        for r in range(n_rows):
            row = [A[r][c] for c in range(n_cols)]
            row.append(zero)
            aug.append(row)

        # Inicializar motor Gauss-Jordan (reutilizando tu implementación)
        engine = GaussJordanEngine(aug)
        self._il_engine = engine

        # Ejecutar hasta finalizar (guardamos pasos)
        while not engine.terminado:
            engine.siguiente()

        # Mostrar pasos
        self.il_log.delete(1.0, tk.END)
        for i, s in enumerate(engine.log):
            self.il_log.insert(tk.END, f"Paso {i+1}: {s.descripcion}\n")
            self.il_log.insert(tk.END, formatear_matriz(s.matriz) + "\n\n")

        # Mostrar matriz reducida
        last = engine.log[-1] if engine.log else None
        if last:
            self.il_view.set_matrix(last.matriz)

        # Analizar resultados
        resultado = engine.analizar()
        tipo, data, clasificacion = resultado

        # Para sistema homogéneo A x = 0:
        # - si solución trivial única => columnas de A son linealmente independientes
        # - si solución no trivial (infinitas) => dependientes
        conclusion = ""
        if tipo == "única" and clasificacion.get("solucion", "") == "trivial":
            conclusion = f"Conclusión: Las columnas (vectores) son LINEALMENTE INDEPENDIENTES.\n\nClasificación: {clasificacion}"
        elif tipo == "inconsistente":
            # teóricamente no debería pasar en homogéneo, pero lo cubrimos
            conclusion = "Conclusión: El sistema es inconsistente (algo no esperado para homogéneo)."
        else:
            # infinitas -> dependencia
            # mostrar relación de dependencia usando la representación paramétrica
            sol_text = engine.conjunto_solucion(resultado)
            conclusion = "Conclusión: Las columnas (vectores) son LINEALMENTE DEPENDIENTES.\n\n"
            conclusion += "Relación(es) de dependencia (forma paramétrica / base del espacio nulo):\n"
            conclusion += sol_text

        self.il_log.insert(tk.END, "\n" + conclusion)
        self.lbl_result.config(text="Independencia analizada.")
        self._update_status("Análisis de independencia completado.")

    def _il_export(self):
        """Exporta los pasos generados en la pestaña de independencia."""
        engine = self._il_engine
        if not engine or not engine.log:
            messagebox.showinfo("Información", "No hay pasos para exportar"); return
        fp = filedialog.asksaveasfilename(defaultextension=".txt",
                                          filetypes=[("Texto","*.txt")],
                                          title="Guardar registro de pasos")
        if not fp: return
        with open(fp, "w", encoding="utf-8") as f:
            for i, s in enumerate(engine.log, start=1):
                f.write(f"Paso {i}: {s.descripcion}\n")
                for fila in s.matriz:
                    f.write(" [ " + " ".join(str(x) for x in fila[:-1]) + " | " + str(fila[-1]) + " ]\n")
                f.write("\n")
            # agregar conclusión final
            resultado = engine.analizar()
            f.write("\nConclusión final:\n")
            f.write(engine.conjunto_solucion(resultado) + "\n")
        messagebox.showinfo("Listo", f"Registro exportado a: {fp}")

    # -------- Tab 8: Determinante --------
    def _tab_determinante(self):
        tab = ttk.Frame(self.nb)
        self.nb.add(tab, text="Determinante")

        frame = ttk.Frame(tab, padding=10)
        frame.pack(fill="both", expand=True)

        # ---- Controles de tamaño ----
        size = ttk.Frame(frame); size.pack(fill="x")
        ttk.Label(size, text="Tamaño (n×n):").pack(side="left")
        self.det_n = tk.Spinbox(size, from_=1, to=12, width=5)
        self.det_n.delete(0, "end"); self.det_n.insert(0, "3")
        self.det_n.pack(side="left", padx=6)

        ttk.Button(
            size, text="Redimensionar",
            command=lambda: self.det_A.set_size(int(self.det_n.get()), int(self.det_n.get()))
        ).pack(side="left", padx=10)

        # ---- Selector de método ----
        method_row = ttk.Frame(frame); method_row.pack(fill="x", pady=(6, 2))
        self.det_use_laplace = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            method_row,
            text="Usar cofactores",
            variable=self.det_use_laplace
        ).pack(side="left")

        self.det_laplace_pref = tk.StringVar(value="auto")  # "auto", "col0", "fila0"
        ttk.Label(method_row, text="Preferencia:").pack(side="left", padx=(12, 4))
        ttk.Combobox(
            method_row, textvariable=self.det_laplace_pref, width=7,
            values=("auto", "col0", "fila0"), state="readonly"
        ).pack(side="left")

        # ---- Matriz A ----
        ttk.Label(frame, text="A:", style="Title.TLabel").pack(anchor="w", pady=(8, 2))
        self.det_A = MatrixInput(frame, rows=3, cols=3, allow_b=False)
        self.det_A.pack(fill="x")

        # ---- Salidas: Resultado y Pasos ----
        out = ttk.Panedwindow(frame, orient="horizontal"); out.pack(fill="both", expand=True, pady=8)

        res_box = ttk.Labelframe(out, text="Resultado", style="Card.TLabelframe", padding=6)
        self.det_out = make_text(res_box, height=10, wrap="word"); self.det_out.pack(fill="both", expand=True)
        out.add(res_box, weight=1)

        log_box = ttk.Labelframe(out, text="Pasos", style="Card.TLabelframe", padding=6)
        self.det_log = make_text(log_box, height=10, wrap="word"); self.det_log.pack(fill="both", expand=True)
        out.add(log_box, weight=1)

        # ---- Botones ----
        btns = ttk.Frame(frame); btns.pack(fill="x")
        ttk.Button(btns, text="Calcular", style="Accent.TButton",
                   command=self._calc_determinante).pack(side="left")
        ttk.Button(btns, text="Limpiar",
                   command=lambda: [self.det_A.clear(),
                                    self.det_out.delete(1.0, tk.END),
                                    self.det_log.delete(1.0, tk.END)]
                   ).pack(side="left", padx=6)

        # Estado interno
        self._det_pasos = []
        self._det_val = None

    def _calc_determinante(self):
        # Leer la matriz y validar que sea cuadrada
        try:
            A = self.det_A.get_matrix()
            n = len(A)
            m = len(A[0]) if A else 0
            if n == 0 or n != m:
                messagebox.showwarning(
                    "Matriz no cuadrada",
                    f"La matriz debe ser cuadrada (n×n). Recibido {n}×{m}."
                )
                return
        except Exception as e:
            messagebox.showerror("Entrada inválida", str(e))
            return

        # Limpiar salidas
        self.det_out.delete(1.0, tk.END)
        self.det_log.delete(1.0, tk.END)

        # Calcular con el método seleccionado
        try:
            if self.det_use_laplace.get():
                # Cofactores (Laplace): "auto", "col0", "fila0"
                pref = self.det_laplace_pref.get()
                det, pasos = determinante_cofactores(A, prefer=pref)
            else:
                # Eliminación Gaussiana (rápido)
                det, pasos = determinante_matriz(A)
        except Exception as e:
            messagebox.showerror("Error", str(e))
            return

        # Guardar y mostrar
        self._det_pasos = pasos
        self._det_val = det

        self.det_out.insert(tk.END, f"det(A) = {det}\n")
        for p in pasos:
            self.det_log.insert(tk.END, p + "\n\n")

        # Si tu app tiene un label de resultado general:
        if hasattr(self, "lbl_result"):
            self.lbl_result.config(text=f"det(A) = {det}")

        # Y si tienes barra de estado:
        if hasattr(self, "_update_status"):
            self._update_status("Determinante calculado.")

    def _det_export(self):
        if not self._det_pasos:
            messagebox.showinfo("Información", "No hay pasos para exportar."); return
        fp = filedialog.asksaveasfilename(defaultextension=".txt",
                                          filetypes=[("Texto","*.txt")],
                                          title="Guardar registro de pasos")
        if not fp: return
        with open(fp, "w", encoding="utf-8") as f:
            for i, p in enumerate(self._det_pasos, start=1):
                f.write(f"Paso {i}:\n{p}\n\n")
            if self._det_val is not None:
                f.write(f"Resultado final: det(A) = {self._det_val}\n")
        messagebox.showinfo("Listo", f"Registro exportado a: {fp}")

#----- Cramer -----
    def _tab_cramer(self):
        tab = ttk.Frame(self.nb)
        self.nb.add(tab, text="Regla de Cramer")

        frame = ttk.Frame(tab, padding=10)
        frame.pack(fill="both", expand=True)

        # ---- Controles de tamaño ----
        size = ttk.Frame(frame)
        size.pack(fill="x")
        ttk.Label(size, text="Tamaño (n×n):").pack(side="left")
        self.cr_n = tk.Spinbox(size, from_=1, to=6, width=5)
        self.cr_n.delete(0, "end")
        self.cr_n.insert(0, "3")
        self.cr_n.pack(side="left", padx=6)

        ttk.Button(
            size, text="Redimensionar",
            command=self._cramer_resize
        ).pack(side="left", padx=10)

        # ---- Matriz A y vector b ----
        matrix_frame = ttk.Frame(frame)
        matrix_frame.pack(fill="x", pady=8)

        # Matriz A
        left_mat = ttk.Frame(matrix_frame)
        left_mat.pack(side="left", fill="x", expand=True)
        ttk.Label(left_mat, text="A (coeficientes)", style="Title.TLabel").pack(anchor="w")
        self.cramer_A = MatrixInput(left_mat, rows=3, cols=3, allow_b=False)
        self.cramer_A.pack(fill="x")

        # Vector b
        right_vec = ttk.Frame(matrix_frame)
        right_vec.pack(side="right", fill="x", padx=(20, 0))
        ttk.Label(right_vec, text="Vector b (términos independientes)", style="Title.TLabel").pack(anchor="w")
        self.cramer_b = MatrixInput(right_vec, rows=3, cols=1, allow_b=False)
        self.cramer_b.pack(fill="x")

        # ---- Salidas: Resultado y Pasos ----
        out = ttk.Panedwindow(frame, orient="horizontal")
        out.pack(fill="both", expand=True, pady=8)

        res_box = ttk.Labelframe(out, text="Solución", style="Card.TLabelframe", padding=6)
        self.cramer_out = make_text(res_box, height=12, wrap="word"); self.cramer_out.pack(fill="both", expand=True)
        out.add(res_box, weight=1)

        log_box = ttk.Labelframe(out, text="Pasos de Cramer", style="Card.TLabelframe", padding=6)
        self.cramer_log = make_text(log_box, height=12, wrap="word"); self.cramer_log.pack(fill="both", expand=True)
        out.add(log_box, weight=1)

        # ---- Botones ----
        btns = ttk.Frame(frame)
        btns.pack(fill="x", pady=8)
        ttk.Button(btns, text="Resolver por Cramer", style="Accent.TButton",
                   command=self._calc_cramer).pack(side="left")
        ttk.Button(btns, text="Limpiar",
                   command=lambda: [self.cramer_A.clear(), self.cramer_b.clear(),
                                    self.cramer_out.delete(1.0, tk.END),
                                    self.cramer_log.delete(1.0, tk.END)]
                   ).pack(side="left", padx=6)
        ttk.Button(btns, text="Ejemplo 2×2",
                   command=self._cramer_example_2x2).pack(side="left", padx=6)
        ttk.Button(btns, text="Ejemplo 3×3",
                   command=self._cramer_example_3x3).pack(side="left", padx=6)

    def _cramer_resize(self):
        try:
            n = int(self.cr_n.get())
            if n <= 0:
                raise ValueError
        except:
            messagebox.showerror("Error", "Tamaño inválido")
            return
        self.cramer_A.set_size(n, n)
        self.cramer_b.set_size(n, 1)

    def _cramer_example_2x2(self):
        """Ejemplo del PDF: 3x₁ - 2x₂ = 6, -5x₁ + 4x₂ = 8"""
        self.cr_n.delete(0, tk.END)
        self.cr_n.insert(0, "2")
        self._cramer_resize()

        # Matriz A
        A_entries = [["3", "-2"], ["-5", "4"]]
        for i in range(2):
            for j in range(2):
                self.cramer_A.entries[i][j].delete(0, tk.END)
                self.cramer_A.entries[i][j].insert(0, A_entries[i][j])

        # Vector b
        b_entries = [["6"], ["8"]]
        for i in range(2):
            self.cramer_b.entries[i][0].delete(0, tk.END)
            self.cramer_b.entries[i][0].insert(0, b_entries[i][0])

    def _cramer_example_3x3(self):
        """Ejemplo genérico 3×3"""
        self.cr_n.delete(0, tk.END)
        self.cr_n.insert(0, "3")
        self._cramer_resize()

        # Matriz A
        A_entries = [["2", "1", "-1"], ["-3", "-1", "2"], ["-2", "1", "2"]]
        for i in range(3):
            for j in range(3):
                self.cramer_A.entries[i][j].delete(0, tk.END)
                self.cramer_A.entries[i][j].insert(0, A_entries[i][j])

        # Vector b
        b_entries = [["8"], ["-11"], ["-3"]]
        for i in range(3):
            self.cramer_b.entries[i][0].delete(0, tk.END)
            self.cramer_b.entries[i][0].insert(0, b_entries[i][0])

    def _calc_cramer(self):
        try:
            A = self.cramer_A.get_matrix()
            b_vec = self.cramer_b.get_matrix()
            b = [row[0] for row in b_vec]  # Convertir a vector simple
        except Exception as e:
            messagebox.showerror("Entrada inválida", str(e))
            return

        n = len(A)
        if n == 0 or len(A[0]) != n:
            messagebox.showerror("Error", "La matriz A debe ser cuadrada")
            return

        if len(b) != n:
            messagebox.showerror("Error", "El vector b debe tener la misma dimensión que A")
            return

        # Limpiar salidas
        self.cramer_out.delete(1.0, tk.END)
        self.cramer_log.delete(1.0, tk.END)

        try:
            solucion, pasos = self._regla_cramer(A, b)
        except Exception as e:
            messagebox.showerror("Error", str(e))
            return

        # Mostrar solución
        self.cramer_out.insert(tk.END, "Solución del sistema:\n\n")
        for i, x in enumerate(solucion, 1):
            self.cramer_out.insert(tk.END, f"x{i} = {x}\n")

        # Mostrar pasos
        for paso in pasos:
            self.cramer_log.insert(tk.END, paso + "\n\n")

        self._update_status("Regla de Cramer aplicada correctamente.")

    def _regla_cramer(self, A, b):
        """
        Resuelve un sistema lineal Ax = b usando la Regla de Cramer.
        Devuelve (solución: list[Fraccion], pasos: list[str])
        """
        from matrices import determinante_matriz, formatear_matriz
        n = len(A)
        pasos = []

        # Paso 1: Calcular determinante de A
        pasos.append("Paso 1: Calcular determinante de A")
        det_A, pasos_det = determinante_matriz(A)
        pasos.append(f"  A:")
        pasos.append(formatear_matriz(A))
        for p in pasos_det:
            pasos.append(p)
        pasos.append(f"det(A) = {det_A}")

        if det_A.es_cero():
            raise ValueError("A no es invertible (det(A) = 0). No se puede aplicar la Regla de Cramer.")

        pasos.append("")  # Línea en blanco

        # Paso 2: Para cada variable, calcular determinante de A_i(b)
        solucion = []
        for i in range(n):
            pasos.append(f"Paso {i + 2}: Calcular x{i + 1}")

            # Crear A_i(b): reemplazar columna i por b
            A_i = [fila[:] for fila in A]  # Copia profunda
            for j in range(n):
                A_i[j][i] = b[j]

            pasos.append(f"Matriz A_{i + 1}(b) (columna {i + 1} reemplazada por b):")
            pasos.append(formatear_matriz(A_i))

            # Calcular determinante de A_i(b)
            det_A_i, pasos_det_i = determinante_matriz(A_i)
            for p in pasos_det_i:
                pasos.append(p)
            pasos.append(f"det(A_{i + 1}(b)) = {det_A_i}")

            # Calcular x_i = det(A_i(b)) / det(A)
            x_i = det_A_i / det_A
            pasos.append(f"x_{i + 1} = det(A_{i + 1}(b)) / det(A) = {det_A_i} / {det_A} = {x_i}")
            solucion.append(x_i)
            pasos.append("")  # Línea en blanco

        return solucion, pasos

    def _tab_sarrus(self):
        tab = ttk.Frame(self.nb)
        self.nb.add(tab, text="Determinante (Sarrus)")

        frame = ttk.Frame(tab, padding=10)
        frame.pack(fill="both", expand=True)

        # ---- Controles de tamaño ----
        size = ttk.Frame(frame)
        size.pack(fill="x")
        ttk.Label(size, text="Tamaño (n×n):").pack(side="left")
        self.sarrus_n = tk.Spinbox(size, from_=2, to=6, width=5)
        self.sarrus_n.delete(0, "end")
        self.sarrus_n.insert(0, "3")
        self.sarrus_n.pack(side="left", padx=6)

        ttk.Button(size, text="Redimensionar", command=self._sarrus_resize).pack(side="left", padx=10)

        # ---- Matriz ----
        matrix_frame = ttk.Frame(frame)
        matrix_frame.pack(fill="x", pady=8)

        left_mat = ttk.Frame(matrix_frame)
        left_mat.pack(fill="x", expand=True)
        ttk.Label(left_mat, text="A (debe ser cuadrada)", style="Title.TLabel").pack(anchor="w")
        self.sarrus_A = MatrixInput(left_mat, rows=3, cols=3, allow_b=False)
        self.sarrus_A.pack(fill="x")

        # ---- Salidas: Resultado y Pasos ----
        out = ttk.Panedwindow(frame, orient="horizontal")
        out.pack(fill="both", expand=True, pady=8)

        res_box = ttk.Labelframe(out, text="Determinante", style="Card.TLabelframe", padding=6)
        self.sarrus_out = make_text(res_box, height=12, wrap="word"); self.sarrus_out.pack(fill="both", expand=True)
        out.add(res_box, weight=1)

        log_box = ttk.Labelframe(out, text="Pasos del método de Sarrus", style="Card.TLabelframe", padding=6)
        self.sarrus_log = make_text(log_box, height=12, wrap="word"); self.sarrus_log.pack(fill="both", expand=True)
        out.add(log_box, weight=1)

        # ---- Botones ----
        btns = ttk.Frame(frame)
        btns.pack(fill="x", pady=8)
        ttk.Button(btns, text="Calcular determinante", style="Accent.TButton",
                   command=self._calc_sarrus).pack(side="left")
        ttk.Button(btns, text="Limpiar",
                   command=lambda: [self.sarrus_A.clear(),
                                    self.sarrus_out.delete(1.0, tk.END),
                                    self.sarrus_log.delete(1.0, tk.END)]
                   ).pack(side="left", padx=6)
        ttk.Button(btns, text="Ejemplo 3×3",
                   command=self._sarrus_example_3x3).pack(side="left", padx=6)
        ttk.Button(btns, text="Ejemplo 4×4",
                   command=self._sarrus_example_4x4).pack(side="left", padx=6)

    def _sarrus_resize(self):
        try:
            n = int(self.sarrus_n.get())
            if n <= 1:
                raise ValueError
        except:
            messagebox.showerror("Error", "Tamaño inválido, debe ser al menos 2x2")
            return
        self.sarrus_A.set_size(n, n)

    def _sarrus_example_3x3(self):
        self.sarrus_n.delete(0, tk.END)
        self.sarrus_n.insert(0, "3")
        self._sarrus_resize()
        A_entries = [["1", "2", "3"],
                     ["0", "4", "5"],
                     ["1", "0", "6"]]
        for i in range(3):
            for j in range(3):
                self.sarrus_A.entries[i][j].delete(0, tk.END)
                self.sarrus_A.entries[i][j].insert(0, A_entries[i][j])

    def _sarrus_example_4x4(self):
        self.sarrus_n.delete(0, tk.END)
        self.sarrus_n.insert(0, "4")
        self._sarrus_resize()
        A_entries = [["1", "2", "1", "3"],
                     ["0", "1", "4", "2"],
                     ["2", "0", "1", "5"],
                     ["1", "3", "2", "0"]]
        for i in range(4):
            for j in range(4):
                self.sarrus_A.entries[i][j].delete(0, tk.END)
                self.sarrus_A.entries[i][j].insert(0, A_entries[i][j])

    def _calc_sarrus(self):
        from matrices import determinante_sarrus, formatear_matriz
        try:
            A = self.sarrus_A.get_matrix()
        except Exception as e:
            messagebox.showerror("Entrada inválida", str(e))
            return

        n = len(A)
        if n == 0 or len(A[0]) != n:
            messagebox.showerror("Error", "La matriz debe ser cuadrada")
            return

        # Limpiar salidas
        self.sarrus_out.delete(1.0, tk.END)
        self.sarrus_log.delete(1.0, tk.END)

        try:
            resultado, pasos = determinante_sarrus(A)
        except Exception as e:
            messagebox.showerror("Error", str(e))
            return

        # Mostrar pasos
        self.sarrus_log.insert(tk.END, "Cálculo del determinante por Sarrus:\n\n")
        self.sarrus_log.insert(tk.END, formatear_matriz(A) + "\n\n")
        for p in pasos:
            self.sarrus_log.insert(tk.END, p + "\n\n")

        # Mostrar resultado
        self.sarrus_out.insert(tk.END, f"Determinante de A:\n\n{resultado}\n")
        self._update_status("Determinante calculado correctamente con Sarrus.")

    # -------- Métodos numéricos (raíces por Bisección) --------
    def _tab_metodos_numericos(self):
        tab = ttk.Frame(self.nb)
        self.nb.add(tab, text="Método de Bisección")

        frame = ttk.Frame(tab, padding=10)
        frame.pack(fill="both", expand=True)

        # ---- Título y descripción ----
        title_frame = ttk.Frame(frame)
        title_frame.pack(fill="x", pady=(0, 10))

        ttk.Label(title_frame, text="Método de Bisección",
                  font=("Times New Roman", 14, "bold")).pack(anchor="w")
        ttk.Label(title_frame, text="Encuentra raíces de f(x) = 0 en [a, b]",
                  font=("Times New Roman", 10)).pack(anchor="w")

        # ---- Entrada de función ----
        func_frame = ttk.LabelFrame(frame, text="Función f(x)", padding=8)
        func_frame.pack(fill="x", pady=5)

        func_input_frame = ttk.Frame(func_frame)
        func_input_frame.pack(fill="x")

        ttk.Label(func_input_frame, text="f(x) =", font=("Times New Roman", 11)).pack(side="left")

        self.fx_entry = tk.Entry(func_input_frame, width=35, font=("Consolas", 11))
        self.fx_entry.pack(side="left", padx=5, fill="x", expand=True)
        self.fx_entry.insert(0, "x**2 + 3*x - 5")
        self.fx_entry.bind('<KeyRelease>', self._on_function_change)

        # ---- Display visual de la función ----
        self.fx_display = ttk.Label(func_frame, text="f(x) = x² + 3x - 5",
                                    font=("Times New Roman", 12, "bold"),
                                    foreground="#4fc3f7")
        self.fx_display.pack(anchor="w", pady=(5, 0))

        # ---- Parámetros de bisección ----
        params_frame = ttk.LabelFrame(frame, text="Parámetros", padding=8)
        params_frame.pack(fill="x", pady=10)

        ttk.Label(params_frame, text="Intervalo [a, b]:", font=("Times New Roman", 10)).grid(row=0, column=0, padx=5,
                                                                                             sticky="w")

        ttk.Label(params_frame, text="a =").grid(row=0, column=1, padx=2)
        self.a_entry = tk.Entry(params_frame, width=12, font=("Consolas", 10))
        self.a_entry.grid(row=0, column=2, padx=5)
        self.a_entry.insert(0, "1.0")

        ttk.Label(params_frame, text="b =").grid(row=0, column=3, padx=2)
        self.b_entry = tk.Entry(params_frame, width=12, font=("Consolas", 10))
        self.b_entry.grid(row=0, column=4, padx=5)
        self.b_entry.insert(0, "2.0")

        ttk.Label(params_frame, text="Tolerancia:").grid(row=0, column=5, padx=(20, 2))
        self.tol_entry = tk.Entry(params_frame, width=12, font=("Consolas", 10))
        self.tol_entry.grid(row=0, column=6, padx=5)
        self.tol_entry.insert(0, "0.00001")

        # Fila para botón de intervalo automático y botones de acción
        auto_frame = ttk.Frame(params_frame)
        auto_frame.grid(row=1, column=0, columnspan=7, pady=(10, 0), sticky="w")

        ttk.Button(auto_frame, text="Buscar Intervalo Automático",
                   command=self._buscar_intervalo_valido).pack(side="left", padx=6)

        # NUEVO: Botón para graficar antes de calcular
        ttk.Button(auto_frame, text="Graficar función",
                   command=self._graficar_funcion).pack(side="left", padx=6)

        self.calc_btn = ttk.Button(auto_frame, text="Calcular Bisección",
                                   style="Accent.TButton",
                                   command=self._calc_biseccion)
        self.calc_btn.pack(side="left", padx=6)

        ttk.Button(auto_frame, text="Limpiar",
                   command=self._limpiar_biseccion).pack(side="left", padx=6)
        
        # ---- Tabla de iteraciones ----
        table_frame = ttk.LabelFrame(frame, text="Iteraciones del Método de Bisección", padding=8)
        table_frame.pack(fill="both", expand=True, pady=10)

        columns = ("k", "a", "b", "c", "f(a)", "f(b)", "f(c)", "error")
        self.bis_tree = ttk.Treeview(table_frame, columns=columns, show="headings", height=15)

        headings = ["k", "a", "b", "c", "f(a)", "f(b)", "f(c)", "Error"]
        widths = [50, 90, 90, 90, 100, 100, 100, 90]

        for col, heading, width in zip(columns, headings, widths):
            self.bis_tree.heading(col, text=heading)
            self.bis_tree.column(col, width=width, anchor="center")

        scrollbar_table = ttk.Scrollbar(table_frame, orient="vertical", command=self.bis_tree.yview)
        self.bis_tree.configure(yscrollcommand=scrollbar_table.set)
        self.bis_tree.pack(side="left", fill="both", expand=True)
        scrollbar_table.pack(side="right", fill="y")

        # ---- Información de estado ----
        status_frame = ttk.Frame(frame)
        status_frame.pack(fill="x", pady=5)

        self.biseccion_status = ttk.Label(status_frame, text="Listo para calcular", font=("Times New Roman", 9))
        self.biseccion_status.pack(anchor="w")

    def _on_function_change(self, event=None):
        """Actualiza el display visual cuando cambia la función"""
        try:
            raw_text = self.fx_entry.get()
            display_text = self._convert_to_display(raw_text)
            self.fx_display.config(text=f"f(x) = {display_text}")
        except:
            self.fx_display.config(text="f(x) = ?")

    def _parse_calculation(self, func_str):
        import re
        import numpy as np

        calc_str = (func_str or "").strip()
        if not calc_str:
            return lambda x: 0

        calc_str = calc_str.replace("^", "**")
        superscript_map = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻", "0123456789-")
        calc_str = calc_str.translate(superscript_map)


        calc_str = re.sub(r'(?<=\d)(?=[A-Za-z\(])', '*', calc_str)


        calc_str = re.sub(r'(?<=[A-Za-z])(?=\d)', '*', calc_str)


        calc_str = re.sub(r'\)\s*\(', ')*(', calc_str)


        calc_str = re.sub(r'(?<=\))(?=[A-Za-z0-9])', '*', calc_str)


        FUNC_NAMES = {
            "sin", "cos", "tan", "asin", "acos", "atan",
            "sinh", "cosh", "tanh", "exp", "sqrt", "abs",
            "ln", "log", "log10", "pow", "np"
        }

        def _mul_before_paren(m):
            token = m.group(1)

            if token in FUNC_NAMES:
                return token + "("

            return token + "*("

        calc_str = re.sub(r'([A-Za-z0-9_]+)\s*\(', _mul_before_paren, calc_str)


        try:
            code = compile(calc_str, "<userfunc>", "eval")
        except Exception as e:
            raise ValueError(f"Invalid function expression: {e}")


        def _log(x, base=None):
            if base is None:
                return np.log10(x)
            return np.log(x) / np.log(base)


        safe_globals = {
            "__builtins__": {"__import__": __import__},
            "sin": np.sin,
            "cos": np.cos,
            "tan": np.tan,
            "asin": np.arcsin,
            "acos": np.arccos,
            "atan": np.arctan,
            "sinh": np.sinh,
            "cosh": np.cosh,
            "tanh": np.tanh,
            "exp": np.exp,
            "sqrt": np.sqrt,
            "abs": np.abs,
            "pow": np.power,
            "ln": np.log,
            "log": _log,
            "log10": np.log10,
            "pi": np.pi,
            "e": np.e,
            "np": np,
        }

        def f(x):
            try:
                return eval(code, safe_globals, {"x": x})
            except Exception as e:
                raise ValueError(f"Error evaluating function at x={x}: {e}")

        return f

    def _convert_to_display(self, text):
        """Convierte a notación matemática visual"""
        if not text.strip():
            return ""

        display = text

        # Reemplazar operadores
        display = display.replace('**', '^').replace('*', '·')

        # Reemplazar funciones para visualización
        function_display = {
            'math.log': 'ln',
            'math.log10': 'log',
            'math.sin': 'sin',
            'math.cos': 'cos',
            'math.tan': 'tan',
            'math.acos': 'acos',
            'math.asin': 'asin',
            'math.atan': 'atan',
            'sqrt': '√',
            'exp': 'e',
            'pi': 'π'
        }

        for py_func, display_func in function_display.items():
            display = display.replace(py_func, display_func)

        # Eliminar math. restante
        display = re.sub(r'math\.', '', display)

        # Multiplicaciones implícitas
        display = re.sub(r'(\d)([a-zA-Z])', r'\1·\2', display)
        display = re.sub(r'([a-zA-Z])(\d)', r'\1·\2', display)

        # Superíndices
        superscript_map = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")
        display = re.sub(r'(\w)\^(\d+)',
                         lambda m: m.group(1) + m.group(2).translate(superscript_map),
                         display)

        return display

    def _buscar_intervalo_valido(self):
        """Busca automáticamente un intervalo donde la función cambie de signo"""
        try:
            func_str = self.fx_entry.get()
            f = self._parse_calculation(func_str)

            from numericos import encontrar_intervalo_automatico
            a, b, mensaje = encontrar_intervalo_automatico(f)

            # Actualizar los campos de intervalo
            self.a_entry.delete(0, tk.END)
            self.a_entry.insert(0, f"{a:.4f}")
            self.b_entry.delete(0, tk.END)
            self.b_entry.insert(0, f"{b:.4f}")

            self.biseccion_status.config(text="Intervalo encontrado automáticamente")

        except Exception as e:
            messagebox.showerror("Error", f"No se pudo encontrar intervalo automático: {str(e)}")

    def _graficar_funcion(self):
        """Muestra una vista previa de f(x) en el intervalo [a, b] sin ejecutar la bisección."""
        try:
            func_str = self.fx_entry.get().strip()
            if not func_str:
                raise ValueError("Ingresa una expresión para f(x).")

            f = self._parse_calculation(func_str)

            # Intentar leer a y b; si no son válidos, buscar intervalo automático
            a_txt = self.a_entry.get().strip()
            b_txt = self.b_entry.get().strip()

            try:
                a = float(a_txt)
                b = float(b_txt)
            except Exception:
                # Fallback: buscar intervalo automáticamente y actualizar entradas
                from numericos import encontrar_intervalo_automatico
                a, b, _ = encontrar_intervalo_automatico(f)
                self.a_entry.delete(0, tk.END); self.a_entry.insert(0, f"{a:.6f}")
                self.b_entry.delete(0, tk.END); self.b_entry.insert(0, f"{b:.6f}")

            # Asegurar orden y evitar intervalo degenerado
            if a > b:
                a, b = b, a
            if a == b:
                a -= 1.0
                b += 1.0

            # Crear ventana emergente
            popup = tk.Toplevel(self)
            popup.title("Vista previa de f(x)")
            popup.geometry("900x550")
            popup.minsize(700, 450)
            popup.transient(self)
            popup.grab_set()
            configurar_estilo_oscuro(popup)

            # Contenedor
            container = ttk.Frame(popup, padding=10)
            container.pack(fill="both", expand=True)

            # Figura de Matplotlib
            fig = plt.figure(figsize=(8, 4.8), dpi=100)
            ax = fig.add_subplot(111)

            # Preparar datos
            import numpy as np
            margen = 0.1 * (b - a if b != a else 1.0)
            xs = np.linspace(a - margen, b + margen, 600)
            ys = []
            for x in xs:
                try:
                    ys.append(f(x))
                except Exception:
                    ys.append(np.nan)

            # Graficar
            ax.plot(xs, ys, linewidth=2, label=f'f(x) = {self._convert_to_display(func_str)}')
            ax.axhline(0, color='k', linestyle='--', alpha=0.7)
            ax.axvline(a, color='r', linestyle='--', alpha=0.7, label=f'a = {a:.4f}')
            ax.axvline(b, color='g', linestyle='--', alpha=0.7, label=f'b = {b:.4f}')
            ax.set_xlabel('x')
            ax.set_ylabel('f(x)')
            ax.set_title('Vista previa de f(x) en el intervalo')
            ax.grid(True, alpha=0.3)
            ax.legend()

            canvas = FigureCanvasTkAgg(fig, master=container)
            canvas.draw()

            toolbar = NavigationToolbar2Tk(canvas, container)
            toolbar.update()
            toolbar.pack(side="bottom", fill="x")

            canvas.get_tk_widget().pack(fill="both", expand=True)

        except Exception as e:
            messagebox.showerror("No se pudo graficar", str(e))

    def _limpiar_biseccion(self):
        """Limpia todos los resultados de bisección"""
        for item in self.bis_tree.get_children():
            self.bis_tree.delete(item)
        self.biseccion_status.config(text="Resultados limpiados - Listo para calcular")

    def _calc_biseccion(self):
        """Calcula bisección y muestra resultados en ventana emergente"""
        # Limpiar tabla
        for item in self.bis_tree.get_children():
            self.bis_tree.delete(item)

        try:
            # USAR PARSER DE EJECUCIÓN
            func_str = self.fx_entry.get()
            f = self._parse_calculation(func_str)
            a = float(self.a_entry.get())
            b = float(self.b_entry.get())

            # Manejar tolerancia vacía (valor por defecto 0)
            tol_str = self.tol_entry.get().strip()
            if tol_str == "":
                tol = 0.00001  # Valor por defecto
                self.tol_entry.delete(0, tk.END)
                self.tol_entry.insert(0, "0.00001")
            else:
                tol = float(tol_str)

            # Validaciones
            if a >= b:
                raise ValueError("Debe cumplirse: a < b")

            # Calcular bisección
            from numericos import biseccion
            raiz, pasos, motivo = biseccion(f, a, b, tol=tol, max_iter=100, usar_error="absoluto")

            # Mostrar en tabla
            self._mostrar_resultados_biseccion(pasos, raiz, motivo)

            # Mostrar ventana emergente con resultados
            self._mostrar_resultados_popup(func_str, f, a, b, raiz, pasos, motivo)

            # Actualizar estado
            self.biseccion_status.config(text=f"Bisección completada - {len(pasos)} iteraciones")
            if hasattr(self, "lbl_result"):
                self.lbl_result.config(text=f"Raíz ≈ {raiz:.8f} ({motivo})")

        except Exception as e:
            messagebox.showerror("Error en bisección", str(e))

    def _mostrar_resultados_popup(self, func_str, f, a, b, raiz, pasos, motivo):
        """Muestra ventana emergente con gráfica y resultados"""
        # Crear ventana emergente
        popup = tk.Toplevel(self)
        popup.title("Resultados - Método de Bisección")
        popup.geometry("1020x600")
        popup.minsize(800, 600)
        popup.transient(self)
        popup.grab_set()

        # Configurar estilo oscuro
        configurar_estilo_oscuro(popup)

        # Frame principal
        main_frame = ttk.Frame(popup, padding=10)
        main_frame.pack(fill="both", expand=True)

        # Título
        title_label = ttk.Label(main_frame, text="Resultados del Método de Bisección",
                                font=("Times New Roman", 14, "bold"))
        title_label.pack(pady=(0, 10))

        # Frame para gráfica y resultados
        content_frame = ttk.PanedWindow(main_frame, orient="horizontal")
        content_frame.pack(fill="both", expand=True, pady=10)

        # Frame para gráfica
        graph_frame = ttk.Labelframe(content_frame, text="Gráfica de la Función",
                                    style="Card.TLabelframe", padding=8)

        # Crear figura de matplotlib
        fig = plt.figure(figsize=(8, 5), dpi=100)
        ax = fig.add_subplot(111)

        # Graficar función
        x_plot = np.linspace(a - (b - a) * 0.1, b + (b - a) * 0.1, 400)
        y_plot = [f(xi) for xi in x_plot]

        ax.plot(x_plot, y_plot, 'b-', linewidth=2, label=f'f(x) = {self._convert_to_display(func_str)}')
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.7)
        ax.axvline(x=a, color='r', linestyle='--', alpha=0.7, label=f'a = {a:.4f}')
        ax.axvline(x=b, color='g', linestyle='--', alpha=0.7, label=f'b = {b:.4f}')
        ax.axvline(x=raiz, color='orange', linestyle='-', alpha=0.8, label=f'Raíz ≈ {raiz:.6f}')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('x')
        ax.set_ylabel('f(x)')
        ax.set_title('Gráfica de la función y solución encontrada')
        ax.legend()

        # Canvas y toolbar interactiva
        canvas = FigureCanvasTkAgg(fig, master=graph_frame)
        canvas.draw()

        toolbar = NavigationToolbar2Tk(canvas, graph_frame)
        toolbar.update()
        toolbar.pack(side="bottom", fill="x")

        canvas.get_tk_widget().pack(fill="both", expand=True)

        content_frame.add(graph_frame, weight=3)

        # Frame para resultados
        results_frame = ttk.Labelframe(content_frame, text="Resultados del Cálculo",
                                       style="Card.TLabelframe", padding=8)

        # Crear área de texto para resultados
        results_text = tk.Text(results_frame, height=20, wrap="word", width=35)
        results_text.pack(fill="both", expand=True)

        # Configurar estilo del texto
        results_text.configure(
            background="#1E1E1E",
            foreground="#FFFFFF",
            font=("Cascadia Code", 10),
            padx=12,
            pady=12,
            spacing1=2,
            spacing2=1,
            spacing3=2
        )

        # Escribir resultados
        results_text.insert(tk.END, "RESULTADOS DEL MÉTODO DE BISECCIÓN\n")

        results_text.insert(tk.END, "FUNCIÓN ANALIZADA:\n")
        results_text.insert(tk.END, f"f(x) = {self._convert_to_display(func_str)}\n\n")

        results_text.insert(tk.END, "INTERVALO INICIAL:\n")
        results_text.insert(tk.END, f"a = {a:.8f}\n")
        results_text.insert(tk.END, f"b = {b:.8f}\n\n")

        results_text.insert(tk.END, "RESULTADO FINAL:\n")
        results_text.insert(tk.END, "─" * 20 + "\n")
        results_text.insert(tk.END, f"Raíz aproximada: {raiz:.10f}\n")
        results_text.insert(tk.END, f"f(raíz) ≈ {f(raiz):.2e}\n")
        results_text.insert(tk.END, f"Iteraciones: {len(pasos)}\n")

        # Calcular error final
        if len(pasos) > 0:
            error_final = pasos[-1].get("error", 0)
            if error_final == error_final:  # No es NaN
                results_text.insert(tk.END, f"Error final: {error_final:.10f}\n")
            else:
                results_text.insert(tk.END, f"Error final: —\n")
        else:
            results_text.insert(tk.END, f"Error final: —\n")

        results_text.insert(tk.END, "CONFIGURACIÓN:\n")
        results_text.insert(tk.END, f"Tolerancia: {float(self.tol_entry.get()):.8f}\n")

        results_text.config(state="disabled")
        content_frame.add(results_frame, weight=1)


    def _mostrar_resultados_biseccion(self, pasos, raiz, motivo):
        """Muestra los resultados de la bisección en la tabla"""
        for row in pasos:
            def fmt(v):
                try:
                    return f"{v:.8g}"
                except:
                    return str(v)

            # Formatear error (puede ser NaN en primera iteración)
            error = row.get("error", 0)
            error_str = "—" if (error != error) else fmt(error)

            self.bis_tree.insert("", "end", values=(
                row["k"], fmt(row["a"]), fmt(row["b"]), fmt(row["c"]),
                fmt(row["fa"]), fmt(row["fb"]), fmt(row["fc"]), error_str
            ))

        # Resaltar última iteración
        if pasos:
            last_iid = self.bis_tree.get_children()[-1]
            self.bis_tree.selection_set(last_iid)
            self.bis_tree.focus(last_iid)

        self._update_status("Bisección completada")

    # -------- Falsa Posicion --------
    def _tab_falsa_posicion(self):
        tab = ttk.Frame(self.nb)
        self.nb.add(tab, text="Falsa Posición")

        frame = ttk.Frame(tab, padding=10)
        frame.pack(fill="both", expand=True)

        # --- Título y descripción
        title_frame = ttk.Frame(frame)
        title_frame.pack(fill="x", pady=(0, 10))
        ttk.Label(title_frame, text="Método de la Falsa Posición (Regula Falso)",
                font=("Times New Roman", 14, "bold")).pack(anchor="w")
        ttk.Label(title_frame, text="Resuelve f(x)=0 en [a,b] usando intersecciones lineales sucesivas.",
                font=("Times New Roman", 10)).pack(anchor="w")

        # --- Entrada de función
        func_frame = ttk.LabelFrame(frame, text="Función", padding=8)
        func_frame.pack(fill="x")
        ttk.Label(func_frame, text="f(x) =").pack(side="left", padx=(4, 6))
        self.fx_entry_fp = ttk.Entry(func_frame, width=48)
        self.fx_entry_fp.pack(side="left", fill="x", expand=True)
        self.fx_entry_fp.insert(0, "x**3 - x - 2")  # ejemplo

        # --- Parámetros
        params = ttk.LabelFrame(frame, text="Parámetros", padding=8)
        params.pack(fill="x", pady=10)

        ttk.Label(params, text="a =").grid(row=0, column=0, sticky="e", padx=4, pady=2)
        self.a_entry_fp = ttk.Entry(params, width=12); self.a_entry_fp.grid(row=0, column=1, padx=4, pady=2)
        self.a_entry_fp.insert(0, "1")

        ttk.Label(params, text="b =").grid(row=0, column=2, sticky="e", padx=4, pady=2)
        self.b_entry_fp = ttk.Entry(params, width=12); self.b_entry_fp.grid(row=0, column=3, padx=4, pady=2)
        self.b_entry_fp.insert(0, "2")

        ttk.Label(params, text="Tolerancia:").grid(row=0, column=4, sticky="e", padx=4, pady=2)
        self.tol_entry_fp = ttk.Entry(params, width=12); self.tol_entry_fp.grid(row=0, column=5, padx=4, pady=2)
        self.tol_entry_fp.insert(0, "0.00001")

        # --- Botones
        auto_frame = ttk.Frame(frame)
        auto_frame.pack(fill="x", pady=(0, 8))
        ttk.Button(auto_frame, text="Buscar Intervalo Automático",
                command=self._buscar_intervalo_valido_fp).pack(side="left", padx=6)
        ttk.Button(auto_frame, text="Graficar función",
                command=self._graficar_funcion_fp).pack(side="left", padx=6)
        ttk.Button(auto_frame, text="Calcular Falsa Posición",
                style="Accent.TButton", command=self._calc_falsa_posicion).pack(side="left", padx=6)
        ttk.Button(auto_frame, text="Limpiar",
                command=self._limpiar_fp).pack(side="left", padx=6)

        # --- Tabla de iteraciones
        table_frame = ttk.LabelFrame(frame, text="Iteraciones del Método de Falsa Posición", padding=8)
        table_frame.pack(fill="both", expand=True, pady=10)

        columns = ("k","a","b","c","f(a)","f(b)","f(c)","error")
        self.fp_tree = ttk.Treeview(table_frame, columns=columns, show="headings", height=15)
        for col, texto, w in zip(columns, ["k","a","b","c","f(a)","f(b)","f(c)","Error"],
                                [50, 90, 90, 90, 100, 100, 100, 90]):
            self.fp_tree.heading(col, text=texto)
            self.fp_tree.column(col, width=w, anchor="center")
        yscroll = ttk.Scrollbar(table_frame, orient="vertical", command=self.fp_tree.yview)
        self.fp_tree.configure(yscrollcommand=yscroll.set)
        self.fp_tree.pack(side="left", fill="both", expand=True)
        yscroll.pack(side="right", fill="y")

        # --- Estado
        status = ttk.Frame(frame); status.pack(fill="x")
        self.fp_status = ttk.Label(status, text="Listo para calcular", font=("Times New Roman", 9))
        self.fp_status.pack(anchor="w")

    def _calc_falsa_posicion(self):
        # limpiar tabla
        for it in self.fp_tree.get_children():
            self.fp_tree.delete(it)
        try:
            func_str = self.fx_entry_fp.get().strip()
            if not func_str:
                raise ValueError("Ingresa una expresión para f(x).")
            f = self._parse_calculation(func_str)
            a = float(self.a_entry_fp.get()); b = float(self.b_entry_fp.get())
            tol_txt = (self.tol_entry_fp.get() or "").strip()
            tol = float(tol_txt) if tol_txt not in ("",) else 1e-6
            if a >= b:
                raise ValueError("Debe cumplirse a < b.")
            from numericos import falsa_posicion
            raiz, pasos, motivo = falsa_posicion(f, a, b, tol=tol, max_iter=100, usar_error="absoluto")
            # tabla
            for p in pasos:
                self.fp_tree.insert("", "end", values=(
                    p["k"], f"{p['a']:.6f}", f"{p['b']:.6f}", f"{p['c']:.6f}",
                    f"{p['fa']:.6f}", f"{p['fb']:.6f}", f"{p['fc']:.6f}",
                    f"{p['error']:.6f}" if p["error"] != float('inf') else "—"
                ))
            # popup
            self._mostrar_resultados_popup_fp(func_str, f, a, b, raiz, pasos, motivo)
            self.fp_status.config(text=f"Falsa Posición completada - {len(pasos)} iteraciones")
            if hasattr(self, "lbl_result"):
                self.lbl_result.config(text=f"Raíz ≈ {raiz:.8f} ({motivo})")
        except Exception as e:
            messagebox.showerror("Error en Falsa Posición", str(e))

    def _mostrar_resultados_popup_fp(self, func_str, f, a, b, raiz, pasos, motivo):
        popup = tk.Toplevel(self)
        popup.title("Resultados - Método de Falsa Posición")
        popup.geometry("1020x600"); popup.minsize(800, 600)
        popup.transient(self); popup.grab_set()
        configurar_estilo_oscuro(popup)

        main = ttk.Frame(popup, padding=10); main.pack(fill="both", expand=True)
        ttk.Label(main, text="Resultados del Método de Falsa Posición",
                font=("Times New Roman", 13, "bold")).pack(anchor="w", pady=(0,6))

        top = ttk.Frame(main); top.pack(fill="both", expand=True)

        # --- Gráfica
        fig = plt.figure(figsize=(6.2, 3.6), dpi=100)
        ax = fig.add_subplot(111)
        margen = 0.1 * (b - a if b != a else 1.0)
        xs = np.linspace(a - margen, b + margen, 600)
        ys = []
        for x in xs:
            try: ys.append(f(x))
            except Exception: ys.append(np.nan)
        ax.plot(xs, ys, linewidth=2, label=f"f(x) = {self._convert_to_display(func_str)}")
        ax.axhline(0, color="k", linestyle="--", alpha=0.7)
        ax.axvline(a, color="r", linestyle="--", alpha=0.7, label=f"a={a:.4f}")
        ax.axvline(b, color="g", linestyle="--", alpha=0.7, label=f"b={b:.4f}")
        ax.axvline(raiz, color="c", linestyle="-.", alpha=0.9, label=f"c≈{raiz:.6f}")
        ax.set_xlabel("x"); ax.set_ylabel("f(x)")
        ax.set_title("Vista de f(x) e iterado final")
        ax.grid(True, alpha=0.3); ax.legend(loc="best")

        left = ttk.Labelframe(top, text="Gráfica", padding=6)
        left.pack(side="left", fill="both", expand=True, padx=(0,8))

        canvas = FigureCanvasTkAgg(fig, master=left)
        canvas.draw()

        toolbar = NavigationToolbar2Tk(canvas, left)
        toolbar.update()
        toolbar.pack(side="bottom", fill="x")

        canvas.get_tk_widget().pack(fill="both", expand=True)

        # --- Resumen y tabla compacta
        right = ttk.Labelframe(top, text="Resumen", padding=8)
        right.pack(side="left", fill="both", expand=True)

        resumen_txt = (
            f"f(x) = {self._convert_to_display(func_str)}\n"
            f"Intervalo: [{a:.6f}, {b:.6f}]\n"
            f"Raíz ≈ {raiz:.10f}\n"
            f"Iteraciones: {len(pasos)}\n"
            f"Motivo: {motivo}"
        )
        ttk.Label(right, text=resumen_txt, justify="left").pack(anchor="w")

    def _buscar_intervalo_valido_fp(self):
        try:
            func_str = self.fx_entry_fp.get().strip()
            if not func_str:
                raise ValueError("Ingresa una expresión para f(x).")
            f = self._parse_calculation(func_str)
            from numericos import encontrar_intervalo_automatico
            a, b, info = encontrar_intervalo_automatico(f)
            self.a_entry_fp.delete(0, tk.END); self.a_entry_fp.insert(0, f"{a:.6f}")
            self.b_entry_fp.delete(0, tk.END); self.b_entry_fp.insert(0, f"{b:.6f}")
            self.fp_status.config(text=info)
        except Exception as e:
            messagebox.showerror("Búsqueda de intervalo", str(e))

    def _graficar_funcion_fp(self):
        try:
            func_str = self.fx_entry_fp.get().strip()
            if not func_str:
                raise ValueError("Ingresa una expresión para f(x).")
            f = self._parse_calculation(func_str)
            a = -5; b = 5
        except Exception:
            # fallback: intenta encontrar intervalo y reintenta
            f = self._parse_calculation(self.fx_entry_fp.get().strip())
            from numericos import encontrar_intervalo_automatico
            a, b, _ = encontrar_intervalo_automatico(f)
            self.a_entry_fp.delete(0, tk.END); self.a_entry_fp.insert(0, f"{a:.6f}")
            self.b_entry_fp.delete(0, tk.END); self.b_entry_fp.insert(0, f"{b:.6f}")

        if a > b: a, b = b, a
        if a == b: a -= 1.0; b += 1.0

        popup = tk.Toplevel(self)
        popup.title("Vista previa de f(x)")
        popup.geometry("900x550"); popup.minsize(700, 450)
        popup.transient(self); popup.grab_set()
        configurar_estilo_oscuro(popup)

        container = ttk.Frame(popup, padding=10); container.pack(fill="both", expand=True)

        fig = plt.figure(figsize=(8, 4.8), dpi=100); ax = fig.add_subplot(111)
        margen = 0.1 * (b - a if b != a else 1.0)
        xs = np.linspace(a - margen, b + margen, 600)
        ys = []
        for x in xs:
            try: ys.append(f(x))
            except Exception: ys.append(np.nan)
        ax.plot(xs, ys, linewidth=2, label=f"f(x) = {self._convert_to_display(self.fx_entry_fp.get())}")
        ax.axhline(0, color='k', linestyle='--', alpha=0.7)
        ax.axvline(a, color='r', linestyle='--', alpha=0.7, label=f'a = {a:.4f}')
        ax.axvline(b, color='g', linestyle='--', alpha=0.7, label=f'b = {b:.4f}')
        ax.set_xlabel('x'); ax.set_ylabel('f(x)')
        ax.set_title('Vista previa de f(x) en el intervalo'); ax.grid(True, alpha=0.3); ax.legend()

        canvas = FigureCanvasTkAgg(fig, master=container)
        canvas.draw()

        toolbar = NavigationToolbar2Tk(canvas, container)
        toolbar.update()
        toolbar.pack(side="bottom", fill="x")

        canvas.get_tk_widget().pack(fill="both", expand=True)

    def _limpiar_fp(self):
        self.fx_entry_fp.delete(0, tk.END)
        self.a_entry_fp.delete(0, tk.END)
        self.b_entry_fp.delete(0, tk.END)
        self.tol_entry_fp.delete(0, tk.END)
        for it in self.fp_tree.get_children():
            self.fp_tree.delete(it)
        self.fp_status.config(text="Listo para calcular")

    #----------Newton Raphson----------
    def _tab_newton_raphson(self):
        tab = ttk.Frame(self.nb)
        self.nb.add(tab, text="Newton-Raphson")

        frame = ttk.Frame(tab, padding=10)
        frame.pack(fill="both", expand=True)

        # --- Título
        ttk.Label(frame, text="Método de Newton–Raphson",
                  font=("Times New Roman", 14, "bold")).pack(anchor="w", pady=(0,10))

        # --- Función
        func_frame = ttk.LabelFrame(frame, text="Función f(x)", padding=8)
        func_frame.pack(fill="x", pady=5)

        ttk.Label(func_frame, text="f(x) =").pack(side="left")
        self.fx_entry_nr = ttk.Entry(func_frame, width=48)
        self.fx_entry_nr.pack(side="left", padx=5, fill="x", expand=True)
        self.fx_entry_nr.insert(0, "x**3 - x - 2")

        # --- Parámetros
        params = ttk.LabelFrame(frame, text="Parámetros", padding=8)
        params.pack(fill="x", pady=8)

        ttk.Label(params, text="x₀ =").grid(row=0, column=0, padx=4, pady=2)
        self.x0_entry_nr = ttk.Entry(params, width=12)
        self.x0_entry_nr.grid(row=0, column=1, padx=4)
        self.x0_entry_nr.insert(0, "1.5")

        ttk.Label(params, text="Tolerancia:").grid(row=0, column=2, padx=4)
        self.tol_entry_nr = ttk.Entry(params, width=12)
        self.tol_entry_nr.grid(row=0, column=3, padx=4)
        self.tol_entry_nr.insert(0, "0.00001")

        # --- Botones
        btns = ttk.Frame(frame)
        btns.pack(fill="x", pady=8)

        ttk.Button(btns, text="Graficar función",
                   command=self._graficar_funcion_nr).pack(side="left", padx=6)

        ttk.Button(btns, text="Calcular Newton-Raphson",
                   style="Accent.TButton",
                   command=self._calc_newton_raphson).pack(side="left", padx=6)

        ttk.Button(btns, text="Limpiar",
                   command=self._limpiar_nr).pack(side="left", padx=6)

        # --- Tabla
        table_frame = ttk.LabelFrame(frame, text="Iteraciones", padding=8)
        table_frame.pack(fill="both", expand=True, pady=10)

        columns = ("k", "x", "fx", "dfx", "error")
        self.nr_tree = ttk.Treeview(table_frame, columns=columns, show="headings", height=14)

        for col, text, w in zip(columns,
                                ["k", "xₖ", "f(xₖ)", "f’(xₖ)", "error"],
                                [50,120,120,120,100]):
            self.nr_tree.heading(col, text=text)
            self.nr_tree.column(col, width=w, anchor="center")

        yscroll = ttk.Scrollbar(table_frame, orient="vertical", command=self.nr_tree.yview)
        self.nr_tree.configure(yscrollcommand=yscroll.set)

        self.nr_tree.pack(side="left", fill="both", expand=True)
        yscroll.pack(side="right", fill="y")

        self.nr_status = ttk.Label(frame, text="Listo para calcular")
        self.nr_status.pack(anchor="w")

    def _calc_newton_raphson(self):
        for i in self.nr_tree.get_children():
            self.nr_tree.delete(i)

        try:
            func_str = self.fx_entry_nr.get().strip()
            f = self._parse_calculation(func_str)

            x0 = float(self.x0_entry_nr.get())
            tol = float(self.tol_entry_nr.get())

            from numericos import newton_raphson
            raiz, pasos, motivo = newton_raphson(f, x0, tol=tol)

            # Mostrar tabla
            for p in pasos:
                self.nr_tree.insert("", "end", values=(
                    p["k"],
                    f"{p['x']:.8f}",
                    f"{p['fx']:.8f}",
                    f"{p['dfx']:.8f}",
                    f"{p['error']:.8f}" if p["error"] != float('inf') else "—"
                ))

            self.nr_status.config(text=f"Raíz ≈ {raiz:.8f} ({motivo})")

            self._mostrar_popup_nr(func_str, f, x0, raiz, pasos, motivo)

            if hasattr(self, "lbl_result"):
                self.lbl_result.config(text=f"Raíz ≈ {raiz:.8f}")

        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _mostrar_popup_nr(self, func_str, f, x0, raiz, pasos, motivo):
        popup = tk.Toplevel(self)
        popup.title("Resultados - Newton-Raphson")
        popup.geometry("950x600")
        popup.transient(self)
        popup.grab_set()
        configurar_estilo_oscuro(popup)

        main = ttk.Frame(popup, padding=10)
        main.pack(fill="both", expand=True)

        ttk.Label(main, text="Newton-Raphson", font=("Times New Roman", 14, "bold")).pack(anchor="w")

        paned = ttk.Panedwindow(main, orient="horizontal")
        paned.pack(fill="both", expand=True)

        # --- Gráfica
        fig = plt.figure(figsize=(6,4), dpi=100)
        ax = fig.add_subplot(111)

        import numpy as np
        xs = np.linspace(raiz - 5, raiz + 5, 400)
        ys = [f(x) for x in xs]

        ax.plot(xs, ys, linewidth=2, label="f(x)")
        ax.axhline(0, color="white")
        ax.axvline(raiz, color="orange", linestyle="--", label=f"Raíz ~ {raiz:.6f}")

        ax.set_title("Gráfica de f(x)")
        ax.grid(True, alpha=0.3)
        ax.legend()

        graph = ttk.Labelframe(paned, text="Gráfica", padding=5)
        canvas = FigureCanvasTkAgg(fig, master=graph)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        paned.add(graph, weight=3)

        # --- Resultados
        resumen = ttk.Labelframe(paned, text="Resumen", padding=10)
        txt = tk.Text(resumen, height=20, wrap="word", background="#1E1E1E",
                      foreground="white", font=("Cascadia Code", 10))
        txt.insert(tk.END,
            f"f(x) = {self._convert_to_display(func_str)}\n"
            f"x0 = {x0}\n\n"
            f"Raíz aproximada:\n  {raiz:.10f}\n\n"
            f"Iteraciones: {len(pasos)}\n"
            f"Motivo de paro:\n  {motivo}\n"
        )
        txt.config(state="disabled")
        txt.pack(fill="both", expand=True)
        paned.add(resumen, weight=1)

    def _graficar_funcion_nr(self):
        try:
            func_str = self.fx_entry_nr.get().strip()
            if not func_str:
                raise ValueError("Ingresa una expresión para f(x).")

            f = self._parse_calculation(func_str)

            # Ventana
            popup = tk.Toplevel(self)
            popup.title("Vista previa de f(x)")
            popup.geometry("900x550")
            configurar_estilo_oscuro(popup)

            # Figura y eje
            fig = plt.figure(figsize=(8, 4.8), dpi=100)
            ax = fig.add_subplot(111)

            # Dominio para graficar
            xs = np.linspace(-10, 10, 500)
            ys = [f(x) for x in xs]

            # Gráfica
            ax.plot(xs, ys, linewidth=2, label=f"f(x) = {self._convert_to_display(func_str)}")
            ax.axhline(0, color='white', linestyle='--', alpha=0.7)
            ax.grid(True, alpha=0.3)
            ax.set_title("f(x)")
            ax.set_xlabel("x")
            ax.set_ylabel("f(x)")
            ax.legend()

            # Canvas + toolbar interactiva
            canvas = FigureCanvasTkAgg(fig, master=popup)
            canvas.draw()

            # Barra de herramientas (zoom, pan, reset, guardar)
            toolbar = NavigationToolbar2Tk(canvas, popup)
            toolbar.update()
            toolbar.pack(side="bottom", fill="x")

            # Colocar la gráfica
            canvas.get_tk_widget().pack(fill="both", expand=True)

        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _limpiar_nr(self):
        self.fx_entry_nr.delete(0, tk.END)
        self.x0_entry_nr.delete(0, tk.END)
        self.tol_entry_nr.delete(0, tk.END)
        for row in self.nr_tree.get_children():
            self.nr_tree.delete(row)
        self.nr_status.config(text="Listo para calcular")

#--------------RUN----------#       
if __name__ == "__main__":
    App().mainloop()
