import threading
import time
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from fraccion import Fraccion
from gauss import GaussJordanEngine, GaussEngine
from matrices import (
    sumar_matrices, multiplicar_matrices,
    multiplicar_escalar_matriz, combinar_escalar_matrices,
    formatear_matriz, Transpuesta, determinante_matriz,
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
    style.configure("TButton", background=fondo, foreground=texto, padding=6, font=("Times New Roman", 10),
                    borderwidth=1, bordercolor=borde)
    style.map("TButton",
              background=[("active", acento), ("pressed", texto)],
              foreground=[("active", "#000000")])
    # --- Treeview (tablas de matrices) ---
    style.configure("Treeview",
                    background="#1E1E1E",  # fondo normal oscuro
                    fieldbackground="#1E1E1E",
                    foreground="#FFFFFF",  # texto blanco normal
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
            ttk.Label(self, text=f"x{j + 1}", anchor="center").grid(row=0, column=j, padx=4, pady=(0, 6))
        if self.allow_b:
            ttk.Label(self, text="|", width=2).grid(row=0, column=self.cols)
            ttk.Label(self, text="b", anchor="center").grid(row=0, column=self.cols + 1, padx=4, pady=(0, 6))

        for i in range(self.rows):
            fila = []
            for j in range(self.cols + (1 if self.allow_b else 0)):
                e = ttk.Entry(self, width=9, justify="center")
                col = j if j < self.cols else j + 1
                e.grid(row=i + 1, column=col, padx=3, pady=3)
                fila.append(e)
            self.entries.append(fila)

            if self.allow_b:
                ttk.Label(self, text="|", width=2).grid(row=i + 1, column=self.cols)

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
                    raise ValueError(f"Valor inválido en fila {i + 1}, columna {j + 1}: '{t}'")
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
                txt = f"x{j + 1}" if j < cols - 1 else "b"
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


# ------------------ Menú de Inicio Mejorado ------------------

class StartMenu(tk.Frame):
    def __init__(self, parent, app_instance):
        super().__init__(parent)
        self.app = app_instance
        self.configure(bg="#1e1e1e")
        self._create_widgets()

    def _create_widgets(self):
        # Frame principal con scroll
        main_frame = tk.Frame(self, bg="#1e1e1e")
        main_frame.pack(fill="both", expand=True)

        # Canvas y scrollbar para contenido desplazable
        canvas = tk.Canvas(main_frame, bg="#1e1e1e", highlightthickness=0)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas, style="TFrame")

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Título principal mejorado
        title_frame = tk.Frame(scrollable_frame, bg="#1e1e1e")
        title_frame.pack(fill="x", pady=(30, 20))

        # Logo/Icono (puedes reemplazar con tu propio logo)
        logo_label = tk.Label(
            title_frame,
            text="🧮",
            font=("Arial", 48),
            bg="#1e1e1e",
            fg="#4fc3f7"
        )
        logo_label.pack(pady=(0, 10))

        title = tk.Label(
            title_frame,
            text="Soluciones de Álgebra Lineal",
            font=("Times New Roman", 28, "bold"),
            bg="#1e1e1e",
            fg="#4fc3f7"
        )
        title.pack(pady=(0, 5))

        subtitle = tk.Label(
            title_frame,
            text="Selecciona un método para comenzar",
            font=("Times New Roman", 14),
            bg="#1e1e1e",
            fg="#cccccc"
        )
        subtitle.pack(pady=5)

        # Línea decorativa
        separator = ttk.Separator(title_frame, orient="horizontal")
        separator.pack(fill="x", padx=100, pady=15)

        # Frame para las categorías
        categories_frame = tk.Frame(scrollable_frame, bg="#1e1e1e")
        categories_frame.pack(fill="both", expand=True, padx=20, pady=10)

        # Categorías de métodos con diseño mejorado
        categories = {
            "📊 Sistemas de Ecuaciones": [
                ("Gauss-Jordan", self.app.show_gauss_jordan),
                ("Gauss", self.app.show_gauss),
                ("Regla de Cramer", self.app.show_cramer)
            ],
            "🧩 Operaciones con Matrices": [
                ("Suma", self.app.show_suma),
                ("Multiplicación", self.app.show_multiplicacion),
                ("Escalar × Matriz", self.app.show_escalar),
                ("Transpuesta", self.app.show_transpuesta),
                ("Inversa", self.app.show_inversa),
                ("Determinante", self.app.show_determinante),
                ("Determinante (Sarrus)", self.app.show_sarrus)
            ],
            "🔍 Análisis de Matrices": [
                ("Independencia Lineal", self.app.show_independencia)
            ],
            "📈 Métodos Numéricos": [
                ("Bisección", self.app.show_biseccion),
                ("Falsa Posición", self.app.show_falsa_posicion),
                ("Newton-Raphson", self.app.show_newton_raphson),
                ("Secante", self.app.show_secante)
            ]
        }

        # Crear categorías en un grid responsive
        row = 0
        col = 0
        max_cols = 4  # Máximo 2 columnas para pantallas grandes

        for category_name, methods in categories.items():
            # Frame para cada categoría con estilo mejorado
            cat_frame = tk.LabelFrame(
                categories_frame,
                text=category_name,
                font=("Times New Roman", 12, "bold"),
                bg="#2d2d2d",
                fg="#4fc3f7",
                padx=15,
                pady=15,
                relief="ridge",
                bd=2
            )
            cat_frame.grid(
                row=row,
                column=col,
                padx=15,
                pady=15,
                sticky="nsew",
                ipadx=10,
                ipady=5
            )

            # Configurar grid weight para expansión
            categories_frame.grid_rowconfigure(row, weight=1)
            categories_frame.grid_columnconfigure(col, weight=1)

            # Botones para cada método en la categoría
            for method_name, command in methods:
                btn = tk.Button(
                    cat_frame,
                    text=method_name,
                    command=command,
                    font=("Times New Roman", 10, "bold"),
                    bg="#2563EB",
                    fg="white",
                    relief="raised",
                    bd=2,
                    padx=25,
                    pady=12,
                    width=20,
                    cursor="hand2"
                )
                btn.pack(pady=6, fill="x")

                # Efectos hover
                btn.bind("<Enter>", lambda e, b=btn: b.configure(bg="#1d4ed8"))
                btn.bind("<Leave>", lambda e, b=btn: b.configure(bg="#2563EB"))

            # Avanzar en el grid
            col += 1
            if col >= max_cols:
                col = 0
                row += 1

        # Footer
        footer_frame = tk.Frame(scrollable_frame, bg="#1e1e1e")
        footer_frame.pack(fill="x", pady=30)

        footer_text = tk.Label(
            footer_frame,
            text="Desarrollado con Python • TKinter • Álgebra Lineal",
            font=("Times New Roman", 10),
            bg="#1e1e1e",
            fg="#666666"
        )
        footer_text.pack()

        # Ajustar scroll al inicio
        canvas.update_idletasks()
        canvas.yview_moveto(0)


# ------------------ App Modificada ------------------

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.root = None
        self.title("Soluciones de Álgebra Lineal")
        self.geometry("1200x800")
        self.minsize(1100, 700)
        configurar_estilo_oscuro(self)

        # Variables de control
        self.engine = None
        self.auto_running = False
        self.auto_thread = None
        self.btn_auto = None
        self.engine_gauss = None
        self.auto_running_gauss = False
        self.auto_thread_gauss = None
        self.btn_auto_gauss = None

        # Crear el menú de inicio
        self.start_menu = StartMenu(self, self)
        self.start_menu.pack(fill="both", expand=True)

        # Inicializar el notebook pero no mostrarlo aún
        self.nb = None
        self.current_tab = None

    def show_single_tab(self, tab_name):
        """Muestra solo la pestaña específica del método seleccionado"""
        # Ocultar menú de inicio
        self.start_menu.pack_forget()

        # Crear notebook si no existe
        if self.nb is None:
            self.nb = ttk.Notebook(self)
            self._build_tabs()

            # Agregar botón de volver al inicio
            self._add_home_button()

        # Ocultar todas las pestañas primero
        for tab in self.nb.tabs():
            self.nb.hide(tab)

        # Buscar y mostrar la pestaña específica
        target_tab = None
        for i, tab in enumerate(self.nb.tabs()):
            if self.nb.tab(tab, "text") == tab_name:
                target_tab = tab
                break

        if target_tab:
            self.nb.select(target_tab)
            self.current_tab = tab_name

        # Mostrar notebook
        self.nb.pack(fill="both", expand=True, padx=10, pady=(0, 8))
        self._build_common_widgets()

    def _add_home_button(self):
        """Agrega un botón para volver al menú de inicio"""
        home_button = ttk.Button(
            self.nb,
            text="Volver al Inicio",
            command=self.show_home,
            style="Accent.TButton"
        )
        # Posicionar el botón en la esquina superior derecha
        home_button.place(relx=0.95, rely=0.02, anchor="ne")

    def _toggle_grid_bisec(self):
        """Alternar la rejilla en la gráfica de bisección"""
        if hasattr(self, 'ax_bisec'):
            self.ax_bisec.grid(not self.ax_bisec.get_gridlines()[0].get_visible())
            self.canvas_bisec.draw()


    def show_home(self):
        """Vuelve al menú de inicio"""
        if self.nb:
            self.nb.pack_forget()
        self.start_menu.pack(fill="both", expand=True)
        self.current_tab = None

    # Métodos para mostrar pestañas específicas (modificados)
    def show_gauss_jordan(self):
        self.show_single_tab("Gauss-Jordan")

    def show_gauss(self):
        self.show_single_tab("Gauss")

    def show_suma(self):
        self.show_single_tab("Suma")

    def show_multiplicacion(self):
        self.show_single_tab("Multiplicación")

    def show_escalar(self):
        self.show_single_tab("Escalar × Matriz")

    def show_transpuesta(self):
        self.show_single_tab("Transpuesta")

    def show_independencia(self):
        self.show_single_tab("Independencia Lineal")

    def show_inversa(self):
        self.show_single_tab("Inversa")

    def show_determinante(self):
        self.show_single_tab("Determinante")

    def show_cramer(self):
        self.show_single_tab("Regla de Cramer")

    def show_sarrus(self):
        self.show_single_tab("Determinante (Sarrus)")

    def show_biseccion(self):
        self.show_single_tab("Método de Bisección")

    def show_falsa_posicion(self):
        self.show_single_tab("Falsa Posición")

    def show_newton_raphson(self):
        self.show_single_tab("Newton-Raphson")

    def show_secante(self):
        self.show_single_tab("Secante")

    def _build_common_widgets(self):
        """Construye los widgets comunes que se muestran después del menú de inicio"""
        # Resultado general + estado (solo si no existen)
        if not hasattr(self, 'lbl_result'):
            result_card = ttk.Labelframe(self, text="Resultado", style="Card.TLabelframe", padding=8)
            result_card.pack(fill="x", padx=10, pady=(0, 8))
            self.lbl_result = ttk.Label(result_card, text="—", style="Title.TLabel")
            self.lbl_result.pack(anchor="w")

            self.status = ttk.Label(self, text="Listo.", style="Status.TLabel")
            self.status.pack(fill="x", side="bottom")

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
    def _build_tabs(self):
        """Construye todas las pestañas (igual que tu código original)"""
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
        self._tab_metodo_biseccion()
        self._tab_falsa_posicion()
        self._tab_newton_raphson()
        self._tab_secante()


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
        top = ttk.Frame(tab);
        top.pack(fill="x", padx=8, pady=(8, 4))
        ttk.Label(top, text="Ecuaciones (m):").pack(side="left", padx=(2, 4))
        self.spin_m = tk.Spinbox(top, from_=1, to=12, width=4)
        self.spin_m.delete(0, "end");
        self.spin_m.insert(0, "3");
        self.spin_m.pack(side="left")
        ttk.Label(top, text="Incógnitas (n):").pack(side="left", padx=(10, 4))
        self.spin_n = tk.Spinbox(top, from_=1, to=12, width=4)
        self.spin_n.delete(0, "end");
        self.spin_n.insert(0, "3");
        self.spin_n.pack(side="left", padx=(0, 8))
        ttk.Button(top, text="Redimensionar", command=self.resize_matrix).pack(side="left", padx=(0, 6))
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

    # -------- Tab 4: Escalar × Matriz / Combinaciones --------
    def _tab_escalar(self):
        tab = ttk.Frame(self.nb)
        self.nb.add(tab, text="Escalar × Matriz")

        frame = ttk.Frame(tab, padding=10)
        frame.pack(fill="both", expand=True)

        # --- Tipo de operación ---
        tipo_frame = ttk.Frame(frame)
        tipo_frame.pack(fill="x", pady=(0, 6))
        ttk.Label(tipo_frame, text="Tipo de operación:").pack(side="left")

        self.es_tipo = tk.StringVar(value="Escalar × A")
        self.es_tipo_cb = ttk.Combobox(
            tipo_frame,
            textvariable=self.es_tipo,
            values=[
                "Escalar × A",
                "(Escalar × A) × B",
                "αA ± βB",
                "A(u+v)",
                "Au + Av"
            ],
            state="readonly",
            width=22
        )
        self.es_tipo_cb.pack(side="left", padx=6)
        self.es_tipo_cb.current(0)

        # --- Tamaño matriz A ---
        sizeA_frame = ttk.Frame(frame)
        sizeA_frame.pack(fill="x")
        ttk.Label(sizeA_frame, text="Matriz A - Filas:").pack(side="left")
        self.es_rA = tk.Spinbox(sizeA_frame, from_=1, to=12, width=5)
        self.es_rA.pack(side="left", padx=4)
        ttk.Label(sizeA_frame, text="Columnas:").pack(side="left")
        self.es_cA = tk.Spinbox(sizeA_frame, from_=1, to=12, width=5)
        self.es_cA.pack(side="left", padx=4)
        self.es_btnA = ttk.Button(
            sizeA_frame, text="Redimensionar A",
            command=self._escalar_resize_A
        )
        self.es_btnA.pack(side="left", padx=10)

        # --- Tamaño matriz B ---
        sizeB_frame = ttk.Frame(frame)
        sizeB_frame.pack(fill="x", pady=(4, 0))
        ttk.Label(sizeB_frame, text="Matriz B / u - Filas:").pack(side="left")
        self.es_rB = tk.Spinbox(sizeB_frame, from_=1, to=12, width=5)
        self.es_rB.pack(side="left", padx=4)
        ttk.Label(sizeB_frame, text="Columnas:").pack(side="left")
        self.es_cB = tk.Spinbox(sizeB_frame, from_=1, to=12, width=5)
        self.es_cB.pack(side="left", padx=4)
        self.es_btnB = ttk.Button(
            sizeB_frame, text="Redimensionar B",
            command=self._escalar_resize_B
        )
        self.es_btnB.pack(side="left", padx=10)

        # --- Tamaño matriz C ---
        sizeC_frame = ttk.Frame(frame)
        sizeC_frame.pack(fill="x", pady=(4, 4))
        ttk.Label(sizeC_frame, text="Matriz C / v - Filas:").pack(side="left")
        self.es_rC = tk.Spinbox(sizeC_frame, from_=1, to=12, width=5)
        self.es_rC.pack(side="left", padx=4)
        ttk.Label(sizeC_frame, text="Columnas:").pack(side="left")
        self.es_cC = tk.Spinbox(sizeC_frame, from_=1, to=12, width=5)
        self.es_cC.pack(side="left", padx=4)
        self.es_btnC = ttk.Button(
            sizeC_frame, text="Redimensionar C",
            command=self._escalar_resize_C
        )
        self.es_btnC.pack(side="left", padx=10)

        # --- Escalares y operador ---
        scal_frame = ttk.Frame(frame)
        scal_frame.pack(fill="x", pady=(4, 4))

        ttk.Label(scal_frame, text="Escalar A (α / λ):").pack(side="left")
        self.es_valA = ttk.Entry(scal_frame, width=8)
        self.es_valA.pack(side="left", padx=4)

        ttk.Label(scal_frame, text="Operador (para αA ± βB):").pack(side="left", padx=(10, 0))
        self.es_op = ttk.Combobox(
            scal_frame, values=["+", "-"],
            width=3, state="readonly"
        )
        self.es_op.set("-")
        self.es_op.pack(side="left", padx=4)

        ttk.Label(scal_frame, text="Escalar B (β):").pack(side="left", padx=(10, 0))
        self.es_valB = ttk.Entry(scal_frame, width=8)
        self.es_valB.pack(side="left", padx=4)

        # --- Matrices A, B, C ---
        matrices_frame = ttk.Frame(frame)
        matrices_frame.pack(fill="x", pady=8)

        # Matriz A
        ttk.Label(matrices_frame, text="Matriz A").pack(anchor="w")
        self.es_A = MatrixInput(matrices_frame, rows=2, cols=2, allow_b=False)
        self.es_A.pack(fill="x", pady=(0, 8))

        # Matriz B / u
        ttk.Label(
            matrices_frame,
            text="Matriz B (se usa como B, o como vector u en A(u+v), Au+Av)"
        ).pack(anchor="w")
        self.es_B = MatrixInput(matrices_frame, rows=2, cols=1, allow_b=False)
        self.es_B.pack(fill="x", pady=(0, 8))

        # Matriz C / v
        ttk.Label(
            matrices_frame,
            text="Matriz C (solo para vectores v en A(u+v) y Au+Av)"
        ).pack(anchor="w")
        self.es_C = MatrixInput(matrices_frame, rows=2, cols=1, allow_b=False)
        self.es_C.pack(fill="x")

        # --- Botones ---
        btns = ttk.Frame(frame)
        btns.pack(fill="x", pady=4)
        ttk.Button(
            btns, text="Calcular",
            style="Accent.TButton",
            command=self._escalar_calc
        ).pack(side="left")
        self.es_next = ttk.Button(
            btns, text="Siguiente paso",
            command=self._escalar_next,
            state="disabled"
        )
        self.es_next.pack(side="left", padx=6)
        ttk.Button(
            btns, text="Exportar log",
            command=self._escalar_export
        ).pack(side="left")

        # --- Área de texto para pasos ---
        self.es_txt = make_text(frame, height=18, wrap="word")
        self.es_txt.pack(fill="both", expand=True)

        self._es_pasos = []
        self._es_idx = 0
        self._es_resultado = []

        # Vincular cambio de modo
        self.es_tipo_cb.bind("<<ComboboxSelected>>", self._escalar_on_mode_change)
        # Estado inicial
        self._escalar_on_mode_change()

    # -------- Helpers para activar/desactivar entradas --------
    def _set_enabled_recursive(self, widget, enabled: bool):
        state = "normal" if enabled else "disabled"
        try:
            # No desactivar el combobox de tipo de operación
            if widget is self.es_tipo_cb:
                return
            widget.configure(state=state)
        except tk.TclError:
            pass
        for child in widget.winfo_children():
            self._set_enabled_recursive(child, enabled)

    def _escalar_on_mode_change(self, event=None):
        modo = self.es_tipo.get()

        # Por defecto deshabilitamos todo y luego activamos lo que toca
        # A
        self._set_enabled_recursive(self.es_A, False)
        self._set_enabled_recursive(self.es_rA, False)
        self._set_enabled_recursive(self.es_cA, False)
        self._set_enabled_recursive(self.es_btnA, False)

        # B
        self._set_enabled_recursive(self.es_B, False)
        self._set_enabled_recursive(self.es_rB, False)
        self._set_enabled_recursive(self.es_cB, False)
        self._set_enabled_recursive(self.es_btnB, False)

        # C
        self._set_enabled_recursive(self.es_C, False)
        self._set_enabled_recursive(self.es_rC, False)
        self._set_enabled_recursive(self.es_cC, False)
        self._set_enabled_recursive(self.es_btnC, False)

        # Escalares y operador
        self._set_enabled_recursive(self.es_valA, False)
        self._set_enabled_recursive(self.es_valB, False)
        self._set_enabled_recursive(self.es_op, False)

        # Ahora activamos según el modo
        if modo == "Escalar × A":
            # Usa A y escalar A
            self._set_enabled_recursive(self.es_A, True)
            self._set_enabled_recursive(self.es_rA, True)
            self._set_enabled_recursive(self.es_cA, True)
            self._set_enabled_recursive(self.es_btnA, True)
            self._set_enabled_recursive(self.es_valA, True)

        elif modo == "(Escalar × A) × B":
            # Usa A, escalar A y B
            self._set_enabled_recursive(self.es_A, True)
            self._set_enabled_recursive(self.es_rA, True)
            self._set_enabled_recursive(self.es_cA, True)
            self._set_enabled_recursive(self.es_btnA, True)
            self._set_enabled_recursive(self.es_valA, True)

            self._set_enabled_recursive(self.es_B, True)
            self._set_enabled_recursive(self.es_rB, True)
            self._set_enabled_recursive(self.es_cB, True)
            self._set_enabled_recursive(self.es_btnB, True)

        elif modo == "αA ± βB":
            # Usa A, B, escalar A, escalar B y operador
            self._set_enabled_recursive(self.es_A, True)
            self._set_enabled_recursive(self.es_rA, True)
            self._set_enabled_recursive(self.es_cA, True)
            self._set_enabled_recursive(self.es_btnA, True)

            self._set_enabled_recursive(self.es_B, True)
            self._set_enabled_recursive(self.es_rB, True)
            self._set_enabled_recursive(self.es_cB, True)
            self._set_enabled_recursive(self.es_btnB, True)

            self._set_enabled_recursive(self.es_valA, True)
            self._set_enabled_recursive(self.es_valB, True)
            self._set_enabled_recursive(self.es_op, True)

        elif modo == "A(u+v)":
            # Usa A, B (u) y C (v) como vectores; no escalares
            self._set_enabled_recursive(self.es_A, True)
            self._set_enabled_recursive(self.es_rA, True)
            self._set_enabled_recursive(self.es_cA, True)
            self._set_enabled_recursive(self.es_btnA, True)

            self._set_enabled_recursive(self.es_B, True)
            self._set_enabled_recursive(self.es_rB, True)
            self._set_enabled_recursive(self.es_cB, True)
            self._set_enabled_recursive(self.es_btnB, True)

            self._set_enabled_recursive(self.es_C, True)
            self._set_enabled_recursive(self.es_rC, True)
            self._set_enabled_recursive(self.es_cC, True)
            self._set_enabled_recursive(self.es_btnC, True)

        else:  # "Au + Av"
            # Usa A, B (u) y C (v); no escalares
            self._set_enabled_recursive(self.es_A, True)
            self._set_enabled_recursive(self.es_rA, True)
            self._set_enabled_recursive(self.es_cA, True)
            self._set_enabled_recursive(self.es_btnA, True)

            self._set_enabled_recursive(self.es_B, True)
            self._set_enabled_recursive(self.es_rB, True)
            self._set_enabled_recursive(self.es_cB, True)
            self._set_enabled_recursive(self.es_btnB, True)

            self._set_enabled_recursive(self.es_C, True)
            self._set_enabled_recursive(self.es_rC, True)
            self._set_enabled_recursive(self.es_cC, True)
            self._set_enabled_recursive(self.es_btnC, True)

    # -------- Redimensionar matrices --------
    def _escalar_resize_A(self):
        try:
            r = int(self.es_rA.get())
            c = int(self.es_cA.get())
        except ValueError:
            messagebox.showerror("Error", "Filas y columnas de A deben ser enteros.")
            return
        self.es_A.set_size(r, c)

    def _escalar_resize_B(self):
        try:
            r = int(self.es_rB.get())
            c = int(self.es_cB.get())
        except ValueError:
            messagebox.showerror("Error", "Filas y columnas de B deben ser enteros.")
            return
        self.es_B.set_size(r, c)

    def _escalar_resize_C(self):
        try:
            r = int(self.es_rC.get())
            c = int(self.es_cC.get())
        except ValueError:
            messagebox.showerror("Error", "Filas y columnas de C deben ser enteros.")
            return
        self.es_C.set_size(r, c)

    # -------- Cálculo principal (igual que antes) --------
    def _escalar_calc(self):
        try:
            modo_str = self.es_tipo.get()

            # Siempre leemos A
            A = self.es_A.get_matrix()
            escA = (self.es_valA.get() or "").strip()
            if not escA:
                escA = "1"  # por defecto 1

            if modo_str == "Escalar × A":
                R, pasos = multiplicar_escalar_matriz(escA, A)

            elif modo_str == "(Escalar × A) × B":
                B = self.es_B.get_matrix()
                if len(A[0]) != len(B):
                    raise ValueError(
                        "Para (Escalar × A) × B, columnas de A deben ser igual a filas de B."
                    )
                R_escalar, pasos1 = multiplicar_escalar_matriz(escA, A)
                R, pasos2 = multiplicar_matrices(R_escalar, B)

                pasos = []
                pasos.append(f"Operación: ({escA}·A) × B")
                pasos.append("")
                pasos.extend(pasos1)
                pasos.append("")
                pasos.extend(pasos2)

            elif modo_str == "αA ± βB":
                B = self.es_B.get_matrix()
                escB = (self.es_valB.get() or "").strip()
                if not escB:
                    escB = "1"

                op = self.es_op.get()
                if op not in ("+", "-"):
                    op = "-"

                if len(A) != len(B) or len(A[0]) != len(B[0]):
                    raise ValueError(
                        "Para αA ± βB, A y B deben tener las mismas dimensiones."
                    )

                R, pasos = combinar_escalar_matrices(escA, A, escB, B, operador=op)

            elif modo_str == "A(u+v)":
                u = self.es_B.get_matrix()
                v = self.es_C.get_matrix()

                if len(u) != len(v) or len(u[0]) != len(v[0]):
                    raise ValueError("u y v deben tener las mismas dimensiones.")

                if len(A[0]) != len(u):
                    raise ValueError(
                        "Dimensiones incompatibles: columnas de A deben ser igual "
                        "a filas de u y v."
                    )

                suma_uv, pasos1 = sumar_matrices(u, v)
                R, pasos2 = multiplicar_matrices(A, suma_uv)

                pasos = []
                pasos.append("Operación: A(u+v)")
                pasos.append("")
                pasos.append("Cálculo de u + v:")
                pasos.extend(pasos1)
                pasos.append("")
                pasos.append("Cálculo de A(u+v):")
                pasos.extend(pasos2)

            else:  # "Au + Av"
                u = self.es_B.get_matrix()
                v = self.es_C.get_matrix()

                if len(A[0]) != len(u) or len(A[0]) != len(v):
                    raise ValueError(
                        "Dimensiones incompatibles: columnas de A deben ser igual "
                        "a filas de u y de v."
                    )

                Au, pasos_Au = multiplicar_matrices(A, u)
                Av, pasos_Av = multiplicar_matrices(A, v)
                suma, pasos_sum = sumar_matrices(Au, Av)

                pasos = []
                pasos.append("Operación: Au + Av")
                pasos.append("")
                pasos.append("Cálculo de Au:")
                pasos.extend(pasos_Au)
                pasos.append("")
                pasos.append("Cálculo de Av:")
                pasos.extend(pasos_Av)
                pasos.append("")
                pasos.append("Suma Au + Av:")
                pasos.extend(pasos_sum)

                R = suma

        except Exception as e:
            messagebox.showerror("Error", str(e))
            return

        self._es_resultado = R
        self._es_pasos = pasos
        self._es_idx = 0

        self.es_txt.delete(1.0, tk.END)
        self.es_txt.insert(
            tk.END,
            "Cálculo inicializado. Presione 'Siguiente paso'.\n"
        )
        self.es_next.config(state="normal")

    def _escalar_next(self):
        if self._es_idx < len(self._es_pasos):
            self.es_txt.insert(
                tk.END,
                self._es_pasos[self._es_idx] + "\n\n"
            )
            self._es_idx += 1
            if self._es_idx == len(self._es_pasos):
                self.es_next.config(state="disabled")
        else:
            self.es_next.config(state="disabled")

    def _escalar_export(self):
        if not self._es_pasos:
            messagebox.showinfo("Info", "No hay pasos para exportar.")
            return
        fp = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Texto", "*.txt")],
            title="Guardar registro de pasos"
        )
        if not fp:
            return
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

    # -------- Métodos numéricos (raíces por Bisección) -------

    def _tab_metodo_biseccion(self):
        tab = ttk.Frame(self.nb)
        self.nb.add(tab, text="Método de Bisección")

        # Frame principal
        main_frame = ttk.Frame(tab, padding=10)
        main_frame.pack(fill="both", expand=True)

        # --- PARTE SUPERIOR: Función + Gráfica ---
        top_paned = ttk.PanedWindow(main_frame, orient="horizontal")
        top_paned.pack(fill="both", expand=True, pady=(0, 10))

        # --- COLUMNA IZQUIERDA: Calculadora Avanzada ---
        left_frame = ttk.Labelframe(top_paned, text="Calculadora Avanzada", style="Card.TLabelframe", padding=15)

        # Display de la función con LaTeX
        func_display_frame = ttk.Frame(left_frame)
        func_display_frame.pack(fill="x", pady=(0, 15))

        # Frame para el display LaTeX
        self.latex_frame = ttk.Frame(func_display_frame, height=80)
        self.latex_frame.pack(fill="x", pady=5)
        self.latex_frame.pack_propagate(False)

        # Inicializar display LaTeX
        self._inicializar_latex_display()

        # Entrada de función
        input_frame = ttk.Frame(left_frame)
        input_frame.pack(fill="x", pady=10)

        ttk.Label(input_frame, text="f(x) =", font=("Arial", 12, "bold")).pack(side="left")
        self.fx_entry_bisec = ttk.Entry(
            input_frame,
            font=("Courier New", 12),
            width=40
        )
        self.fx_entry_bisec.pack(side="left", fill="x", expand=True, padx=10)
        self.fx_entry_bisec.insert(0, "x**3 - 3*x**2")
        self.fx_entry_bisec.bind('<KeyRelease>', self._on_function_change_bisec)

        # --- CALCULADORA COMPLETA ---
        calc_frame = ttk.Frame(left_frame)
        calc_frame.pack(fill="x", pady=10)

        # Fila 1: Botones superiores
        row0 = ttk.Frame(calc_frame)
        row0.pack(fill="x", pady=2)

        buttons_row0 = [
            ("2nd", lambda: self._toggle_second_functions(), "#666666"),
            ("const", lambda: self._insert_constant(), "#666666"),
            ("T", lambda: self._insert_variable("t"), "#666666"),
            ("e", lambda: self._insert_constant("e"), "#666666"),
            ("[::]", lambda: self._insert_matrix(), "#666666"),
            ("x", lambda: self._insert_variable("x"), "#4CAF50"),
            ("(", lambda: self._insert_operator("("), "#2196F3"),
            (",", lambda: self._insert_operator(","), "#2196F3"),
            (")", lambda: self._insert_operator(")"), "#2196F3"),
            ("⇌", lambda: self._clear_entry(), "#FF5722")
        ]

        for i, (text, command, color) in enumerate(buttons_row0):
            btn = tk.Button(
                row0,
                text=text,
                font=("Arial", 10, "bold"),
                bg=color,
                fg="white",
                relief="raised",
                bd=2,
                width=4,
                height=1,
                command=command
            )
            btn.pack(side="left", padx=1, pady=1)

        # Fila 2: Funciones trigonométricas
        row1 = ttk.Frame(calc_frame)
        row1.pack(fill="x", pady=2)

        buttons_row1 = [
            ("sin", lambda: self._insert_function("sin"), "#9C27B0"),
            ("sinh", lambda: self._insert_function("sinh"), "#9C27B0"),
            ("cot", lambda: self._insert_function("cot"), "#9C27B0"),
            ("y√x", lambda: self._insert_function("yroot"), "#FF9800"),
            ("xʸ", lambda: self._insert_operator("**"), "#FF9800"),
            ("7", lambda: self._insert_digit("7"), "#37474F"),
            ("8", lambda: self._insert_digit("8"), "#37474F"),
            ("9", lambda: self._insert_digit("9"), "#37474F"),
            ("÷", lambda: self._insert_operator("/"), "#2196F3")
        ]

        for i, (text, command, color) in enumerate(buttons_row1):
            btn = tk.Button(
                row1,
                text=text,
                font=("Arial", 10, "bold"),
                bg=color,
                fg="white",
                relief="raised",
                bd=2,
                width=4,
                height=1,
                command=command
            )
            btn.pack(side="left", padx=1, pady=1)

        # Fila 3: Más funciones
        row2 = ttk.Frame(calc_frame)
        row2.pack(fill="x", pady=2)

        buttons_row2 = [
            ("cos", lambda: self._insert_function("cos"), "#9C27B0"),
            ("cosh", lambda: self._insert_function("cosh"), "#9C27B0"),
            ("sec", lambda: self._insert_function("sec"), "#9C27B0"),
            ("³√x", lambda: self._insert_function("cbrt"), "#FF9800"),
            ("x³", lambda: self._insert_power("3"), "#FF9800"),
            ("4", lambda: self._insert_digit("4"), "#37474F"),
            ("5", lambda: self._insert_digit("5"), "#37474F"),
            ("6", lambda: self._insert_digit("6"), "#37474F"),
            ("×", lambda: self._insert_operator("*"), "#2196F3")
        ]

        for i, (text, command, color) in enumerate(buttons_row2):
            btn = tk.Button(
                row2,
                text=text,
                font=("Arial", 10, "bold"),
                bg=color,
                fg="white",
                relief="raised",
                bd=2,
                width=4,
                height=1,
                command=command
            )
            btn.pack(side="left", padx=1, pady=1)

        # Fila 4: Funciones adicionales
        row3 = ttk.Frame(calc_frame)
        row3.pack(fill="x", pady=2)

        buttons_row3 = [
            ("tan", lambda: self._insert_function("tan"), "#9C27B0"),
            ("tanh", lambda: self._insert_function("tanh"), "#9C27B0"),
            ("csc", lambda: self._insert_function("csc"), "#9C27B0"),
            ("√x", lambda: self._insert_function("sqrt"), "#FF9800"),
            ("x²", lambda: self._insert_power("2"), "#FF9800"),
            ("1", lambda: self._insert_digit("1"), "#37474F"),
            ("2", lambda: self._insert_digit("2"), "#37474F"),
            ("3", lambda: self._insert_digit("3"), "#37474F"),
            ("-", lambda: self._insert_operator("-"), "#2196F3")
        ]

        for i, (text, command, color) in enumerate(buttons_row3):
            btn = tk.Button(
                row3,
                text=text,
                font=("Arial", 10, "bold"),
                bg=color,
                fg="white",
                relief="raised",
                bd=2,
                width=4,
                height=1,
                command=command
            )
            btn.pack(side="left", padx=1, pady=1)

        # Fila 5: Última fila
        row4 = ttk.Frame(calc_frame)
        row4.pack(fill="x", pady=2)

        buttons_row4 = [
            ("nCr", lambda: self._insert_combination(), "#E91E63"),
            ("nPr", lambda: self._insert_permutation(), "#E91E63"),
            ("%", lambda: self._insert_operator("%"), "#E91E63"),
            ("log", lambda: self._insert_function("log"), "#FF9800"),
            ("10ˣ", lambda: self._insert_function("exp10"), "#FF9800"),
            ("0", lambda: self._insert_digit("0"), "#37474F"),
            (".", lambda: self._insert_decimal(), "#37474F"),
            ("⇌", lambda: self._backspace(), "#FF5722"),
            ("+", lambda: self._insert_operator("+"), "#2196F3")
        ]

        for i, (text, command, color) in enumerate(buttons_row4):
            btn = tk.Button(
                row4,
                text=text,
                font=("Arial", 10, "bold"),
                bg=color,
                fg="white",
                relief="raised",
                bd=2,
                width=4,
                height=1,
                command=command
            )
            btn.pack(side="left", padx=1, pady=1)

        # Fila 6: Botones de control
        row5 = ttk.Frame(calc_frame)
        row5.pack(fill="x", pady=2)

        buttons_row5 = [
            ("π", lambda: self._insert_constant("pi"), "#607D8B"),
            ("e", lambda: self._insert_constant("e"), "#607D8B"),
            ("∞", lambda: self._insert_constant("inf"), "#607D8B"),
            ("ln", lambda: self._insert_function("ln"), "#FF9800"),
            ("eˣ", lambda: self._insert_function("exp"), "#FF9800"),
            ("(", lambda: self._insert_operator("("), "#2196F3"),
            (")", lambda: self._insert_operator(")"), "#2196F3"),
            ("=", lambda: self._actualizar_grafica_bisec(), "#4CAF50"),
            ("C", lambda: self._clear_entry(), "#F44336")
        ]

        for i, (text, command, color) in enumerate(buttons_row5):
            btn = tk.Button(
                row5,
                text=text,
                font=("Arial", 10, "bold"),
                bg=color,
                fg="white",
                relief="raised",
                bd=2,
                width=4,
                height=1,
                command=command
            )
            btn.pack(side="left", padx=1, pady=1)

        # --- PARÁMETROS DEL MÉTODO ---
        params_frame = ttk.LabelFrame(left_frame, text="Parámetros del Método", padding=10)
        params_frame.pack(fill="x", pady=15)

        # Intervalo
        interval_frame = ttk.Frame(params_frame)
        interval_frame.pack(fill="x", pady=5)

        ttk.Label(interval_frame, text="Intervalo [a, b]:", font=("Arial", 10, "bold")).pack(side="left")
        ttk.Label(interval_frame, text="a =").pack(side="left", padx=(15, 2))
        self.a_entry_bisec = ttk.Entry(interval_frame, width=10, font=("Arial", 10))
        self.a_entry_bisec.pack(side="left", padx=2)
        self.a_entry_bisec.insert(0, "1.0")

        ttk.Label(interval_frame, text="b =").pack(side="left", padx=(10, 2))
        self.b_entry_bisec = ttk.Entry(interval_frame, width=10, font=("Arial", 10))
        self.b_entry_bisec.pack(side="left", padx=2)
        self.b_entry_bisec.insert(0, "3.0")

        # Tolerancia y botones
        control_frame = ttk.Frame(params_frame)
        control_frame.pack(fill="x", pady=10)

        ttk.Label(control_frame, text="Tolerancia:", font=("Arial", 10)).pack(side="left")
        self.tol_entry_bisec = ttk.Entry(control_frame, width=10, font=("Arial", 10))
        self.tol_entry_bisec.pack(side="left", padx=5)
        self.tol_entry_bisec.insert(0, "0.00001")

        ttk.Button(control_frame, text="🔍 Buscar Intervalo",
                   command=self._buscar_intervalo_valido_bisec).pack(side="left", padx=5)
        ttk.Button(control_frame, text="🚀 Calcular Bisección",
                   style="Accent.TButton",
                   command=self._calc_biseccion_new).pack(side="left", padx=5)

        top_paned.add(left_frame, weight=1)

        # --- COLUMNA DERECHA: Gráfica (estilo GeoGebra) ---
        right_frame = ttk.Labelframe(top_paned, text="Gráfica Interactiva", style="Card.TLabelframe", padding=10)

        # Controles de la gráfica
        graph_controls = ttk.Frame(right_frame)
        graph_controls.pack(fill="x", pady=(0, 10))

        ttk.Button(graph_controls, text="🔄 Actualizar",
                   command=self._actualizar_grafica_bisec).pack(side="left")
        ttk.Button(graph_controls, text="➖ Zoom -",
                   command=lambda: self._zoom_grafica_bisec(1.2)).pack(side="left", padx=5)
        ttk.Button(graph_controls, text="➕ Zoom +",
                   command=lambda: self._zoom_grafica_bisec(0.8)).pack(side="left", padx=5)

        # Frame para la gráfica
        self.graph_frame_bisec = ttk.Frame(right_frame)
        self.graph_frame_bisec.pack(fill="both", expand=True)

        # Inicializar gráfica
        self._inicializar_grafica_bisec()

        top_paned.add(right_frame, weight=2)

        # --- PARTE INFERIOR: Tabla de iteraciones ---
        table_frame = ttk.Labelframe(main_frame, text="📊 Tabla de Iteraciones - Método de Bisección",
                                     style="Card.TLabelframe", padding=8)
        table_frame.pack(fill="both", expand=True)

        # Crear tabla
        columns = ("k", "a", "b", "c", "f(a)", "f(b)", "f(c)", "error")
        self.bis_tree_new = ttk.Treeview(table_frame, columns=columns, show="headings", height=12)

        headings = ["Iter", "a", "b", "c", "f(a)", "f(b)", "f(c)", "Error"]
        widths = [60, 100, 100, 100, 120, 120, 120, 100]

        for col, heading, width in zip(columns, headings, widths):
            self.bis_tree_new.heading(col, text=heading)
            self.bis_tree_new.column(col, width=width, anchor="center")

        # Scrollbar para la tabla
        scrollbar_table = ttk.Scrollbar(table_frame, orient="vertical", command=self.bis_tree_new.yview)
        self.bis_tree_new.configure(yscrollcommand=scrollbar_table.set)

        self.bis_tree_new.pack(side="left", fill="both", expand=True)
        scrollbar_table.pack(side="right", fill="y")

        # Estado
        self.biseccion_status_new = ttk.Label(main_frame,
                                              text="🟢 Listo para calcular - Ingresa una función y parámetros")
        self.biseccion_status_new.pack(anchor="w", pady=5)

        # Inicializar gráfica después de crear todos los componentes
        self.after(100, self._actualizar_grafica_bisec)

    def _inicializar_latex_display(self):
        """Inicializa el display LaTeX para la función"""
        try:
            # Crear figura para LaTeX
            self.fig_latex = plt.figure(figsize=(8, 1), dpi=100)
            self.ax_latex = self.fig_latex.add_subplot(111)
            self.ax_latex.axis('off')

            # Canvas para LaTeX
            self.canvas_latex = FigureCanvasTkAgg(self.fig_latex, master=self.latex_frame)
            self.canvas_latex.get_tk_widget().pack(fill="both", expand=True)

            # Mostrar función inicial
            self._actualizar_latex_display("x^3 - 3x^2")

        except Exception as e:
            print(f"Error inicializando LaTeX: {e}")

    def _actualizar_latex_display(self, func_text):
        """Actualiza el display LaTeX con la función"""
        try:
            self.ax_latex.clear()
            self.ax_latex.axis('off')

            # Convertir a formato LaTeX
            latex_text = self._convert_to_latex(func_text)

            # Renderizar LaTeX
            self.ax_latex.text(0.5, 0.5, f"${latex_text}$",
                               fontsize=16, ha='center', va='center',
                               transform=self.ax_latex.transAxes)

            self.canvas_latex.draw()

        except Exception as e:
            print(f"Error actualizando LaTeX: {e}")

    def _convert_to_latex(self, text):
        """Convierte texto de función a formato LaTeX"""
        if not text.strip():
            return "f(x) = "

        # Reemplazos para LaTeX
        latex_text = text

        # Operadores
        latex_text = latex_text.replace('**', '^')
        latex_text = latex_text.replace('*', '\\cdot ')

        # Funciones matemáticas
        func_replacements = {
            'sin': '\\sin', 'cos': '\\cos', 'tan': '\\tan',
            'sinh': '\\sinh', 'cosh': '\\cosh', 'tanh': '\\tanh',
            'cot': '\\cot', 'sec': '\\sec', 'csc': '\\csc',
            'log': '\\log', 'ln': '\\ln', 'sqrt': '\\sqrt',
            'exp': 'e^', 'pi': '\\pi', 'inf': '\\infty'
        }

        for func, latex_func in func_replacements.items():
            latex_text = latex_text.replace(func, latex_func)

        # Raíces
        latex_text = re.sub(r'sqrt\(([^)]+)\)', r'\\sqrt{\1}', latex_text)
        latex_text = re.sub(r'cbrt\(([^)]+)\)', r'\\sqrt[3]{\1}', latex_text)
        latex_text = re.sub(r'yroot\(([^,]+),([^)]+)\)', r'\\sqrt[\1]{\2}', latex_text)

        # Fracciones implícitas
        latex_text = re.sub(r'(\d)/(\d)', r'\\frac{\1}{\2}', latex_text)

        return f"f(x) = {latex_text}"

    # --- FUNCIONES DE LA CALCULADORA ---

    def _insert_digit(self, digit):
        """Inserta un dígito"""
        self.fx_entry_bisec.insert(tk.END, digit)
        self._on_function_change_bisec()

    def _insert_operator(self, op):
        """Inserta un operador"""
        self.fx_entry_bisec.insert(tk.END, op)
        self._on_function_change_bisec()

    def _insert_function(self, func):
        """Inserta una función"""
        if func in ["sin", "cos", "tan", "sinh", "cosh", "tanh", "cot", "sec", "csc", "log", "ln", "sqrt", "cbrt"]:
            self.fx_entry_bisec.insert(tk.END, f"{func}(")
        elif func == "exp":
            self.fx_entry_bisec.insert(tk.END, "exp(")
        elif func == "exp10":
            self.fx_entry_bisec.insert(tk.END, "10**")
        elif func == "yroot":
            self.fx_entry_bisec.insert(tk.END, "yroot(")
        self._on_function_change_bisec()

    def _insert_power(self, power):
        """Inserta una potencia"""
        self.fx_entry_bisec.insert(tk.END, f"**{power}")
        self._on_function_change_bisec()

    def _insert_constant(self, const="pi"):
        """Inserta una constante"""
        if const == "pi":
            self.fx_entry_bisec.insert(tk.END, "pi")
        elif const == "e":
            self.fx_entry_bisec.insert(tk.END, "e")
        elif const == "inf":
            self.fx_entry_bisec.insert(tk.END, "inf")
        self._on_function_change_bisec()

    def _insert_variable(self, var):
        """Inserta una variable"""
        self.fx_entry_bisec.insert(tk.END, var)
        self._on_function_change_bisec()

    def _insert_decimal(self):
        """Inserta punto decimal"""
        self.fx_entry_bisec.insert(tk.END, ".")
        self._on_function_change_bisec()

    def _insert_combination(self):
        """Inserta combinación nCr"""
        self.fx_entry_bisec.insert(tk.END, "nCr(")
        self._on_function_change_bisec()

    def _insert_permutation(self):
        """Inserta permutación nPr"""
        self.fx_entry_bisec.insert(tk.END, "nPr(")
        self._on_function_change_bisec()

    def _insert_matrix(self):
        """Inserta matriz"""
        self.fx_entry_bisec.insert(tk.END, "matrix(")
        self._on_function_change_bisec()

    def _clear_entry(self):
        """Limpia la entrada"""
        self.fx_entry_bisec.delete(0, tk.END)
        self._on_function_change_bisec()

    def _backspace(self):
        """Elimina el último carácter"""
        current = self.fx_entry_bisec.get()
        if current:
            self.fx_entry_bisec.delete(0, tk.END)
            self.fx_entry_bisec.insert(0, current[:-1])
        self._on_function_change_bisec()

    def _toggle_second_functions(self):
        """Alterna funciones secundarias (placeholder)"""
        messagebox.showinfo("2nd", "Funciones secundarias activadas")

    def _on_function_change_bisec(self, event=None):
        """Actualiza el display LaTeX cuando cambia la función"""
        try:
            raw_text = self.fx_entry_bisec.get()
            self._actualizar_latex_display(raw_text)
        except:
            self._actualizar_latex_display("")

    def _inicializar_grafica_bisec(self):
        """Inicializa la gráfica estilo GeoGebra"""
        if hasattr(self, 'fig_bisec'):
            return

        self.fig_bisec = plt.figure(figsize=(6, 4), dpi=100)
        self.ax_bisec = self.fig_bisec.add_subplot(111)

        # Estilo GeoGebra
        self.ax_bisec.set_facecolor('#F5F5F5')
        self.fig_bisec.patch.set_facecolor('#FFFFFF')
        self.ax_bisec.grid(True, color='gray', linestyle='--', alpha=0.7)
        self.ax_bisec.axhline(y=0, color='k', linewidth=1)
        self.ax_bisec.axvline(x=0, color='k', linewidth=1)
        self.ax_bisec.set_xlabel('x', fontsize=12)
        self.ax_bisec.set_ylabel('f(x)', fontsize=12)
        self.ax_bisec.set_title('Gráfica de la función', fontsize=14, pad=20)

        # Canvas
        self.canvas_bisec = FigureCanvasTkAgg(self.fig_bisec, master=self.graph_frame_bisec)
        self.canvas_bisec.draw()
        self.canvas_bisec.get_tk_widget().pack(fill="both", expand=True)

        # Toolbar
        self.toolbar_bisec = NavigationToolbar2Tk(self.canvas_bisec, self.graph_frame_bisec)
        self.toolbar_bisec.update()
        self.toolbar_bisec.pack(side="bottom", fill="x")

    def _actualizar_grafica_bisec(self):
        """Actualiza la gráfica con la función actual"""
        try:
            func_str = self.fx_entry_bisec.get()
            f = self._parse_calculation(func_str)

            # Limpiar gráfica
            self.ax_bisec.clear()

            # Obtener intervalo
            try:
                a = float(self.a_entry_bisec.get())
                b = float(self.b_entry_bisec.get())
                if a >= b:
                    a, b = -5, 5
            except:
                a, b = -5, 5

            # Calcular puntos
            x_vals = np.linspace(a - 1, b + 1, 400)
            y_vals = [f(x) for x in x_vals]

            # Graficar
            self.ax_bisec.plot(x_vals, y_vals, 'b-', linewidth=2, label=f'f(x) = {self._convert_to_display(func_str)}')
            self.ax_bisec.axhline(y=0, color='k', linestyle='-', alpha=0.5)

            # Marcar intervalo si es válido
            try:
                fa = f(a)
                fb = f(b)
                self.ax_bisec.axvline(x=a, color='r', linestyle='--', alpha=0.7, label=f'a = {a:.2f}')
                self.ax_bisec.axvline(x=b, color='g', linestyle='--', alpha=0.7, label=f'b = {b:.2f}')
                self.ax_bisec.plot(a, fa, 'ro', markersize=6)
                self.ax_bisec.plot(b, fb, 'go', markersize=6)
            except:
                pass

            # Restaurar estilo
            self.ax_bisec.set_facecolor('#F5F5F5')
            self.ax_bisec.grid(True, color='gray', linestyle='--', alpha=0.7)
            self.ax_bisec.axhline(y=0, color='k', linewidth=1)
            self.ax_bisec.axvline(x=0, color='k', linewidth=1)
            self.ax_bisec.set_xlabel('x', fontsize=12)
            self.ax_bisec.set_ylabel('f(x)', fontsize=12)
            self.ax_bisec.set_title('Gráfica de la función', fontsize=14, pad=20)
            self.ax_bisec.legend()

            self.canvas_bisec.draw()

        except Exception as e:
            print(f"Error al graficar: {e}")

    def _zoom_grafica_bisec(self, factor):
        """Aplica zoom a la gráfica"""
        try:
            xlim = self.ax_bisec.get_xlim()
            ylim = self.ax_bisec.get_ylim()

            x_center = (xlim[0] + xlim[1]) / 2
            y_center = (ylim[0] + ylim[1]) / 2

            x_range = (xlim[1] - xlim[0]) * factor
            y_range = (ylim[1] - ylim[0]) * factor

            self.ax_bisec.set_xlim(x_center - x_range / 2, x_center + x_range / 2)
            self.ax_bisec.set_ylim(y_center - y_range / 2, y_center + y_range / 2)

            self.canvas_bisec.draw()
        except:
            pass

    def _insert_function_bisec(self, func):
        """Inserta una función predefinida"""
        current = self.fx_entry_bisec.get()
        if func == "x²":
            self.fx_entry_bisec.insert(tk.END, "x**2")
        elif func == "x³":
            self.fx_entry_bisec.insert(tk.END, "x**3")
        elif func == "e^x":
            self.fx_entry_bisec.insert(tk.END, "exp(x)")
        elif func == "ln(x)":
            self.fx_entry_bisec.insert(tk.END, "log(x)")
        elif func == "√x":
            self.fx_entry_bisec.insert(tk.END, "sqrt(x)")
        else:
            self.fx_entry_bisec.insert(tk.END, func)
        self._on_function_change_bisec()

    def _insert_operator_bisec(self, op):
        """Inserta un operador"""
        if op == "^":
            op = "**"
        elif op == "=":
            self._actualizar_grafica_bisec()
            return
        self.fx_entry_bisec.insert(tk.END, op)
        self._on_function_change_bisec()

    def _on_function_change_bisec(self, event=None):
        """Actualiza el display LaTeX cuando cambia la función"""
        try:
            raw_text = self.fx_entry_bisec.get()
            self._actualizar_latex_display(raw_text)
        except Exception as e:
            print(f"Error actualizando LaTeX: {e}")
            # Si hay error, limpia el display LaTeX
            try:
                self._actualizar_latex_display("")
            except:
                pass

    def _buscar_intervalo_valido_bisec(self):
        """Busca automáticamente un intervalo válido"""
        try:
            func_str = self.fx_entry_bisec.get()
            f = self._parse_calculation(func_str)

            from numericos import encontrar_intervalo_automatico
            a, b, mensaje = encontrar_intervalo_automatico(f)

            self.a_entry_bisec.delete(0, tk.END)
            self.a_entry_bisec.insert(0, f"{a:.4f}")
            self.b_entry_bisec.delete(0, tk.END)
            self.b_entry_bisec.insert(0, f"{b:.4f}")

            self._actualizar_grafica_bisec()
            self.biseccion_status_new.config(text="Intervalo encontrado automáticamente")

        except Exception as e:
            messagebox.showerror("Error", f"No se pudo encontrar intervalo: {str(e)}")

    def _calc_biseccion_new(self):
        """Calcula bisección con las nuevas entradas"""
        # Limpiar tabla
        for item in self.bis_tree_new.get_children():
            self.bis_tree_new.delete(item)

        try:
            func_str = self.fx_entry_bisec.get()
            f = self._parse_calculation(func_str)
            a = float(self.a_entry_bisec.get())
            b = float(self.b_entry_bisec.get())
            tol = float(self.tol_entry_bisec.get())

            if a >= b:
                raise ValueError("Debe cumplirse: a < b")

            # Verificar cambio de signo
            fa = f(a)
            fb = f(b)
            if fa * fb > 0:
                raise ValueError("La función debe tener signos opuestos en a y b (f(a)*f(b) < 0)")

            from numericos import biseccion
            raiz, pasos, motivo = biseccion(f, a, b, tol=tol, max_iter=100, usar_error="absoluto")

            # Mostrar en tabla
            self._mostrar_resultados_biseccion_new(pasos, raiz, motivo)

            # Actualizar gráfica con la raíz encontrada
            self._actualizar_grafica_con_raiz_bisec(raiz, f)

            self.biseccion_status_new.config(
                text=f"Bisección completada - {len(pasos)} iteraciones - Raíz ≈ {raiz:.8f}")

        except Exception as e:
            messagebox.showerror("Error en bisección", str(e))

    def _actualizar_grafica_con_raiz_bisec(self, raiz, f):
        """Actualiza la gráfica marcando la raíz encontrada"""
        try:
            # Limpiar y volver a graficar
            self._actualizar_grafica_bisec()
            # Marcar la raíz
            self.ax_bisec.plot(raiz, f(raiz), 'ro', markersize=10, markerfacecolor='red',
                               markeredgecolor='darkred', markeredgewidth=2,
                               label=f'Raíz ≈ {raiz:.6f}')
            self.ax_bisec.legend()
            self.canvas_bisec.draw()
        except Exception as e:
            print(f"Error al actualizar gráfica con raíz: {e}")

    def _mostrar_resultados_biseccion_new(self, pasos, raiz, motivo):
        """Muestra los resultados de la bisección en la tabla nueva"""
        for row in pasos:
            def fmt(v):
                try:
                    return f"{v:.8g}"
                except:
                    return str(v)

            # Formatear error (puede ser NaN en primera iteración)
            error = row.get("error", 0)
            error_str = "—" if (error != error) or error == float('inf') else f"{error:.6f}"

            self.bis_tree_new.insert("", "end", values=(
                row["k"],
                f"{row['a']:.6f}",
                f"{row['b']:.6f}",
                f"{row['c']:.6f}",
                f"{row['fa']:.6f}",
                f"{row['fb']:.6f}",
                f"{row['fc']:.6f}",
                error_str
            ))

        # Resaltar última iteración
        if pasos:
            last_iid = self.bis_tree_new.get_children()[-1]
            self.bis_tree_new.selection_set(last_iid)
            self.bis_tree_new.focus(last_iid)

    # Asegúrate de tener esta función de parseo en tu clase
    def _parse_calculation(self, func_str):
        """Convierte string a función ejecutable"""
        import re
        import numpy as np

        calc_str = (func_str or "").strip()
        if not calc_str:
            return lambda x: 0

        calc_str = calc_str.replace("^", "**")

        # Reemplazar multiplicaciones implícitas
        calc_str = re.sub(r'(?<=\d)(?=[a-zA-Z\(])', '*', calc_str)
        calc_str = re.sub(r'(?<=[a-zA-Z\)])(?=\d)', '*', calc_str)
        calc_str = re.sub(r'(?<=[a-zA-Z\)])(?=[a-zA-Z\(])', '*', calc_str)

        # Funciones seguras
        safe_globals = {
            "sin": np.sin, "cos": np.cos, "tan": np.tan,
            "asin": np.arcsin, "acos": np.arccos, "atan": np.arctan,
            "sinh": np.sinh, "cosh": np.cosh, "tanh": np.tanh,
            "exp": np.exp, "sqrt": np.sqrt, "abs": np.abs,
            "log": np.log, "log10": np.log10, "ln": np.log,
            "pi": np.pi, "e": np.e
        }

        try:
            code = compile(calc_str, "<string>", "eval")
        except Exception as e:
            raise ValueError(f"Expresión inválida: {e}")

        def f(x):
            try:
                return eval(code, {"__builtins__": {}}, {**safe_globals, "x": x})
            except Exception as e:
                raise ValueError(f"Error evaluando f({x}): {e}")

        return f

    def _convert_to_display(self, text):
        """Convierte a notación matemática visual"""
        if not text.strip():
            return ""

        display = text
        # Reemplazar operadores
        display = display.replace('**', '^').replace('*', '·')
        # Reemplazar funciones
        display = display.replace('math.', '')
        # Multiplicaciones implícitas
        display = re.sub(r'(\d)([a-zA-Z])', r'\1·\2', display)
        display = re.sub(r'([a-zA-Z])(\d)', r'\1·\2', display)

        return display

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
        ax.axhline(0, color="k", linestyle="-", alpha=0.7)
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
        ax.axhline(0, color='k', linestyle='-', alpha=0.7)
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
        ax.axhline(0, color="black")
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
            ax.axhline(0, color='black', linestyle='-', alpha=0.7)
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

    #----------Secante---------#
    def _tab_secante(self):
        tab = ttk.Frame(self.nb)
        self.nb.add(tab, text="Secante")

        frame = ttk.Frame(tab, padding=10)
        frame.pack(fill="both", expand=True)

        # --- Título y descripción
        title_frame = ttk.Frame(frame)
        title_frame.pack(fill="x", pady=(0, 10))
        ttk.Label(title_frame, text="Método de la Secante",
                  font=("Times New Roman", 14, "bold")).pack(anchor="w")
        ttk.Label(title_frame, text="Aproxima raíces de f(x) = 0 usando dos puntos iniciales x₀ y x₁.",
                  font=("Times New Roman", 10)).pack(anchor="w")

        # --- Entrada de datos
        input_card = ttk.LabelFrame(frame, text="Datos de entrada", padding=8)
        input_card.pack(fill="x", pady=5)

        # f(x)
        row0 = ttk.Frame(input_card)
        row0.pack(fill="x", pady=2)
        ttk.Label(row0, text="f(x) =").pack(side="left")
        self.fx_entry_sec = ttk.Entry(row0, width=40)
        self.fx_entry_sec.pack(side="left", padx=5)
        self.fx_entry_sec.bind("<KeyRelease>", self._on_function_change_sec)

        self.fx_display_sec = ttk.Label(
            input_card,
            text="f(x) = —",
            font=("Times New Roman", 9, "italic")
        )
        self.fx_display_sec.pack(anchor="w", pady=(2, 4))

        # x0, x1 y tolerancia
        row1 = ttk.Frame(input_card)
        row1.pack(fill="x", pady=2)

        ttk.Label(row1, text="x₀:").pack(side="left")
        self.x0_entry_sec = ttk.Entry(row1, width=10, justify="center")
        self.x0_entry_sec.pack(side="left", padx=4)

        ttk.Label(row1, text="x₁:").pack(side="left")
        self.x1_entry_sec = ttk.Entry(row1, width=10, justify="center")
        self.x1_entry_sec.pack(side="left", padx=4)

        ttk.Label(row1, text="Tolerancia:").pack(side="left", padx=(10, 0))
        self.tol_entry_sec = ttk.Entry(row1, width=10, justify="center")
        self.tol_entry_sec.insert(0, "1e-6")
        self.tol_entry_sec.pack(side="left", padx=4)

        # --- Botones
        btn_row = ttk.Frame(input_card)
        btn_row.pack(fill="x", pady=(8, 2))

        ttk.Button(btn_row, text="Calcular raíz",
                   style="Accent.TButton",
                   command=self._calcular_secante).pack(side="left")

        ttk.Button(btn_row, text="Graficar f(x)",
                   command=self._graficar_funcion_sec).pack(side="left", padx=5)

        ttk.Button(btn_row, text="Limpiar",
                   command=self._limpiar_secante).pack(side="left", padx=5)

        # --- Tabla de iteraciones
        table_frame = ttk.LabelFrame(frame, text="Iteraciones del Método de la Secante", padding=8)
        table_frame.pack(fill="both", expand=True, pady=10)

        columns = ("k", "x0", "x1", "x2", "fx0", "fx1", "error")
        self.sec_tree = ttk.Treeview(table_frame, columns=columns, show="headings", height=15)

        headings = ["k", "x₀", "x₁", "x₂", "f(x₀)", "f(x₁)", "Error"]
        widths = [50, 90, 90, 90, 100, 100, 90]

        for col, heading, width in zip(columns, headings, widths):
            self.sec_tree.heading(col, text=heading)
            self.sec_tree.column(col, width=width, anchor="center")

        scrollbar_table = ttk.Scrollbar(table_frame, orient="vertical", command=self.sec_tree.yview)
        self.sec_tree.configure(yscrollcommand=scrollbar_table.set)
        self.sec_tree.pack(side="left", fill="both", expand=True)
        scrollbar_table.pack(side="right", fill="y")

        # --- Estado
        status_frame = ttk.Frame(frame)
        status_frame.pack(fill="x", pady=5)

        self.secante_status = ttk.Label(
            status_frame,
            text="Listo para calcular",
            font=("Times New Roman", 9)
        )
        self.secante_status.pack(anchor="w")

    def _on_function_change_sec(self, event=None):
        """Actualiza la vista bonita de f(x) para la pestaña de Secante."""
        try:
            raw_text = self.fx_entry_sec.get()
            display_text = self._convert_to_display(raw_text)
            self.fx_display_sec.config(text=f"f(x) = {display_text}")
        except Exception:
            self.fx_display_sec.config(text="f(x) = ?")

    def _calcular_secante(self):
        """Ejecuta el método de la secante y llena la tabla."""
        # Limpiar tabla
        for it in self.sec_tree.get_children():
            self.sec_tree.delete(it)

        try:
            func_str = self.fx_entry_sec.get().strip()
            if not func_str:
                raise ValueError("Ingresa una expresión para f(x).")

            f = self._parse_calculation(func_str)

            x0 = float(self.x0_entry_sec.get())
            x1 = float(self.x1_entry_sec.get())

            tol_txt = (self.tol_entry_sec.get() or "").strip()
            tol = float(tol_txt) if tol_txt not in ("",) else 1e-6

            from numericos import secante
            raiz, pasos, motivo = secante(f, x0, x1, tol=tol, max_iter=100, usar_error="absoluto")

            # Guardar para usar en gráficas posteriores
            self.sec_last_root = raiz
            self.sec_last_func_str = func_str

            # Llenar tabla
            for p in pasos:
                self.sec_tree.insert(
                    "",
                    "end",
                    values=(
                        p["k"],
                        f"{p['x0']:.6f}",
                        f"{p['x1']:.6f}",
                        f"{p['x2']:.6f}",
                        f"{p['fx0']:.6f}",
                        f"{p['fx1']:.6f}",
                        f"{p['error']:.6f}" if p["error"] != float('inf') else "—"
                    )
                )

            # Mostrar ventana emergente con resultados
            self._mostrar_resultados_popup_secante(func_str, f, x0, x1, raiz, pasos, motivo)

            self.secante_status.config(
                text=f"Secante completado - {len(pasos)} iteraciones ({motivo})"
            )
            if hasattr(self, "lbl_result"):
                self.lbl_result.config(text=f"Raíz ≈ {raiz:.8f} (Secante: {motivo})")

        except Exception as e:
            messagebox.showerror("Error en método de la Secante", str(e))

    def _mostrar_resultados_popup_secante(self, func_str, f, x0, x1, raiz, pasos, motivo):
        """Muestra ventana emergente con gráfica y resultados detallados del método de la Secante."""
        popup = tk.Toplevel(self)
        popup.title("Resultados - Método de la Secante")
        popup.geometry("1020x600")
        popup.minsize(800, 600)
        popup.transient(self)
        popup.grab_set()
        configurar_estilo_oscuro(popup)

        # Frame principal
        main_frame = ttk.Frame(popup, padding=10)
        main_frame.pack(fill="both", expand=True)

        # Título
        title_label = ttk.Label(main_frame, text="Resultados del Método de la Secante",
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

        # Determinar rango para graficar
        all_x_values = [x0, x1, raiz]
        for paso in pasos:
            all_x_values.extend([paso.get('x0', 0), paso.get('x1', 0), paso.get('x2', 0)])

        min_x = min(all_x_values)
        max_x = max(all_x_values)
        margen = 0.2 * (max_x - min_x) if max_x != min_x else 1.0

        x_plot = np.linspace(min_x - margen, max_x + margen, 400)
        y_plot = [f(xi) for xi in x_plot]

        # Graficar función
        ax.plot(x_plot, y_plot, 'b-', linewidth=2, label=f'f(x) = {self._convert_to_display(func_str)}')
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.7)

        # Marcar puntos importantes
        ax.axvline(x=x0, color='r', linestyle='--', alpha=0.7, label=f'x₀ = {x0:.4f}')
        ax.axvline(x=x1, color='g', linestyle='--', alpha=0.7, label=f'x₁ = {x1:.4f}')
        ax.axvline(raiz, color="orange", linestyle="--", alpha=0.9,label=f"Raíz secante ≈ {raiz:.4f}")
        ax.plot(raiz, f(raiz), "o", color="orange")

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
        results_text.insert(tk.END, "RESULTADOS DEL MÉTODO DE LA SECANTE\n")
        results_text.insert(tk.END, "=" * 40 + "\n\n")

        results_text.insert(tk.END, "FUNCIÓN ANALIZADA:\n")
        results_text.insert(tk.END, f"f(x) = {self._convert_to_display(func_str)}\n\n")

        results_text.insert(tk.END, "PUNTOS INICIALES:\n")
        results_text.insert(tk.END, f"x₀ = {x0:.8f}\n")
        results_text.insert(tk.END, f"x₁ = {x1:.8f}\n\n")

        results_text.insert(tk.END, "RESULTADO FINAL:\n")
        results_text.insert(tk.END, "─" * 20 + "\n")
        results_text.insert(tk.END, f"Raíz aproximada: {raiz:.10f}\n")
        results_text.insert(tk.END, f"f(raíz) ≈ {f(raiz):.2e}\n")
        results_text.insert(tk.END, f"Iteraciones: {len(pasos)}\n")

        # Calcular error final
        if len(pasos) > 0:
            error_final = pasos[-1].get("error", 0)
            if error_final == error_final and error_final != float('inf'):  # No es NaN ni infinito
                results_text.insert(tk.END, f"Error final: {error_final:.10f}\n")
            else:
                results_text.insert(tk.END, f"Error final: —\n")
        else:
            results_text.insert(tk.END, f"Error final: —\n")

        results_text.insert(tk.END, "\nCONFIGURACIÓN:\n")
        results_text.insert(tk.END, f"Tolerancia: {float(self.tol_entry_sec.get()):.8f}\n")
        results_text.insert(tk.END, f"Criterio de parada: {motivo}\n")

        results_text.config(state="disabled")
        content_frame.add(results_frame, weight=1)

    def _graficar_funcion_sec(self):
        """Muestra una vista previa de f(x) usando x₀ y x₁ como referencia, y la raíz si ya fue calculada."""
        try:
            func_str = self.fx_entry_sec.get().strip()
            if not func_str:
                raise ValueError("Ingresa una expresión para f(x).")

            f = self._parse_calculation(func_str)

            # Intentar usar x0 y x1 para el rango
            try:
                x0 = float(self.x0_entry_sec.get())
                x1 = float(self.x1_entry_sec.get())
                a = min(x0, x1)
                b = max(x0, x1)
                if a == b:
                    a -= 1.0
                    b += 1.0
            except Exception:
                x0 = x1 = None
                a, b = -5.0, 5.0

            popup = tk.Toplevel(self)
            popup.title("Vista previa de f(x) - Secante")
            popup.geometry("900x550")
            popup.minsize(700, 450)
            popup.transient(self)
            popup.grab_set()
            configurar_estilo_oscuro(popup)

            container = ttk.Frame(popup, padding=10)
            container.pack(fill="both", expand=True)

            ttk.Label(
                container,
                text=f"f(x) = {self._convert_to_display(func_str)}",
                font=("Times New Roman", 11, "bold")
            ).pack(anchor="w", pady=(0, 8))

            fig = plt.figure(figsize=(6.5, 4), dpi=100)
            ax = fig.add_subplot(111)

            margen = 0.1 * (b - a if b != a else 1.0)
            xs = np.linspace(a - margen, b + margen, 600)
            ys = []
            for x in xs:
                try:
                    ys.append(f(x))
                except Exception:
                    ys.append(np.nan)

            # Gráfica principal
            ax.plot(xs, ys, linewidth=2, label=f"f(x) = {self._convert_to_display(func_str)}")
            ax.axhline(0, color="black", linestyle="-", alpha=0.7)

            # Marcar x0 y x1 si existen
            if x0 is not None:
                ax.axvline(x0, color="red", linestyle="--", alpha=0.8, label=f"x₀ = {x0:.4f}")
            if x1 is not None:
                ax.axvline(x1, color="green", linestyle="--", alpha=0.8, label=f"x₁ = {x1:.4f}")


            ax.set_xlabel("x")
            ax.set_ylabel("f(x)")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best")

            canvas = FigureCanvasTkAgg(fig, master=container)
            canvas.draw()

            # Toolbar interactiva (zoom, pan, guardar, etc.)
            toolbar = NavigationToolbar2Tk(canvas, container)
            toolbar.update()
            toolbar.pack(side="bottom", fill="x")

            canvas.get_tk_widget().pack(fill="both", expand=True)

        except Exception as e:
            messagebox.showerror("Error al graficar", str(e))
            
    def _limpiar_secante(self):
        """Limpia entradas y tabla de la pestaña Secante."""
        self.fx_entry_sec.delete(0, tk.END)
        self.x0_entry_sec.delete(0, tk.END)
        self.x1_entry_sec.delete(0, tk.END)
        self.tol_entry_sec.delete(0, tk.END)
        self.tol_entry_sec.insert(0, "1e-6")
        for it in self.sec_tree.get_children():
            self.sec_tree.delete(it)
        self.secante_status.config(text="Listo para calcular")

#--------------RUN----------#       
if __name__ == "__main__":
    App().mainloop()
