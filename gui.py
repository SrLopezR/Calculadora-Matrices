# gui.py
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import time
from fraccion import Fraccion
from gauss import GaussJordanEngine
from matrices import sumar_matrices, multiplicar_matrices, multiplicar_escalar_matriz, formatear_matriz, Transpuesta

# ====================== WIDGETS DE MATRIZ ======================
class MatrixInput(ttk.Frame):
    def __init__(self, master, rows=3, cols=3, allow_b=True, **kw):
        super().__init__(master, **kw)
        self.rows, self.cols = rows, cols
        self.allow_b = allow_b  # True si hay columna b (Gauss)
        self.entries=[]
        self._build()

    def _build(self):
        # Encabezados
        for j in range(self.cols):
            ttk.Label(self, text=f"x{j+1}", anchor="center").grid(row=0,column=j,padx=4,pady=4)
        if self.allow_b:
            ttk.Label(self, text="|", width=2).grid(row=0, column=self.cols)
            ttk.Label(self, text="b", anchor="center").grid(row=0,column=self.cols+1,padx=4,pady=4)

        for i in range(self.rows):
            fila=[]
            for j in range(self.cols + (1 if self.allow_b else 0)):
                e=ttk.Entry(self, width=8, justify="center")
                col=j if j<self.cols else j+1
                e.grid(row=i+1, column=col, padx=2,pady=2)
                fila.append(e)
            self.entries.append(fila)
            if self.allow_b:
                ttk.Label(self, text="|", width=2).grid(row=i+1, column=self.cols)

    def get_matrix(self):
        M=[]
        for i in range(self.rows):
            fila=[]
            for j in range(len(self.entries[i])):
                t=self.entries[i][j].get().strip()
                if not t: t='0'
                try:
                    fila.append(Fraccion(t))
                except Exception:
                    raise ValueError(f"Valor inválido en fila {i+1}, columna {j+1}: '{t}'")
            M.append(fila)
        return M

    def set_size(self, rows, cols):
        for child in list(self.winfo_children()): child.destroy()
        self.rows, self.cols = rows, cols
        self.entries=[]
        self._build()

    def clear(self):
        for fila in self.entries:
            for e in fila:
                e.delete(0, tk.END)

class MatrixView(ttk.Frame):
    def __init__(self, master, **kw):
        super().__init__(master, **kw)
        self.tree = ttk.Treeview(self, show='headings', height=8)
        self.tree.pack(fill="both", expand=True)
        self._cols=0
        style=ttk.Style(self)
        style.configure("Pivot.TREE", background="#FFF1C1")

    def set_matrix(self,M):
        if not M: return
        rows=len(M)
        cols=len(M[0])
        if cols!=self._cols:
            self.tree.delete(*self.tree.get_children())
            self.tree["columns"]=[f"c{j}" for j in range(cols)]
            for j in range(cols):
                txt=f"x{j+1}" if j<cols-1 else "b"
                self.tree.heading(f"c{j}", text=txt)
                self.tree.column(f"c{j}", width=60, anchor="center")
            self._cols=cols
        self.tree.delete(*self.tree.get_children())
        for i in range(rows):
            vals=[str(x) for x in M[i]]
            self.tree.insert('', 'end', values=vals)

    def highlight(self,row=None, col=None):
        for iid in self.tree.get_children(): self.tree.item(iid,tags=())
        if row is not None:
            try:
                iid=self.tree.get_children()[row]
                self.tree.item(iid, tags=("pivot",))
                self.tree.tag_configure("pivot", background="#FFF1C1")
            except IndexError: pass

# ====================== APP PRINCIPAL ======================
class MainApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Soluciones Matrices")
        self.geometry("980x680")
        self.engine=None
        self.auto_running=False
        self.auto_thread=None
        self._build_ui()

    def _build_ui(self):
        top=ttk.Frame(self); top.pack(fill="x", padx=10,pady=8)
        ttk.Label(top,text="Ecuaciones (m):").pack(side="left")
        self.spin_m=tk.Spinbox(top,from_=1,to=12,width=5); self.spin_m.delete(0,tk.END); self.spin_m.insert(0,"3"); self.spin_m.pack(side="left", padx=(4,12))
        ttk.Label(top,text="Incógnitas (n):").pack(side="left")
        self.spin_n=tk.Spinbox(top,from_=1,to=12,width=5); self.spin_n.delete(0,tk.END); self.spin_n.insert(0,"3"); self.spin_n.pack(side="left", padx=(4,12))
        ttk.Button(top,text="Redimensionar",command=self.resize_matrix).pack(side="left")
        ttk.Button(top,text="Ejemplo",command=self.load_example).pack(side="left", padx=5)
        ttk.Button(top,text="Limpiar",command=self.clear_inputs).pack(side="left")
        ttk.Button(top,text="Sumar matrices", command=self.sumar_gui).pack(side="left", padx=5)
        ttk.Button(top,text="Multiplicar matrices", command=self.multiplicar_gui).pack(side="left", padx=5)
        ttk.Button(top, text="Escalar × matriz", command=self.abrir_escalar).pack(side="left", padx=6)
        ttk.Button(top, text="Transpuesta", command=self.Transouesta_gui).pack(side="left", padx=7)

        self.input_frame=ttk.LabelFrame(self,text="Matriz aumentada (coeficientes | términos independientes)")
        self.input_frame.pack(fill="x", padx=10,pady=6)
        self.matrix_input=MatrixInput(self.input_frame, rows=3, cols=3)
        self.matrix_input.pack(fill="x", padx=8,pady=8)

        actions=ttk.Frame(self); actions.pack(fill="x", padx=10,pady=6)
        ttk.Button(actions,text="Resolver (inicializar)",command=self.start_engine).pack(side="left")
        ttk.Button(actions,text="Siguiente paso",command=self.next_step).pack(side="left", padx=6)
        self.btn_auto=ttk.Button(actions,text="Reproducir",command=self.toggle_auto); self.btn_auto.pack(side="left")
        ttk.Button(actions,text="Reiniciar",command=self.reset).pack(side="left", padx=6)
        ttk.Button(actions,text="Exportar pasos",command=self.export_log).pack(side="left")

        body=ttk.Frame(self); body.pack(fill="both", expand=True, padx=10,pady=4)
        left=ttk.Frame(body); left.pack(side="left", fill="both", expand=True)
        right=ttk.Frame(body); right.pack(side="left", fill="both", expand=True, padx=(8,0))

        ttk.Label(left,text="Matriz y pivotes").pack(anchor="w")
        self.matrix_view=MatrixView(left); self.matrix_view.pack(fill="both", expand=True)
        ttk.Label(right,text="Pasos / Operaciones").pack(anchor="w")
        self.txt_log=tk.Text(right,height=12,wrap="word"); self.txt_log.pack(fill="both", expand=True)

        self.result_frame=ttk.LabelFrame(self,text="Resultado"); self.result_frame.pack(fill="x", padx=10,pady=10)
        self.lbl_result=ttk.Label(self.result_frame,text="—"); self.lbl_result.pack(anchor="w", padx=8,pady=8)

        self.status=ttk.Label(self, relief="sunken", anchor="w"); self.status.pack(fill="x", side="bottom")

    # ============= Helpers y Gauss =============
    def resize_matrix(self):
        try:
            m=int(self.spin_m.get()); n=int(self.spin_n.get())
            if m<=0 or n<=0: raise ValueError
        except: messagebox.showerror("Error","Dimensiones inválidas"); return
        self.matrix_input.set_size(m,n)

    def load_example(self):
        ejemplo=[["1","1","1","6"], ["2","-1","1","3"], ["1","2","-1","3"]]
        self.matrix_input.set_size(3,3)
        for i in range(3):
            for j in range(4):
                self.matrix_input.entries[i][j].delete(0,tk.END)
                self.matrix_input.entries[i][j].insert(0, ejemplo[i][j])

    def clear_inputs(self):
        self.matrix_input.clear()
        self.txt_log.delete(1.0, tk.END)
        self.lbl_result.config(text="—")

    def start_engine(self):
        try: A=self.matrix_input.get_matrix()
        except Exception as e: messagebox.showerror("Entrada inválida",str(e)); return
        self.engine=GaussJordanEngine(A)
        self._render_last_step()
        self._log("Inicializado. Use 'Siguiente paso' o 'Reproducir'.")
        self._update_status("Listo para ejecutar.")

    def next_step(self):
        if not self.engine: messagebox.showinfo("Información","Primero presione 'Resolver (inicializar)'"); return
        step=self.engine.siguiente()
        if step is None: self._log("No hay más pasos.")
        self._render_last_step()
        if self.engine.terminado: self._show_result()

    def toggle_auto(self):
        if not self.engine: messagebox.showinfo("Información","Primero presione 'Resolver (inicializar)'"); return
        if not self.auto_running:
            self.auto_running=True
            self.btn_auto.config(text="Pausar")
            self.auto_thread=threading.Thread(target=self._auto_run, daemon=True)
            self.auto_thread.start()
        else:
            self.auto_running=False
            self.btn_auto.config(text="Reproducir")

    def _auto_run(self):
        while self.auto_running and self.engine and not self.engine.terminado:
            self.next_step()
            time.sleep(1.2)
        self.auto_running=False
        self.btn_auto.config(text="Reproducir")

    def reset(self):
        self.engine=None
        self.txt_log.delete(1.0, tk.END)
        self.matrix_view.set_matrix([[Fraccion(0)]])
        self.lbl_result.config(text="—")
        self._update_status("Reiniciado.")

    def export_log(self):
        if not self.engine or not self.engine.log: messagebox.showinfo("Información","No hay pasos para exportar"); return
        fp=filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Texto","*.txt")], title="Guardar registro de pasos")
        if not fp: return
        with open(fp,'w',encoding='utf-8') as f:
            for i,s in enumerate(self.engine.log,start=1):
                f.write(f"Paso {i}: {s.descripcion}\n")
                for fila in s.matriz:
                    f.write(" [ "+ " ".join(str(x) for x in fila[:-1]) + " | " + str(fila[-1]) + " ]\n")
                f.write("\n")
        messagebox.showinfo("Listo", f"Registro exportado a: {fp}")

    def _render_last_step(self):
        if not self.engine or not self.engine.log: return
        step=self.engine.log[-1]
        self.matrix_view.set_matrix(step.matriz)
        self.matrix_view.highlight(step.pivote_row, step.pivote_col)
        self._log(step.descripcion)

    def _log(self,text): self.txt_log.insert(tk.END, text+"\n"); self.txt_log.see(tk.END)
    def _show_result(self):
        tipo,data=self.engine.analizar()
        if tipo=="inconsistente": msg="El sistema es INCONSISTENTE: no tiene solución."
        elif tipo=="única": msg="Solución ÚNICA:\n " + "\n ".join(f"x{i+1}={v}" for i,v in enumerate(data))
        else:
            libres, base = data
            txt = ["Infinitas soluciones:"]
            if libres:
                txt.append(" Variables libres: " + ", ".join(f"x{j + 1}" for j in libres))
            # Show dependent variables in terms of free variables
            n = len(base[0]) if base else 0
            for i in range(n):
                if i not in libres:
                    expr = []
                    for idx, j in enumerate(libres):
                        coef = base[idx][i]
                        if coef != 0:
                            term = f"{coef}·x{j + 1}" if coef != 1 else f"x{j + 1}"
                            expr.append(term)
                    if expr:
                        txt.append(f" x{i + 1} = " + " + ".join(expr))
            txt.append(" Base del espacio nulo:")
            for k, vec in enumerate(base, start=1):
                txt.append(" v" + str(k) + " = (" + ", ".join(str(x) for x in vec) + ")")
            msg = "\n".join(txt)
        self.lbl_result.config(text=msg)
        self._update_status("Cálculo finalizado.")

    def _update_status(self,s): self.status.config(text=s)

    # ================== SUMA / MULTIPLICACION ==================
    def abrir_escalar(self):
        EscalarStepApp()
    def sumar_gui(self):
        self.matrices_gui("Sumar matrices", sumar_matrices, suma=True)
    def Transouesta_gui(self):
        self.matriz_gui("Transpuesta de matrices", Transpuesta,)


    def multiplicar_gui(self):
        self.matrices_gui("Multiplicar matrices", multiplicar_matrices, suma=False)

    def matriz_gui(self, title, operation):
        win=tk.Toplevel(self)
        win.title(title)
        win.geometry("700x550")
        # Selección de tamaño
        size_frame=ttk.Frame(win); size_frame.pack(fill="x", padx=8, pady=4)
        ttk.Label(size_frame, text="Filas:").pack(side="left")
        spin_rows=tk.Spinbox(size_frame, from_=1,to=8,width=5); spin_rows.delete(0,"end"); spin_rows.insert(0,"2"); spin_rows.pack(side="left")
        ttk.Label(size_frame, text="Columnas:").pack(side="left")
        spin_cols=tk.Spinbox(size_frame, from_=1,to=8,width=5); spin_cols.delete(0,"end"); spin_cols.insert(0,"2"); spin_cols.pack(side="left")


    def matrices_gui(self, title, operation, suma=False):
        win=tk.Toplevel(self)
        win.title(title)
        win.geometry("700x550")

        # Selección de tamaños
        size_frame=ttk.Frame(win); size_frame.pack(fill="x", padx=8, pady=4)
        ttk.Label(size_frame, text="Filas A:").pack(side="left")
        spin_A_rows=tk.Spinbox(size_frame, from_=1,to=8,width=5); spin_A_rows.delete(0,"end"); spin_A_rows.insert(0,"2"); spin_A_rows.pack(side="left")
        ttk.Label(size_frame, text="Columnas A:").pack(side="left")
        spin_A_cols=tk.Spinbox(size_frame, from_=1,to=8,width=5); spin_A_cols.delete(0,"end"); spin_A_cols.insert(0,"2"); spin_A_cols.pack(side="left")

        ttk.Label(size_frame, text="Filas B:").pack(side="left", padx=(10,0))
        spin_B_rows=tk.Spinbox(size_frame, from_=1,to=8,width=5); spin_B_rows.delete(0,"end"); spin_B_rows.insert(0,"2"); spin_B_rows.pack(side="left")
        ttk.Label(size_frame, text="Columnas B:").pack(side="left")
        spin_B_cols=tk.Spinbox(size_frame, from_=1,to=8,width=5); spin_B_cols.delete(0,"end"); spin_B_cols.insert(0,"2"); spin_B_cols.pack(side="left")

        def resize_inputs():
            try:
                ra,ca=int(spin_A_rows.get()), int(spin_A_cols.get())
                rb,cb=int(spin_B_rows.get()), int(spin_B_cols.get())
            except:
                messagebox.showerror("Error","Dimensiones inválidas"); return

            if suma and (ra!=rb or ca!=cb):
                messagebox.showerror("Error","Para la suma, ambas matrices deben tener la misma dimensión.")
                return
            if not suma and ca!=rb:
                messagebox.showerror("Error","Para la multiplicación: columnas de A deben coincidir con filas de B.")
                return

            inputA.set_size(ra,ca)
            inputB.set_size(rb,cb)

        ttk.Button(size_frame, text="Redimensionar", command=resize_inputs).pack(side="left", padx=10)

        # Entradas
        ttk.Label(win,text="Matriz A").pack(anchor="w")
        inputA=MatrixInput(win, rows=2, cols=2, allow_b=False)
        inputA.pack(fill="x", padx=8,pady=6)

        ttk.Label(win,text="Matriz B").pack(anchor="w")
        inputB=MatrixInput(win, rows=2, cols=2, allow_b=False)
        inputB.pack(fill="x", padx=8,pady=6)

        txt_log=tk.Text(win,height=12, wrap="word"); txt_log.pack(fill="both", expand=True, padx=8,pady=6)
        lbl_result=ttk.Label(win, text="—"); lbl_result.pack(anchor="w", padx=8,pady=6)

        def calcular():
            try:
                A=inputA.get_matrix(); B=inputB.get_matrix()
                R,pasos=operation(A,B)
            except Exception as e:
                messagebox.showerror("Error",str(e))
                return
            lbl_result.config(text=formatear_matriz(R))
            txt_log.delete(1.0, tk.END)
            for p in pasos: txt_log.insert(tk.END,p+"\n")

        ttk.Button(win,text="Calcular", command=calcular).pack(side="left", padx=10,pady=10)
        ttk.Button(win,text="Limpiar", command=lambda:[inputA.clear(), inputB.clear(), txt_log.delete(1.0,tk.END), lbl_result.config(text="—")]).pack(pady=6)
# ================== Multiplicacion de escalar ==========================

class EscalarStepApp(tk.Toplevel):
    """Ventana para multiplicar escalar por matriz paso a paso"""
    def __init__(self):
        super().__init__()
        self.title("Multiplicar escalar por matriz")
        self.geometry("900x600")
        self.pasos = []
        self.resultado = []
        self.step_idx = 0
        self.build_ui()

    def build_ui(self):
        top = ttk.Frame(self)
        top.pack(fill="x", padx=10, pady=6)
        ttk.Label(top, text="Filas:").pack(side="left")
        self.spin_r = tk.Spinbox(top, from_=1, to=12, width=5)
        self.spin_r.pack(side="left", padx=4)
        ttk.Label(top, text="Columnas:").pack(side="left")
        self.spin_c = tk.Spinbox(top, from_=1, to=12, width=5)
        self.spin_c.pack(side="left", padx=4)
        ttk.Label(top, text="Escalar:").pack(side="left")
        self.entry_escalar = ttk.Entry(top, width=10)
        self.entry_escalar.pack(side="left", padx=4)
        ttk.Button(top, text="Redimensionar", command=self.redim).pack(side="left", padx=6)

        self.frame_matrices = ttk.Frame(self)
        self.frame_matrices.pack(fill="both", expand=True)

        self.matrixA = MatrixInput(self.frame_matrices, rows=2, cols=2, allow_b=False)
        self.matrixA.grid(row=0, column=0, padx=8, pady=8)

        actions = ttk.Frame(self)
        actions.pack(fill="x", padx=10, pady=6)
        ttk.Button(actions, text="Calcular", command=self.calcular).pack(side="left", padx=4)
        self.btn_step = ttk.Button(actions, text="Siguiente paso", command=self.siguiente_paso, state="disabled")
        self.btn_step.pack(side="left", padx=4)
        ttk.Button(actions, text="Exportar log", command=self.export_log).pack(side="left", padx=4)

        self.txt_log = tk.Text(self, height=20)
        self.txt_log.pack(fill="both", expand=True, padx=10, pady=8)

    def redim(self):
        r = int(self.spin_r.get())
        c = int(self.spin_c.get())
        self.matrixA.set_size(r,c)

    def calcular(self):
        try:
            A = self.matrixA.get_matrix()
            escalar = self.entry_escalar.get()
            R, pasos = multiplicar_escalar_matriz(escalar, A)
        except Exception as e:
            messagebox.showerror("Error", str(e))
            return
        self.resultado = R
        self.pasos = pasos
        self.step_idx = 0
        self.txt_log.delete(1.0, tk.END)
        self.txt_log.insert(tk.END, "Cálculo inicializado. Presione 'Siguiente paso'.\n")
        self.btn_step.config(state="normal")

    def siguiente_paso(self):
        if self.step_idx < len(self.pasos):
            self.txt_log.insert(tk.END, self.pasos[self.step_idx]+"\n\n")
            self.step_idx += 1
            if self.step_idx == len(self.pasos):
                self.btn_step.config(state="disabled")
        else:
            self.btn_step.config(state="disabled")

    def export_log(self):
        if not self.pasos:
            messagebox.showinfo("Info","No hay pasos para exportar.")
            return
        fp = filedialog.asksaveasfilename(defaultextension=".txt",
                                          filetypes=[("Texto","*.txt")],
                                          title="Guardar registro de pasos")
        if not fp:
            return
        with open(fp,"w",encoding="utf-8") as f:
            for i,p in enumerate(self.pasos,start=1):
                f.write(f"Paso {i}: {p}\n\n")
            f.write("Resultado final:\n")
            for fila in self.resultado:
                f.write(" ".join(str(x) for x in fila)+"\n")
        messagebox.showinfo("Listo", f"Registro exportado a: {fp}")


if __name__=="__main__":
    app=MainApp()
    app.mainloop()
