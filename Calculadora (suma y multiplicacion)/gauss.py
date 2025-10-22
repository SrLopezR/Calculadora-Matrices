from fraccion import Fraccion
from dataclasses import dataclass

# ------------------------------------------------------------
# Clase Step: representa un "paso" de operación elemental
# ------------------------------------------------------------
@dataclass
class Step:
    descripcion: str
    matriz: list
    pivote_row: int | None
    pivote_col: int | None


# ------------------------------------------------------------
# Clase principal del método de Gauss-Jordan
# ------------------------------------------------------------
class GaussJordanEngine:
    def __init__(self, aug_matrix):
        self.A = [fila[:] for fila in aug_matrix]   # Copia profunda de la matriz aumentada
        self.m = len(self.A)                        # Número de filas (ecuaciones)
        self.n = len(self.A[0]) - 1                 # Número de variables (columnas sin el término independiente)
        self.fila = 0                               # Fila actual en proceso
        self.col_pivotes = []                       # Columnas donde hay pivote (indican independencia lineal)
        self.terminado = False
        self.log = []                               # Historial de pasos realizados
        self._snapshot("Inicio", None, None)        # Guarda el estado inicial de la matriz

    def _copy(self):
        # Crea una copia de la matriz actual
        return [[Fraccion(x.num, x.den) for x in fila] for fila in self.A]

    def _snapshot(self, descripcion, prow, pcol):
        # Registra un paso del método, con descripción y pivote usado
        self.log.append(Step(descripcion, self._copy(), prow, pcol))

    # --------------------------------------------------------
    # Realiza el siguiente paso del método de Gauss-Jordan
    # --------------------------------------------------------
    def siguiente(self):
        if self.terminado:
            return None

        while self.col_actual() < self.n and self.fila < self.m:
            c = self.col_actual()   # Columna actual (variable en análisis)
            sel = None

            # Búsqueda de fila con pivote ≠ 0 → determina independencia de esa columna
            for r in range(self.fila, self.m):
                if not self.A[r][c].es_cero():
                    sel = r
                    break

            # Si no se encuentra pivote, la columna es combinación lineal de anteriores (dependiente)
            if sel is None:
                self._snapshot(f"No hay pivote en la columna {c+1}.", None, c)
                self.col_pivotes.append(None)
                self.fila += 1
                return self.log[-1]

            # Si el pivote no está en la fila actual, se intercambian filas
            if sel != self.fila:
                self.A[self.fila], self.A[sel] = self.A[sel], self.A[self.fila]
                self._snapshot(f"Intercambiar F{self.fila+1} ↔ F{sel+1}", self.fila, c)
                return self.log[-1]

            # Normaliza el pivote (hace que valga 1)
            piv = self.A[self.fila][c]
            if not piv.es_uno():
                inv = Fraccion(piv.den, piv.num)
                for j in range(c, self.n+1):
                    self.A[self.fila][j] = self.A[self.fila][j] * inv
                self._snapshot(f"Multiplicar F{self.fila+1} por 1/({piv}) para pivote = 1", self.fila, c)
                return self.log[-1]

            # Elimina los valores en la misma columna (hace ceros arriba y abajo del pivote)
            for r in range(self.m):
                if r == self.fila:
                    continue
                factor = self.A[r][c]
                if factor.es_cero():
                    continue
                for j in range(c, self.n+1):
                    self.A[r][j] = self.A[r][j] - factor * self.A[self.fila][j]
                self._snapshot(f"F{r+1} = F{r+1} - ({factor})·F{self.fila+1}", self.fila, c)
                return self.log[-1]

            # Registra la columna como pivote → variable independiente (vector base)
            self.col_pivotes.append(c)
            self.fila += 1
            return self.log[-1]

        # Si ya no hay más columnas por procesar, el método terminó
        self._snapshot("Fin Gauss-Jordan", None, None)
        self.terminado = True
        return self.log[-1]

    def col_actual(self):
        return len(self.col_pivotes)

    # --------------------------------------------------------
    # Analiza el tipo de sistema: único, inconsistente o infinito
    # --------------------------------------------------------
    def analizar(self):
        """
        Analiza el sistema y devuelve:
        - tipo: inconsistente | única | infinitas
        - solución o información adicional
        - clasificación: homogéneo / no homogéneo, trivial / no trivial
        """
        A = self.A
        m, n = self.m, self.n

        # Determina si es homogéneo (columna independiente = 0)
        es_homogeneo = all(A[i][n].es_cero() for i in range(m))

        # Comprueba inconsistencia (fila de ceros = cte ≠ 0)
        for i in range(m):
            if all(A[i][j].es_cero() for j in range(n)) and not A[i][n].es_cero():
                return ("inconsistente", None,
                        {"tipo": "no homogéneo", "solucion": "ninguna"})

        # Caso de solución única: todas las columnas son linealmente independientes
        if len([x for x in self.col_pivotes if x is not None]) == n:
            sol = [A[i][n] for i in range(n)]
            clasificacion = {
                "tipo": "homogéneo" if es_homogeneo else "no homogéneo",
                "solucion": "trivial" if es_homogeneo and all(s.es_cero() for s in sol) else "única"
            }
            return ("única", sol, clasificacion)

        # Caso de infinitas soluciones → hay variables libres (dependencia lineal)
        libres = [j for j in range(n) if j not in self.col_pivotes]
        particular = [Fraccion(0) for _ in range(n)]

        # Construye el vector solución particular
        for i, pc in enumerate(self.col_pivotes):
            if pc is not None:
                particular[pc] = A[i][n]

        # --------------------------------------------------------
        # Construcción de la BASE del espacio nulo (N(A)):
        # Cada vector 'vec' es un vector linealmente independiente
        # que forma parte de la base del subespacio solución homogéneo.
        # --------------------------------------------------------
        base = []
        for f in libres:
            vec = [Fraccion(0) for _ in range(n)]
            vec[f] = Fraccion(1)  # Se asigna 1 a la variable libre → genera un vector base
            for i, pc in enumerate(self.col_pivotes):
                if pc is not None:
                    # Expresa las variables dependientes como combinación lineal de las libres
                    vec[pc] = -A[i][f]
            base.append(vec)  # Cada 'vec' es un vector independiente de los demás

        clasificacion = {
            "tipo": "homogéneo" if es_homogeneo else "no homogéneo",
            "solucion": "no trivial"
        }
        return ("infinitas", (particular, libres, base), clasificacion)

    # --------------------------------------------------------
    # Muestra el conjunto solución en forma vectorial / paramétrica
    # --------------------------------------------------------
    def conjunto_solucion(self, resultado):
        """
        Devuelve una representación del conjunto solución en forma paramétrica.
        Reemplaza t1, t2... por x1, x2, x3...
        """
        tipo, data, clasificacion = resultado

        # Sistema sin solución (espacio vacío)
        if tipo == "inconsistente":
            return "El sistema es inconsistente. No tiene solución."

        # Sistema con solución única (punto en el espacio)
        if tipo == "única":
            sol = data
            sol_text = ", ".join(str(s) for s in sol)
            return f"Solución única:\n(x1, x2, ..., xn) = ({sol_text})\nClasificación: {clasificacion}"

        # Sistema con infinitas soluciones → subespacio afín
        if tipo == "infinitas":
            particular, libres, base = data
            out = "Solución general:\n"
            part_str = "(" + ", ".join(str(x) for x in particular) + ")"
            out += f"x = {part_str}"

            # Cada término xj*(vector) representa una combinación lineal
            # de vectores base linealmente independientes.
            for j, v in enumerate(base):
                param = f"x{libres[j]+1}"  # Parámetro libre asociado al vector base
                vec_str = "(" + ", ".join(str(x) for x in v) + ")"
                out += f" + {param}*{vec_str}"

            out += f"\nClasificación: {clasificacion}"
            return out
