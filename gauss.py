from fraccion import Fraccion 


class PasoGauss:
    def __init__(self, matriz, descripcion, pivote_row=None, pivote_col=None):
        self.matriz = [[x for x in fila] for fila in matriz]  # Copia profunda
        self.descripcion = descripcion
        self.pivote_row = pivote_row
        self.pivote_col = pivote_col


class GaussJordanEngine:
    def __init__(self, matriz_aumentada):
        self.matriz_original = [[x for x in fila] for fila in matriz_aumentada]
        self.matriz_actual = [[x for x in fila] for fila in matriz_aumentada]
        self.log = []
        self.paso_actual = 0
        self.filas = len(matriz_aumentada)
        self.columnas = len(matriz_aumentada[0]) if matriz_aumentada else 0
        self.terminado = False
        self.fila_actual = 0
        self.col_actual = 0

        # Registrar estado inicial
        self._agregar_paso("Estado inicial")

    def _agregar_paso(self, descripcion, pivote_row=None, pivote_col=None):
        paso = PasoGauss(self.matriz_actual, descripcion, pivote_row, pivote_col)
        self.log.append(paso)
        self.paso_actual += 1

    def _intercambiar_filas(self, i, j):
        if i != j:
            self.matriz_actual[i], self.matriz_actual[j] = self.matriz_actual[j], self.matriz_actual[i]
            self._agregar_paso(f"Intercambiar fila {i + 1} con fila {j + 1}", i, self.col_actual)

    def _multiplicar_fila(self, fila, escalar):
        for j in range(self.columnas):
            self.matriz_actual[fila][j] = self.matriz_actual[fila][j] * escalar
        self._agregar_paso(f"Multiplicar fila {fila + 1} por {escalar}", fila, self.col_actual)

    def _sumar_filas(self, fila_destino, fila_fuente, escalar):
        for j in range(self.columnas):
            self.matriz_actual[fila_destino][j] = (
                self.matriz_actual[fila_destino][j]
                + self.matriz_actual[fila_fuente][j] * escalar
            )
        desc = f"F{fila_destino + 1} → F{fila_destino + 1} + ({escalar})×F{fila_fuente + 1}"
        self._agregar_paso(desc, fila_destino, self.col_actual)

    def _encontrar_pivote(self, fila_inicio, col):
        for i in range(fila_inicio, self.filas):
            if not self.matriz_actual[i][col].es_cero():
                return i
        return None

    def siguiente(self):
        if self.terminado:
            return None

        # Fase de eliminación hacia adelante + atrás (Gauss-Jordan)
        while self.fila_actual < self.filas and self.col_actual < self.columnas - 1:
            # Buscar pivote en columna actual
            fila_pivote = self._encontrar_pivote(self.fila_actual, self.col_actual)

            if fila_pivote is None:
                # No hay pivote en esta columna, pasar a siguiente columna
                self.col_actual += 1
                continue

            # Mover fila con pivote a posición actual si es necesario
            if fila_pivote != self.fila_actual:
                self._intercambiar_filas(self.fila_actual, fila_pivote)
                return self.log[-1]

            # Normalizar fila pivote
            pivote = self.matriz_actual[self.fila_actual][self.col_actual]
            if pivote != Fraccion(1):
                self._multiplicar_fila(self.fila_actual, pivote.reciproco())
                return self.log[-1]

            # Eliminar en otras filas (arriba y abajo)
            for i in range(self.filas):
                if i != self.fila_actual and not self.matriz_actual[i][self.col_actual].es_cero():
                    factor = -self.matriz_actual[i][self.col_actual]
                    self._sumar_filas(i, self.fila_actual, factor)
                    return self.log[-1]

            self.fila_actual += 1
            self.col_actual += 1

        self.terminado = True
        self._agregar_paso("Proceso completado")
        return self.log[-1]

    # ---------- análisis detallado y forma paramétrica ----------

    def _info_pivotes_y_vars(self):
        """Obtiene columnas pivote, variables básicas y libres a partir de la matriz reducida."""
        n_vars = self.columnas - 1
        matriz = self.matriz_actual

        pivot_cols = []
        for i in range(self.filas):
            for j in range(n_vars):
                if not matriz[i][j].es_cero():
                    pivot_cols.append(j)
                    break

        pivot_cols = sorted(set(pivot_cols))
        basic_vars = [f"x{j + 1}" for j in pivot_cols]
        free_cols = [j for j in range(n_vars) if j not in pivot_cols]
        free_vars = [f"x{j + 1}" for j in free_cols]

        return pivot_cols, basic_vars, free_cols, free_vars

    def analizar(self):
        """Analiza el sistema y devuelve información detallada sobre la solución."""
        # Asegurarse de que el proceso Gauss-Jordan terminó
        if not self.terminado:
            while not self.terminado:
                self.siguiente()

        matriz = self.matriz_actual
        filas = self.filas
        n_vars = self.columnas - 1  # sin la columna de términos independientes

        # Cálculo de rangos y detección de inconsistencia
        rank_A = 0
        rank_AB = 0
        inconsistente = False

        for i in range(filas):
            hay_no_cero_A = any(not matriz[i][j].es_cero() for j in range(n_vars))
            hay_no_cero_AB = hay_no_cero_A or (not matriz[i][n_vars].es_cero())
            if hay_no_cero_A:
                rank_A += 1
            if hay_no_cero_AB:
                rank_AB += 1

            # Fila del tipo 0 ... 0 | b (b != 0)  -> sistema inconsistente
            if (not hay_no_cero_A) and (not matriz[i][n_vars].es_cero()):
                inconsistente = True

        pivot_cols, basic_vars, free_cols, free_vars = self._info_pivotes_y_vars()

        info_basica = {
            "rank_A": rank_A,
            "rank_AB": rank_AB,
            "pivot_cols": pivot_cols,      # índices 0-based
            "basic_vars": basic_vars,      # nombres 'x1', 'x3', ...
            "free_vars": free_vars,        # nombres 'x2', 'x4', ...
        }

        detalles = {
            "ecuaciones_parametricas": [],
        }

        # Caso inconsistente
        if inconsistente or rank_A != rank_AB:
            tipo = "inconsistente"
            return tipo, info_basica, detalles

        # Sistema consistente
        if rank_A == n_vars:
            tipo = "única"
        else:
            tipo = "infinitas"

        # Construcción de las ecuaciones paramétricas a partir de la RREF
        ecuaciones = []
        for i in range(filas):
            # Localizar pivote en esta fila
            pivote_col = None
            for j in range(n_vars):
                if not matriz[i][j].es_cero():
                    pivote_col = j
                    break
            if pivote_col is None:
                continue  # fila completamente cero, no genera ecuación

            # x_pivote = b_i - sum(a_ij * x_j_libre)
            const_term = matriz[i][n_vars]
            terms = []
            for j in free_cols:
                coef = matriz[i][j]
                if coef.es_cero():
                    continue
                coef_rhs = -coef  # pasa restando
                var_name = f"x{j + 1}"

                coef_str = str(coef_rhs)
                # Manejo de signos bonito
                if coef_str == "1":
                    terms.append(f"+ {var_name}")
                elif coef_str == "-1":
                    terms.append(f"- {var_name}")
                else:
                    signo = "-" if coef_str.startswith("-") else "+"
                    if coef_str[0] in "+-":
                        coef_str = coef_str[1:]
                    terms.append(f"{signo} {coef_str}·{var_name}")

            rhs = str(const_term)
            for t in terms:
                rhs += " " + t

            ecuaciones.append(f"x{pivote_col + 1} = {rhs}")

        # También indicar expresamente qué variables son libres
        for name in free_vars:
            ecuaciones.append(f"{name} es libre")

        detalles["ecuaciones_parametricas"] = ecuaciones

        return tipo, info_basica, detalles

    def conjunto_solucion(self, resultado=None):
        """
        Devuelve un texto amigable para mostrar en la barra de resultados.
        Si no se le pasa 'resultado', llama internamente a self.analizar().
        """
        if resultado is None:
            resultado = self.analizar()

        tipo, info, detalles = resultado

        if tipo == "inconsistente":
            return "El sistema es inconsistente. No tiene solución."

        pivot_cols = info.get("pivot_cols", [])
        basic_vars = info.get("basic_vars", [])
        free_vars = info.get("free_vars", [])
        ecuaciones = detalles.get("ecuaciones_parametricas", [])

        # Tipo de solución
        if tipo == "única":
            encabezado = "Sistema consistente con solución única.\n"
        else:
            encabezado = "Sistema consistente con infinitas soluciones.\n"

        # Info de pivotes / variables
        if pivot_cols:
            cols_str = ", ".join(str(c + 1) for c in pivot_cols)
            encabezado += f"Columnas pivote: {cols_str}. "
        if basic_vars:
            encabezado += "Variables básicas: " + ", ".join(basic_vars) + ". "
        if free_vars:
            encabezado += "Variables libres: " + ", ".join(free_vars) + ". "
        encabezado += "\n\nConjunto solución (forma paramétrica):\n"

        cuerpo = "\n".join(ecuaciones) if ecuaciones else "(No se pudieron generar ecuaciones.)"
        return encabezado + cuerpo


# ===================== MÉTODO DE GAUSS (simple) =====================

class GaussEngine:
    """
    Método de Gauss (solo eliminación hacia adelante hasta forma escalonada).
    La GUI verá los pasos de Gauss. Para clasificar el sistema y obtener
    la solución en forma paramétrica, internamente reutilizamos Gauss-Jordan.
    """

    def __init__(self, matriz_aumentada):
        self.matriz_original = [[x for x in fila] for fila in matriz_aumentada]
        self.matriz_actual = [[x for x in fila] for fila in matriz_aumentada]
        self.log = []
        self.paso_actual = 0
        self.filas = len(matriz_aumentada)
        self.columnas = len(matriz_aumentada[0]) if matriz_aumentada else 0
        self.terminado = False
        self.fila_actual = 0
        self.col_actual = 0

        self._agregar_paso("Estado inicial (Gauss)")

    def _agregar_paso(self, descripcion, pivote_row=None, pivote_col=None):
        paso = PasoGauss(self.matriz_actual, descripcion, pivote_row, pivote_col)
        self.log.append(paso)
        self.paso_actual += 1

    def _intercambiar_filas(self, i, j):
        if i != j:
            self.matriz_actual[i], self.matriz_actual[j] = self.matriz_actual[j], self.matriz_actual[i]
            self._agregar_paso(f"Intercambiar fila {i + 1} con fila {j + 1}", i, self.col_actual)

    def _sumar_filas(self, fila_destino, fila_fuente, escalar):
        for j in range(self.columnas):
            self.matriz_actual[fila_destino][j] = (
                self.matriz_actual[fila_destino][j]
                + self.matriz_actual[fila_fuente][j] * escalar
            )
        desc = f"F{fila_destino + 1} → F{fila_destino + 1} + ({escalar})×F{fila_fuente + 1}"
        self._agregar_paso(desc, fila_destino, self.col_actual)

    def _encontrar_pivote(self, fila_inicio, col):
        for i in range(fila_inicio, self.filas):
            if not self.matriz_actual[i][col].es_cero():
                return i
        return None

    def siguiente(self):
        """
        Un paso de Gauss:
        - Busca pivote en la columna actual,
        - intercambia filas si es necesario,
        - hace ceros *debajo* del pivote.
        """
        if self.terminado:
            return None

        while self.fila_actual < self.filas and self.col_actual < self.columnas - 1:
            fila_pivote = self._encontrar_pivote(self.fila_actual, self.col_actual)

            if fila_pivote is None:
                # No hay pivote en esta columna
                self.col_actual += 1
                continue

            # Subir la fila con pivote
            if fila_pivote != self.fila_actual:
                self._intercambiar_filas(self.fila_actual, fila_pivote)
                return self.log[-1]

            # Pivote actual
            pivote = self.matriz_actual[self.fila_actual][self.col_actual]

            # Hacer ceros debajo del pivote
            for i in range(self.fila_actual + 1, self.filas):
                if not self.matriz_actual[i][self.col_actual].es_cero():
                    factor = -(self.matriz_actual[i][self.col_actual] / pivote)
                    self._sumar_filas(i, self.fila_actual, factor)
                    return self.log[-1]

            # Ya no hay nada debajo en esta columna: avanzar
            self.fila_actual += 1
            self.col_actual += 1

        self.terminado = True
        self._agregar_paso("Proceso completado (Gauss - forma escalonada)")
        return self.log[-1]

    # ----------- análisis: reutilizamos Gauss-Jordan internamente -----------

    def analizar(self):
        """
        Ejecuta Gauss si falta y luego usa GaussJordanEngine sobre la matriz
        escalonada actual para obtener tipo, pivotes y solución paramétrica.
        """
        if not self.terminado:
            while not self.terminado:
                self.siguiente()

        # Crear un Gauss-Jordan auxiliar SOLO para analizar
        gj = GaussJordanEngine(self.matriz_actual)
        while not gj.terminado:
            gj.siguiente()

        # Usamos su método analizar, que ya devuelve el triple
        return gj.analizar()

    def conjunto_solucion(self, resultado=None):
        """
        Mismo formato de salida que GaussJordanEngine.conjunto_solucion.
        """
        if resultado is None:
            resultado = self.analizar()

        tipo, info, detalles = resultado

        if tipo == "inconsistente":
            return "El sistema es inconsistente. No tiene solución."

        pivot_cols = info.get("pivot_cols", [])
        basic_vars = info.get("basic_vars", [])
        free_vars = info.get("free_vars", [])
        ecuaciones = detalles.get("ecuaciones_parametricas", [])

        if tipo == "única":
            encabezado = "Sistema consistente con solución única.\n"
        else:
            encabezado = "Sistema consistente con infinitas soluciones.\n"

        if pivot_cols:
            cols_str = ", ".join(str(c + 1) for c in pivot_cols)
            encabezado += f"Columnas pivote: {cols_str}. "
        if basic_vars:
            encabezado += "Variables básicas: " + ", ".join(basic_vars) + ". "
        if free_vars:
            encabezado += "Variables libres: " + ", ".join(free_vars) + ". "
        encabezado += "\n\nConjunto solución (forma paramétrica):\n"

        cuerpo = "\n".join(ecuaciones) if ecuaciones else "(No se pudieron generar ecuaciones.)"
        return encabezado + cuerpo
