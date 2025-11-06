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
            self.matriz_actual[fila_destino][j] = (self.matriz_actual[fila_destino][j] +
                                                   self.matriz_actual[fila_fuente][j] * escalar)
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

        # Fase de eliminación hacia adelante
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

            # Eliminar en otras filas
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

    def analizar(self):
        """Analiza el sistema y devuelve información sobre la solución"""
        if not self.terminado:
            # Completar el proceso si no está terminado
            while not self.terminado:
                self.siguiente()

        matriz_reducida = self.matriz_actual
        filas = self.filas
        cols = self.columnas - 1  # Columnas sin el vector b

        # Contar pivotes
        pivotes = []
        for i in range(filas):
            for j in range(cols):
                if not matriz_reducida[i][j].es_cero():
                    pivotes.append((i, j))
                    break

        rank_A = len(pivotes)
        rank_AB = rank_A

        # Verificar consistencia
        for i in range(filas):
            fila_cero = all(matriz_reducida[i][j].es_cero() for j in range(cols))
            if fila_cero and not matriz_reducida[i][cols].es_cero():
                return "inconsistente", {"rank_A": rank_A, "rank_AB": rank_AB}, {}

        if rank_A == cols:
            return "única", {"rank_A": rank_A, "rank_AB": rank_AB}, {"solucion": "única"}
        else:
            return "infinitas", {"rank_A": rank_A, "rank_AB": rank_AB}, {"variables_libres": cols - rank_A}

    def conjunto_solucion(self, resultado):
        """Genera descripción del conjunto solución"""
        tipo, data, clasificacion = resultado

        if tipo == "única":
            sol = []
            for i in range(min(self.filas, self.columnas - 1)):
                sol.append(self.matriz_actual[i][self.columnas - 1])
            vars_str = ", ".join(f"x{i + 1} = {sol[i]}" for i in range(len(sol)))
            return f"Solución única: {vars_str}"

        elif tipo == "infinitas":
            vars_libres = clasificacion.get("variables_libres", 0)
            return f"Infinitas soluciones ({vars_libres} variable(s) libre(s))"

        else:  # inconsistente
            return "Sistema inconsistente (sin solución)"