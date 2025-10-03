# gauss.py
from fraccion import Fraccion

from dataclasses import dataclass

@dataclass
class Step:
    descripcion: str
    matriz: list
    pivote_row: int | None
    pivote_col: int | None

class GaussJordanEngine:
    def __init__(self, aug_matrix):
        self.A = [fila[:] for fila in aug_matrix]
        self.m = len(self.A)
        self.n = len(self.A[0]) - 1
        self.fila = 0
        self.col_pivotes = []
        self.terminado = False
        self.log = []
        self._snapshot("Inicio", None, None)

    def _copy(self):
        return [[Fraccion(x.num, x.den) for x in fila] for fila in self.A]

    def _snapshot(self, descripcion, prow, pcol):
        self.log.append(Step(descripcion, self._copy(), prow, pcol))

    def siguiente(self):
        if self.terminado:
            return None
        while self.col_actual() < self.n and self.fila < self.m:
            c = self.col_actual()
            sel = None
            for r in range(self.fila, self.m):
                if not self.A[r][c].es_cero():
                    sel = r
                    break
            if sel is None:
                self._snapshot(f"No hay pivote en la columna {c+1}.", None, c)
                self.col_pivotes.append(None)
                self.fila += 1
                return self.log[-1]
            if sel != self.fila:
                self.A[self.fila], self.A[sel] = self.A[sel], self.A[self.fila]
                self._snapshot(f"Intercambiar F{self.fila+1} ↔ F{sel+1}", self.fila, c)
                return self.log[-1]
            piv = self.A[self.fila][c]
            if not piv.es_uno():
                inv = Fraccion(piv.den, piv.num)
                for j in range(c, self.n+1):
                    self.A[self.fila][j] = self.A[self.fila][j] * inv
                self._snapshot(f"Multiplicar F{self.fila+1} por 1/({piv}) para pivote = 1", self.fila, c)
                return self.log[-1]
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
            self.col_pivotes.append(c)
            self.fila += 1
            return self.log[-1]
        self._snapshot("Fin Gauss‑Jordan", None, None)
        self.terminado = True
        return self.log[-1]

    def col_actual(self):
        return len(self.col_pivotes)

    def analizar(self):
        A = self.A
        m, n = self.m, self.n
        for i in range(m):
            if all(A[i][j].es_cero() for j in range(n)) and not A[i][n].es_cero():
                return ("inconsistente", None)
        if len([x for x in self.col_pivotes if x is not None]) == n:
            sol = [A[i][n] for i in range(n)]
            return ("única", sol)
        libres = [j for j in range(n) if j not in self.col_pivotes]
        particular = [Fraccion(0) for _ in range(n)]
        for i, pc in enumerate(self.col_pivotes):
            if pc is not None:
                particular[pc] = A[i][n]
        base = []
        for f in libres:
            vec = [Fraccion(0) for _ in range(n)]
            vec[f] = Fraccion(1)
            for i, pc in enumerate(self.col_pivotes):
                if pc is not None:
                    vec[pc] = -A[i][f]
            base.append(vec)
        return ("infinitas", ( libres, base))
