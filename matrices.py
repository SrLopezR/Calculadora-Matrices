# matrices.py
from fraccion import Fraccion, parse_token

def sumar_matrices(A, B):
    """Suma dos matrices de igual tamaño y devuelve (resultado, pasos)."""
    m, n = len(A), len(A[0])
    if len(B) != m or len(B[0]) != n:
        raise ValueError("Las matrices deben tener las mismas dimensiones para sumarse.")

    R = []
    pasos = []

    # Recorremos cada posición (i,j) sumando elemento a elemento.
    # Cada elemento A[i][j] y B[i][j] representan posiciones equivalentes
    # dentro de las mismas coordenadas del espacio matricial.
    for i in range(m):
        fila = []
        for j in range(n):
            s = A[i][j] + B[i][j]
            fila.append(s)
            pasos.append(f"R[{i+1},{j+1}] = {A[i][j]} + {B[i][j]} = {s}")
        R.append(fila)

    return R, pasos


def multiplicar_matrices(A, B):
    """Multiplica dos matrices A y B y devuelve (resultado, pasos)."""
    m, n = len(A), len(A[0])  # A es m x n
    p, q = len(B), len(B[0])  # B es p x q

    if n != p:
        raise ValueError(
            "Para multiplicar matrices: columnas de A deben coincidir con filas de B."
        )

    # R tendrá dimensiones m x q
    R = [[Fraccion(0) for _ in range(q)] for _ in range(m)]
    pasos = []

    # La multiplicación de matrices puede interpretarse como:
    #  - Tomar una fila de A
    #  - Tomar una columna de B
    #  - Calcular su producto escalar
    #
    # Por teoría, eso equivale a multiplicar A por la transpuesta de B,
    # si quisiéramos ver las filas de B en vez de sus columnas (A * Bᵗ).
    for i in range(m):
        for j in range(q):
            sum_terms = []
            acc = Fraccion(0)
            for k in range(n):
                prod = A[i][k] * B[k][j]
                acc = acc + prod
                sum_terms.append(f"{A[i][k]}*{B[k][j]}")
                pasos.append(f"Parcial R[{i+1},{j+1}] = {A[i][k]}*{B[k][j]} = {prod}")
            R[i][j] = acc
            pasos.append(
                f"R[{i+1},{j+1}] = " + " + ".join(sum_terms) + f" = {acc}"
            )

    return R, pasos


def Transpuesta(M):
    """Devuelve la transpuesta de la matriz M."""
    # La transpuesta intercambia filas por columnas:
    #  - Si M es m x n, su transpuesta T será n x m.
    #  - T[j][i] = M[i][j].
    #
    # En términos vectoriales, los vectores fila de M se convierten en
    # vectores columna en T, y viceversa.
    #

    if not M or not M[0]:
        return []
    m, n = len(M), len(M[0])
    T = [[Fraccion(0) for _ in range(m)] for _ in range(n)]
    for i in range(m):
        for j in range(n):
            # Aquí se realiza el intercambio de índices.
            # Cada elemento (i,j) de M pasa a la posición (j,i) en T.
            T[j][i] = M[i][j]
            # Si visualizamos M como una tabla:
            # Fila i → Columna i en T (rotación sobre diagonal principal)
    return T


def multiplicar_escalar_matriz(escalar, matriz):
    """
    Multiplica un escalar por una matriz, mostrando los pasos.
    Usa la clase Fraccion para manejar los valores.
    """
    pasos = []

    # Convertir escalar a Fraccion
    if isinstance(escalar, str):
        escalar = parse_token(escalar)
    elif not isinstance(escalar, Fraccion):
        escalar = Fraccion(escalar)

    # Paso 1: matriz original
    pasos.append("Matriz original:\n" + formatear_matriz(matriz))

    # Paso 2: multiplicación elemento por elemento (en texto)
    paso_operacion = []
    for fila in matriz:
        fila_operacion = []
        for elemento in fila:
            elem = elemento if isinstance(elemento, Fraccion) else parse_token(str(elemento))
            fila_operacion.append(f"{escalar}*({elem})")
        paso_operacion.append(fila_operacion)
    pasos.append("Multiplicación escalar por cada elemento:\n" + formatear_matriz(paso_operacion))

    # Paso 3: resultado final
    matriz_resultado = []
    for fila in matriz:
        fila_res = []
        for elemento in fila:
            elem = elemento if isinstance(elemento, Fraccion) else parse_token(str(elemento))
            fila_res.append(escalar * elem)
        matriz_resultado.append(fila_res)

    pasos.append("Resultado final:\n" + formatear_matriz(matriz_resultado))

    return matriz_resultado, pasos


def formatear_matriz(matriz):
    """
    Devuelve la matriz como texto con corchetes, usando str() de Fraccion.
    """
    texto = ""
    for fila in matriz:
        texto += "[" + "  ".join(str(x) for x in fila) + "]\n"
    return texto

def formatear_matriz_aumentada(matriz, split):
    """
    Devuelve la matriz con separador vertical en la columna 'split' (índice entero),
    útil para mostrar [A | I] o estados intermedios de Gauss-Jordan.
    """
    texto = ""
    for fila in matriz:
        izq = "  ".join(str(x) for x in fila[:split])
        der = "  ".join(str(x) for x in fila[split:])
        texto += f"[{izq} | {der}]\n"
    return texto

def comprobar_invertibilidad(M):
    """
    Comprueba invertibilidad aplicando SOLO eliminación hacia adelante (Gauss),
    sin normalizar pivotes ni hacer ceros arriba (no Gauss-Jordan).
    Requiere M cuadrada. Devuelve:
        (es_invertible: bool, U: matriz escalonada, pasos: list[str], num_pivotes: int, determinante: Fraccion)
    """
    from copy import deepcopy
    n = len(M)
    if n == 0 or len(M[0]) != n:
        raise ValueError("La matriz debe ser cuadrada para comprobar invertibilidad.")

    A = deepcopy(M)  # trabajamos sobre copia
    pasos = []
    pasos.append("Estado inicial (A):\n" + formatear_matriz(A))

    swaps = 0
    pivotes = 0
    det = Fraccion(1)

    for i in range(n):
        # Buscar pivote ≠ 0 en columna i
        piv_row = None
        for r in range(i, n):
            if not A[r][i].es_cero():
                piv_row = r
                break

        if piv_row is None:
            # Columna sin pivote → rango < n → no invertible
            pasos.append(f"No hay pivote en la columna {i+1}. Rango < n ⇒ matriz NO invertible.")
            # Determinante = 0
            return (False, A, pasos, pivotes, Fraccion(0))

        # Intercambio si el pivote no está ya en la fila i
        if piv_row != i:
            A[i], A[piv_row] = A[piv_row], A[i]
            swaps += 1
            pasos.append(f"Intercambiar F{i+1} ↔ F{piv_row+1}\n" + formatear_matriz(A))

        # Registrar pivote (SIN normalizar)
        piv = A[i][i]
        det = det * piv  # producto de pivotes (ajustaremos signo por swaps al final)
        pivotes += 1

        # Hacer ceros debajo del pivote (eliminación hacia adelante)
        for r in range(i+1, n):
            factor = A[r][i]
            if factor.es_cero():
                continue
            # F_r = F_r - (factor/piv)*F_i  pero SIN cambiar la fila pivote
            mult = factor / piv
            for c in range(i, n):
                A[r][c] = A[r][c] - mult * A[i][c]
            pasos.append(f"F{r+1} = F{r+1} - ({factor}/{piv})·F{i+1}\n" + formatear_matriz(A))

        pasos.append(f"Columna {i+1} lista (cero debajo del pivote):\n" + formatear_matriz(A))

    # Ajuste de signo por número de swaps
    if swaps % 2 == 1:
        det = Fraccion(-det.num, det.den)

    pasos.append(f"Pivotes encontrados: {pivotes} de {n}.")
    pasos.append(f"Determinante: {det}")
    return (True, A, pasos, pivotes, det)

def inversa_matriz(M):
    """
    Calcula la inversa de una matriz cuadrada M usando Gauss-Jordan con paso a paso.
    Devuelve (inversa, pasos:list[str]). Lanza ValueError si no es cuadrada o no es invertible.
    """
    from copy import deepcopy
    m = len(M)
    if m == 0 or len(M[0]) != m:
        raise ValueError("La matriz debe ser cuadrada para calcular su inversa.")

    # Construir [A | I]
    A = deepcopy(M)
    I = [[Fraccion(1 if i == j else 0) for j in range(m)] for i in range(m)]
    for i in range(m):
        A[i].extend(I[i])

    pasos = []
    pasos.append("Matriz aumentada inicial [A | I]:\n" + formatear_matriz_aumentada(A, m))

    # Gauss-Jordan columna por columna
    for i in range(m):
        # 1) Asegurar pivote en (i,i): buscar fila con entrada != 0 en la columna i
        if A[i][i].es_cero():
            swap_row = None
            for k in range(i+1, m):
                if not A[k][i].es_cero():
                    swap_row = k
                    break
            if swap_row is None:
                raise ValueError("La matriz no es invertible (determinante = 0).")
            A[i], A[swap_row] = A[swap_row], A[i]
            pasos.append(f"Intercambiar filas: F{i+1} ↔ F{swap_row+1}\n" +
                         formatear_matriz_aumentada(A, m))

        # 2) Normalizar pivote a 1
        piv = A[i][i]
        if piv.es_cero():
            # Seguridad adicional
            raise ValueError("La matriz no es invertible (determinante = 0).")
        if not piv.es_uno():
            inv = Fraccion(piv.den, piv.num)  # 1/pivote
            for j in range(2*m):
                A[i][j] = A[i][j] * inv
            pasos.append(f"Normalizar pivote: F{i+1} = (1/({piv}))·F{i+1}\n" +
                         formatear_matriz_aumentada(A, m))

        # 3) Anular el resto de la columna i
        for k in range(m):
            if k == i: 
                continue
            factor = A[k][i]
            if factor.es_cero():
                continue
            for j in range(2*m):
                A[k][j] = A[k][j] - factor * A[i][j]
            pasos.append(f"Anular columna {i+1}: F{k+1} = F{k+1} - ({factor})·F{i+1}\n" +
                         formatear_matriz_aumentada(A, m))

        pasos.append(f"Columna {i+1} lista (pivote 1 y ceros arriba/abajo):\n" +
                     formatear_matriz_aumentada(A, m))

    # Extraer inversa (parte derecha)
    inv = [fila[m:] for fila in A]
    pasos.append("Matriz inversa obtenida:\n" + formatear_matriz(inv))
    return inv, pasos

def determinante_matriz(M):
    """
    Calcula el determinante de una matriz cuadrada M usando eliminación gaussiana (no normaliza pivotes).
    Devuelve (determinante: Fraccion, pasos: list[str]).
    - Si hay intercambios de filas, el determinante cambia de signo por cada intercambio.
    - Si no se encuentra pivote en alguna columna, el determinante es 0.
    """
    from copy import deepcopy
    n = len(M)
    if n == 0 or len(M[0]) != n:
        raise ValueError("La matriz debe ser cuadrada para calcular el determinante.")

    A = deepcopy(M)
    pasos = []
    pasos.append("Estado inicial (A):\n" + formatear_matriz(A))

    swaps = 0
    det = Fraccion(1)

    for i in range(n):
        # Buscar pivote en o debajo de la fila i
        piv_row = None
        for r in range(i, n):
            if not A[r][i].es_cero():
                piv_row = r
                break

        if piv_row is None:
            pasos.append(f"No hay pivote en la columna {i+1} ⇒ determinante = 0")
            return Fraccion(0), pasos

        if piv_row != i:
            A[i], A[piv_row] = A[piv_row], A[i]
            swaps += 1
            pasos.append(f"Intercambiar F{i+1} ↔ F{piv_row+1}\n" + formatear_matriz(A))

        piv = A[i][i]
        det = det * piv  # producto de pivotes sin normalizar

        # Eliminar hacia adelante (ceros abajo del pivote)
        for r in range(i + 1, n):
            factor = A[r][i]
            if factor.es_cero():
                continue
            mult = factor / piv
            for c in range(i, n):
                A[r][c] = A[r][c] - mult * A[i][c]
            pasos.append(f"F{r+1} = F{r+1} - ({factor}/{piv})·F{i+1}\n" + formatear_matriz(A))

        pasos.append(f"Columna {i+1} lista.\n" + formatear_matriz(A))

    # Ajuste de signo por swaps
    if swaps % 2 == 1:
        det = Fraccion(-det.num, det.den)
        pasos.append("Número impar de intercambios ⇒ cambiar signo del determinante.")

    pasos.append(f"Determinante = {det}")
    return det, pasos

def determinante_cofactores(M, prefer="auto"):
    """
    Determinante por expansión de cofactores (Laplace) con registro de pasos 'tipo libro'.
    Devuelve (det: Fraccion, pasos: list[str]).
    - prefer: "auto" (elige fila/col con más ceros), "fila0", "col0", o ("fila", i), ("col", j).
    Nota: Complejidad O(n!), usar para n pequeños o cuando se requieren pasos teóricos.
    """
    from copy import deepcopy

    n = len(M)
    if n == 0 or len(M[0]) != n:
        raise ValueError("La matriz debe ser cuadrada para calcular el determinante.")

    A = deepcopy(M)
    pasos = []

    def sgn(k):  # (-1)^k
        return 1 if k % 2 == 0 else -1

    def minor(mat, i, j):
        return [row[:j] + row[j+1:] for r, row in enumerate(mat) if r != i]

    def fr(s):
        # helper para formateo corto del Fraccion
        return str(s)

    def choose_line(mat):
        n = len(mat)
        if isinstance(prefer, tuple) and prefer and prefer[0] in ("fila", "col"):
            kind, idx = prefer
            if kind == "fila" and 0 <= idx < n:
                return ("fila", idx)
            if kind == "col" and 0 <= idx < n:
                return ("col", idx)

        if prefer == "fila0":
            return ("fila", 0)
        if prefer == "col0":
            return ("col", 0)

        # AUTO: fila o columna con más ceros (rompe empates favoreciendo primera columna)
        best = ("col", 0, -1)  # (kind, idx, zeros)
        # columnas
        for j in range(n):
            z = sum(1 for i in range(n) if mat[i][j].es_cero())
            if z > best[2]:
                best = ("col", j, z)
        # filas
        for i in range(n):
            z = sum(1 for j in range(n) if mat[i][j].es_cero())
            if z > best[2]:
                best = ("fila", i, z)
        return (best[0], best[1])

    def laplace(mat, depth=0):
        n = len(mat)
        indent = "  " * depth

        if n == 1:
            pasos.append(f"{indent}det {formatear_matriz(mat)} = {fr(mat[0][0])}")
            return mat[0][0]

        kind, idx = choose_line(mat)

        # Línea de “enunciado” tipo: det A = a11·C11 + 0·C21 + ...
        terms = []
        if kind == "col":
            j = idx
            for i in range(n):
                a = mat[i][j]
                coef = "0" if a.es_cero() else fr(a)
                sign = "-" if sgn(i + j) < 0 else "+"
                # para el primer término no imprimimos el "+" inicial
                t = f"{('- ' if sign=='-' else '')}{coef}·C{i+1}{j+1}" if i == 0 else f" {'-' if sign=='-' else '+'} {coef}·C{i+1}{j+1}"
                terms.append(t)
            encabezado = f"{indent}det = " + "".join(terms)
            pasos.append(encabezado)

            det_total = Fraccion(0)
            for i in range(n):
                a = mat[i][j]
                if a.es_cero():
                    continue
                sign = Fraccion(1 if sgn(i + j) > 0 else -1, 1)
                Mij = minor(mat, i, j)
                pasos.append(f"{indent}→ expandiendo por columna {j+1}, elemento a{i+1}{j+1}={fr(a)} (signo {('+' if sign.num>0 else '-')})")
                pasos.append(f"{indent}   menor M{i+1}{j+1} =\n{formatear_matriz(Mij)}")
                d = laplace(Mij, depth + 1)
                det_total = det_total + a * sign * d
            if depth == 0:
                pasos.append(f"{indent}Determinante = {fr(det_total)}")
            return det_total

        else:  # kind == "fila"
            i = idx
            for j in range(n):
                a = mat[i][j]
                coef = "0" if a.es_cero() else fr(a)
                sign = "-" if sgn(i + j) < 0 else "+"
                t = f"{('- ' if sign=='-' else '')}{coef}·C{i+1}{j+1}" if j == 0 else f" {'-' if sign=='-' else '+'} {coef}·C{i+1}{j+1}"
                terms.append(t)
            encabezado = f"{indent}det = " + "".join(terms)
            pasos.append(encabezado)

            det_total = Fraccion(0)
            for j in range(n):
                a = mat[i][j]
                if a.es_cero():
                    continue
                sign = Fraccion(1 if sgn(i + j) > 0 else -1, 1)
                Mij = minor(mat, i, j)
                pasos.append(f"{indent}→ expandiendo por fila {i+1}, elemento a{i+1}{j+1}={fr(a)} (signo {('+' if sign.num>0 else '-')})")
                pasos.append(f"{indent}   menor M{i+1}{j+1} =\n{formatear_matriz(Mij)}")
                d = laplace(Mij, depth + 1)
                det_total = det_total + a * sign * d
            if depth == 0:
                pasos.append(f"{indent}Determinante = {fr(det_total)}")
            return det_total

    det = laplace(A, 0)
    return det, pasos


def regla_cramer(A, b):
    """
    Resuelve un sistema lineal Ax = b usando la Regla de Cramer.
    Devuelve (solución: list[Fraccion], pasos: list[str])
    """
    n = len(A)
    pasos = []

    # Paso 1: Calcular determinante de A
    pasos.append("Paso 1: Calcular determinante de A")
    det_A, pasos_det = determinante_matriz(A)
    pasos.extend(pasos_det)
    pasos.append(f"det(A) = {det_A}")

    if det_A.es_cero():
        raise ValueError("La matriz A no es invertible (det(A) = 0). No se puede aplicar la Regla de Cramer.")

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
        pasos.extend(pasos_det_i)
        pasos.append(f"det(A_{i + 1}(b)) = {det_A_i}")

        # Calcular x_i = det(A_i(b)) / det(A)
        x_i = det_A_i / det_A
        pasos.append(f"x_{i + 1} = det(A_{i + 1}(b)) / det(A) = {det_A_i} / {det_A} = {x_i}")
        solucion.append(x_i)
        pasos.append("")  # Línea en blanco

    return solucion, pasos

# ====== PRUEBAS RÁPIDAS ======
if __name__ == "__main__":
    A = [[Fraccion(1), Fraccion(2)], [Fraccion(3), Fraccion(4)]]
    B = [[Fraccion(2), Fraccion(0)], [Fraccion(1), Fraccion(2)]]

    print("=== Suma de matrices ===")
    S, pasos_suma = sumar_matrices(A, B)
    for p in pasos_suma:
        print(p)
    print("Resultado:", [[str(x) for x in fila] for fila in S])

    print("\n=== Multiplicación de matrices ===")
    M, pasos_mult = multiplicar_matrices(A, B)
    for p in pasos_mult:
        print(p)
    print("Resultado:", [[str(x) for x in fila] for fila in M])
