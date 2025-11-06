from fraccion import Fraccion
import copy


def formatear_matriz(M):
    """Convierte matriz a string formateado"""
    if not M:
        return "[]"

    filas_str = []
    for fila in M:
        fila_str = "[ " + " ".join(f"{str(x):>8}" for x in fila) + " ]"
        filas_str.append(fila_str)

    return "\n".join(filas_str)


def sumar_matrices(A, B):
    """Suma dos matrices con pasos detallados"""
    if len(A) != len(B) or len(A[0]) != len(B[0]):
        raise ValueError("Las matrices deben tener las mismas dimensiones")

    pasos = []
    resultado = []

    pasos.append("Suma de matrices:")
    pasos.append(f"A = \n{formatear_matriz(A)}")
    pasos.append(f"B = \n{formatear_matriz(B)}")
    pasos.append("")

    for i in range(len(A)):
        fila_resultado = []
        for j in range(len(A[0])):
            valor = A[i][j] + B[i][j]
            fila_resultado.append(valor)
            pasos.append(
                f"C[{i + 1},{j + 1}] = A[{i + 1},{j + 1}] + B[{i + 1},{j + 1}] = {A[i][j]} + {B[i][j]} = {valor}")
        resultado.append(fila_resultado)
        pasos.append("")

    pasos.append("Resultado:")
    pasos.append(formatear_matriz(resultado))

    return resultado, pasos


def multiplicar_matrices(A, B):
    """Multiplica dos matrices con pasos detallados"""
    if len(A[0]) != len(B):
        raise ValueError("El número de columnas de A debe igualar el número de filas de B")

    pasos = []
    resultado = [[Fraccion(0) for _ in range(len(B[0]))] for _ in range(len(A))]

    pasos.append("Multiplicación de matrices:")
    pasos.append(f"A ({len(A)}×{len(A[0])}) = \n{formatear_matriz(A)}")
    pasos.append(f"B ({len(B)}×{len(B[0])}) = \n{formatear_matriz(B)}")
    pasos.append("")

    for i in range(len(A)):
        for j in range(len(B[0])):
            pasos.append(f"Calculando C[{i + 1},{j + 1}]:")
            suma_parcial = Fraccion(0)
            for k in range(len(B)):
                producto = A[i][k] * B[k][j]
                suma_parcial = suma_parcial + producto
                pasos.append(f"  + A[{i + 1},{k + 1}]×B[{k + 1},{j + 1}] = {A[i][k]} × {B[k][j]} = {producto}")
            resultado[i][j] = suma_parcial
            pasos.append(f"  C[{i + 1},{j + 1}] = {suma_parcial}")
            pasos.append("")

    pasos.append("Resultado:")
    pasos.append(formatear_matriz(resultado))

    return resultado, pasos


def multiplicar_escalar_matriz(escalar_str, A):
    """Multiplica escalar por matriz con pasos detallados"""
    try:
        escalar = Fraccion(escalar_str)
    except:
        raise ValueError(f"Escalar inválido: {escalar_str}")

    pasos = []
    resultado = []

    pasos.append(f"Multiplicación escalar: {escalar} × A")
    pasos.append(f"A = \n{formatear_matriz(A)}")
    pasos.append("")

    for i in range(len(A)):
        fila_resultado = []
        for j in range(len(A[0])):
            valor = escalar * A[i][j]
            fila_resultado.append(valor)
            pasos.append(f"B[{i + 1},{j + 1}] = {escalar} × {A[i][j]} = {valor}")
        resultado.append(fila_resultado)
        pasos.append("")

    pasos.append("Resultado:")
    pasos.append(formatear_matriz(resultado))

    return resultado, pasos


def Transpuesta(A):
    """Calcula la transpuesta de una matriz"""
    if not A:
        return []

    filas = len(A)
    columnas = len(A[0])

    transpuesta = [[Fraccion(0) for _ in range(filas)] for _ in range(columnas)]

    for i in range(filas):
        for j in range(columnas):
            transpuesta[j][i] = A[i][j]

    return transpuesta


def determinante_matriz(A):
    """Calcula determinante por eliminación gaussiana"""
    if len(A) != len(A[0]):
        raise ValueError("La matriz debe ser cuadrada")

    n = len(A)
    M = copy.deepcopy(A)
    det = Fraccion(1)
    pasos = []

    pasos.append(f"Cálculo del determinante de matriz {n}×{n}")
    pasos.append(f"Matriz inicial:\n{formatear_matriz(M)}")
    pasos.append("")

    for i in range(n):
        # Encontrar pivote
        pivote = i
        while pivote < n and M[pivote][i].es_cero():
            pivote += 1

        if pivote == n:
            det = Fraccion(0)
            pasos.append("Fila de ceros encontrada → det = 0")
            break

        if pivote != i:
            M[i], M[pivote] = M[pivote], M[i]
            det = det * Fraccion(-1)
            pasos.append(f"Intercambio fila {i + 1} con fila {pivote + 1} → det = -det")
            pasos.append(f"Matriz después del intercambio:\n{formatear_matriz(M)}")

        # Usar fila i para eliminar
        for j in range(i + 1, n):
            if not M[j][i].es_cero():
                factor = M[j][i] / M[i][i]
                pasos.append(f"Eliminar elemento ({j + 1},{i + 1}) usando factor {factor}")

                for k in range(i, n):
                    M[j][k] = M[j][k] - factor * M[i][k]

        det = det * M[i][i]
        pasos.append(f"Multiplicar det por pivote M[{i + 1},{i + 1}] = {M[i][i]}")
        pasos.append(f"det parcial = {det}")
        pasos.append(f"Matriz actual:\n{formatear_matriz(M)}")
        pasos.append("")

    pasos.append(f"Determinante final: {det}")
    return det, pasos


def determinante_cofactores(A, prefer="auto"):
    """Calcula determinante por método de cofactores"""
    n = len(A)

    if n == 1:
        return A[0][0], ["Matriz 1×1 → det = A[1,1] = " + str(A[0][0])]

    if n == 2:
        det = A[0][0] * A[1][1] - A[0][1] * A[1][0]
        pasos = [
            "Matriz 2×2:",
            f"det = ({A[0][0]})×({A[1][1]}) - ({A[0][1]})×({A[1][0]})",
            f"det = {A[0][0] * A[1][1]} - {A[0][1] * A[1][0]}",
            f"det = {det}"
        ]
        return det, pasos

    pasos = []
    pasos.append(f"Calculando determinante {n}×{n} por cofactores")
    pasos.append(f"Matriz:\n{formatear_matriz(A)}")

    # Elegir fila/columna con más ceros
    if prefer == "auto":
        fila_ceros = [sum(1 for x in fila if x.es_cero()) for fila in A]
        col_ceros = [sum(1 for i in range(n) if A[i][j].es_cero()) for j in range(n)]

        if max(fila_ceros) >= max(col_ceros):
            prefer = "fila0"
            fila = fila_ceros.index(max(fila_ceros))
        else:
            prefer = "col0"
            col = col_ceros.index(max(col_ceros))

    det = Fraccion(0)
    if prefer == "fila0":
        fila = 0  # Simplificación: usar primera fila
        pasos.append(f"Desarrollando por fila {fila + 1}")

        for j in range(n):
            if not A[fila][j].es_cero():
                signo = Fraccion(1) if (fila + j) % 2 == 0 else Fraccion(-1)
                menor = _submatriz(A, fila, j)
                cofactor_det, cofactor_pasos = determinante_cofactores(menor, prefer)
                termino = signo * A[fila][j] * cofactor_det

                pasos.append(f"Término ({fila + 1},{j + 1}): signo={signo}, elemento={A[fila][j]}")
                pasos.append(f"Menor:\n{formatear_matriz(menor)}")
                pasos.extend(["  " + p for p in cofactor_pasos])
                pasos.append(f"cofactor = {signo} × {A[fila][j]} × {cofactor_det} = {termino}")

                det = det + termino
                pasos.append(f"Suma parcial: {det}")

    pasos.append(f"Determinante final: {det}")
    return det, pasos


def _submatriz(A, fila_omitir, col_omitir):
    """Crea submatriz omitiendo fila y columna dadas"""
    return [
        [A[i][j] for j in range(len(A)) if j != col_omitir]
        for i in range(len(A)) if i != fila_omitir
    ]


def determinante_sarrus(A):
    """Calcula determinante por regla de Sarrus (solo 3x3)"""
    n = len(A)
    if n != 3:
        raise ValueError("La regla de Sarrus solo aplica para matrices 3×3")

    pasos = []
    pasos.append("Regla de Sarrus para matriz 3×3")
    pasos.append(f"Matriz:\n{formatear_matriz(A)}")

    # Términos positivos (diagonales principales)
    pos1 = A[0][0] * A[1][1] * A[2][2]
    pos2 = A[0][1] * A[1][2] * A[2][0]
    pos3 = A[0][2] * A[1][0] * A[2][1]

    # Términos negativos (diagonales secundarias)
    neg1 = A[0][2] * A[1][1] * A[2][0]
    neg2 = A[0][0] * A[1][2] * A[2][1]
    neg3 = A[0][1] * A[1][0] * A[2][2]

    det = pos1 + pos2 + pos3 - neg1 - neg2 - neg3

    pasos.append("Términos positivos:")
    pasos.append(f"  A₁₁×A₂₂×A₃₃ = {A[0][0]}×{A[1][1]}×{A[2][2]} = {pos1}")
    pasos.append(f"  A₁₂×A₂₃×A₃₁ = {A[0][1]}×{A[1][2]}×{A[2][0]} = {pos2}")
    pasos.append(f"  A₁₃×A₂₁×A₃₂ = {A[0][2]}×{A[1][0]}×{A[2][1]} = {pos3}")

    pasos.append("Términos negativos:")
    pasos.append(f"  A₁₃×A₂₂×A₃₁ = {A[0][2]}×{A[1][1]}×{A[2][0]} = {neg1}")
    pasos.append(f"  A₁₁×A₂₃×A₃₂ = {A[0][0]}×{A[1][2]}×{A[2][1]} = {neg2}")
    pasos.append(f"  A₁₂×A₂₁×A₃₃ = {A[0][1]}×{A[1][0]}×{A[2][2]} = {neg3}")

    pasos.append(f"det = ({pos1} + {pos2} + {pos3}) - ({neg1} + {neg2} + {neg3})")
    pasos.append(f"det = {pos1 + pos2 + pos3} - {neg1 + neg2 + neg3}")
    pasos.append(f"det = {det}")

    return det, pasos


def comprobar_invertibilidad(A):
    """Comprueba si una matriz es invertible usando Gauss"""
    n = len(A)
    M = copy.deepcopy(A)
    pasos = []
    pivotes = []

    pasos.append("Comprobando invertibilidad por eliminación Gaussiana")
    pasos.append(f"Matriz {n}×{n}:\n{formatear_matriz(M)}")

    for i in range(n):
        # Buscar pivote
        pivote = i
        while pivote < n and M[pivote][i].es_cero():
            pivote += 1

        if pivote == n:
            pasos.append(f"No se encontró pivote en columna {i + 1} → matriz no invertible")
            return False, M, pasos, pivotes, Fraccion(0)

        if pivote != i:
            M[i], M[pivote] = M[pivote], M[i]
            pasos.append(f"Intercambio fila {i + 1} con fila {pivote + 1}")

        pivotes.append((i, i))

        # Eliminar
        for j in range(i + 1, n):
            if not M[j][i].es_cero():
                factor = M[j][i] / M[i][i]
                pasos.append(f"Eliminar fila {j + 1} usando factor {factor}")

                for k in range(i, n):
                    M[j][k] = M[j][k] - factor * M[i][k]

        pasos.append(f"Matriz después de columna {i + 1}:\n{formatear_matriz(M)}")

    # Calcular determinante aproximado (producto de pivotes)
    det = Fraccion(1)
    for i in range(n):
        det = det * M[i][i]

    pasos.append(f"Matriz triangular superior obtenida")
    pasos.append(f"Producto de pivotes ≈ {det}")
    pasos.append("Matriz ES invertible (rank = n)")

    return True, M, pasos, pivotes, det


def inversa_matriz(A):
    """Calcula la inversa de una matriz por Gauss-Jordan"""
    n = len(A)
    if len(A[0]) != n:
        raise ValueError("La matriz debe ser cuadrada")

    # Crear matriz aumentada [A | I]
    M = []
    for i in range(n):
        fila = A[i][:]  # Copiar fila de A
        # Añadir identidad
        fila.extend([Fraccion(1) if j == i else Fraccion(0) for j in range(n)])
        M.append(fila)

    pasos = []
    pasos.append(f"Cálculo de inversa de matriz {n}×{n}")
    pasos.append(f"Matriz aumentada [A|I]:\n{formatear_matriz(M)}")

    # Aplicar Gauss-Jordan
    for i in range(n):
        # Buscar pivote
        pivote = i
        while pivote < n and M[pivote][i].es_cero():
            pivote += 1

        if pivote == n:
            raise ValueError("Matriz no es invertible (determinante = 0)")

        if pivote != i:
            M[i], M[pivote] = M[pivote], M[i]
            pasos.append(f"Intercambio fila {i + 1} con fila {pivote + 1}")

        # Normalizar fila pivote
        pivote_val = M[i][i]
        if pivote_val != Fraccion(1):
            for j in range(2 * n):
                M[i][j] = M[i][j] / pivote_val
            pasos.append(f"Normalizar fila {i + 1} dividiendo por {pivote_val}")

        # Eliminar en otras filas
        for k in range(n):
            if k != i and not M[k][i].es_cero():
                factor = M[k][i]
                for j in range(2 * n):
                    M[k][j] = M[k][j] - factor * M[i][j]
                pasos.append(f"Eliminar elemento ({k + 1},{i + 1}) usando factor {factor}")

        pasos.append(f"Después de procesar columna {i + 1}:\n{formatear_matriz(M)}")

    # Extraer inversa
    inversa = []
    for i in range(n):
        inversa.append(M[i][n:])

    pasos.append("Matriz inversa encontrada:")
    pasos.append(formatear_matriz(inversa))

    return inversa, pasos