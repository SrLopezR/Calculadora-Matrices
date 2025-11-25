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
                f"C[{i + 1},{j + 1}] = A[{i + 1},{j + 1}] + B[{i + 1},{j + 1}] = "
                f"{A[i][j]} + {B[i][j]} = {valor}"
            )
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
                pasos.append(
                    f"  + A[{i + 1},{k + 1}]×B[{k + 1},{j + 1}] = "
                    f"{A[i][k]} × {B[k][j]} = {producto}"
                )
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
    except Exception:
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


def combinar_escalar_matrices(escalarA_str, A, escalarB_str, B, operador="+"):
    """
    Calcula una combinación lineal de matrices del tipo:
        C = escalarA * A  +/-  escalarB * B

    operador: "+" para escalarA*A + escalarB*B
              "-" para escalarA*A - escalarB*B
    Devuelve (matriz_resultado, pasos)
    """
    # Validar dimensiones
    if len(A) != len(B) or len(A[0]) != len(B[0]):
        raise ValueError("Las matrices A y B deben tener las mismas dimensiones")

    # Parsear escalares
    try:
        escalarA = Fraccion(escalarA_str)
    except Exception:
        raise ValueError(f"Escalar inválido para A: {escalarA_str}")

    try:
        escalarB = Fraccion(escalarB_str)
    except Exception:
        raise ValueError(f"Escalar inválido para B: {escalarB_str}")

    if operador not in ("+", "-"):
        raise ValueError("Operador inválido, use '+' o '-'")

    pasos = []
    resultado = []

    simbolo = "+" if operador == "+" else "-"
    pasos.append(f"Combinación lineal de matrices: {escalarA}·A {simbolo} {escalarB}·B")
    pasos.append(f"A = \n{formatear_matriz(A)}")
    pasos.append(f"B = \n{formatear_matriz(B)}")
    pasos.append("")

    for i in range(len(A)):
        fila_resultado = []
        for j in range(len(A[0])):
            prodA = escalarA * A[i][j]
            prodB = escalarB * B[i][j]

            if operador == "+":
                valor = prodA + prodB
                pasos.append(
                    f"C[{i+1},{j+1}] = {escalarA}·A[{i+1},{j+1}] + {escalarB}·B[{i+1},{j+1}] = "
                    f"{escalarA}·{A[i][j]} + {escalarB}·{B[i][j]} = {prodA} + {prodB} = {valor}"
                )
            else:
                valor = prodA - prodB
                pasos.append(
                    f"C[{i+1},{j+1}] = {escalarA}·A[{i+1},{j+1}] - {escalarB}·B[{i+1},{j+1}] = "
                    f"{escalarA}·{A[i][j]} - {escalarB}·{B[i][j]} = {prodA} - {prodB} = {valor}"
                )

            fila_resultado.append(valor)
        resultado.append(fila_resultado)
        pasos.append("")

    pasos.append("Resultado:")
    pasos.append(formatear_matriz(resultado))

    return resultado, pasos


# ====== Funciones extra para vectores y combinaciones lineales ======

def sumar_vectores(u, v):
    """Suma dos 'vectores' (matrices del mismo tamaño) con pasos detallados."""
    if len(u) != len(v) or len(u[0]) != len(v[0]):
        raise ValueError("Los vectores deben tener las mismas dimensiones")

    pasos = []
    resultado = []

    pasos.append("Suma de vectores (tratados como matrices):")
    pasos.append(f"u = \n{formatear_matriz(u)}")
    pasos.append(f"v = \n{formatear_matriz(v)}")
    pasos.append("")

    for i in range(len(u)):
        fila_resultado = []
        for j in range(len(u[0])):
            valor = u[i][j] + v[i][j]
            fila_resultado.append(valor)
            pasos.append(
                f"w[{i + 1},{j + 1}] = u[{i + 1},{j + 1}] + v[{i + 1},{j + 1}] = "
                f"{u[i][j]} + {v[i][j]} = {valor}"
            )
        resultado.append(fila_resultado)
        pasos.append("")

    pasos.append("Resultado de u + v:")
    pasos.append(formatear_matriz(resultado))

    return resultado, pasos


def multiplicar_matriz_vector(A, x):
    """Multiplica matriz A por 'vector' columna x con pasos detallados."""
    if len(A[0]) != len(x):
        raise ValueError("Columnas de A deben igualar filas del vector")

    pasos = []
    pasos.append("Multiplicación A·x:")
    pasos.append(f"A = \n{formatear_matriz(A)}")
    pasos.append(f"x = \n{formatear_matriz(x)}")
    pasos.append("")

    resultado = []
    for i in range(len(A)):
        suma = Fraccion(0)
        pasos.append(f"Fila {i+1}:")
        for j in range(len(A[0])):
            prod = A[i][j] * x[j][0]
            suma = suma + prod
            pasos.append(f"  {A[i][j]} × {x[j][0]} = {prod}")
        pasos.append(f"  → y[{i+1}] = {suma}")
        resultado.append([suma])
        pasos.append("")

    pasos.append("Resultado A·x:")
    pasos.append(formatear_matriz(resultado))

    return resultado, pasos


def Au_mas_Av(A, u, v):
    """Calcula Au + Av con pasos detallados."""
    Au, pasos1 = multiplicar_matriz_vector(A, u)
    Av, pasos2 = multiplicar_matriz_vector(A, v)
    suma, pasos3 = sumar_vectores(Au, Av)

    pasos = ["Operación: Au + Av", ""]
    pasos.append("Cálculo de Au:")
    pasos.extend(pasos1)
    pasos.append("")
    pasos.append("Cálculo de Av:")
    pasos.extend(pasos2)
    pasos.append("")
    pasos.append("Suma Au + Av:")
    pasos.extend(pasos3)

    return suma, pasos


def A_por_u_mas_v(A, u, v):
    """Calcula A(u+v) con pasos detallados."""
    suma_uv, pasos1 = sumar_vectores(u, v)
    R, pasos2 = multiplicar_matriz_vector(A, suma_uv)

    pasos = ["Operación: A(u + v)", ""]
    pasos.append("Paso 1: Calcular u + v")
    pasos.extend(pasos1)
    pasos.append("")
    pasos.append("Paso 2: Multiplicar A por (u+v)")
    pasos.extend(pasos2)

    return R, pasos


# ====== Resto de funciones tal como estaban ======

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
                    M[j][k] = M[j][k] - factor * M[i][i]

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

    pasos.append("Matriz triangular superior obtenida")
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
