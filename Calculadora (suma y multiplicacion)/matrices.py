# matrices.py
from fraccion import Fraccion, parse_token

def sumar_matrices(A, B):
    """Suma dos matrices de igual tamaño y devuelve (resultado, pasos)."""
    m, n = len(A), len(A[0])
    if len(B) != m or len(B[0]) != n:
        raise ValueError("Las matrices deben tener las mismas dimensiones para sumarse.")

    R = []
    pasos = []

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

    R = [[Fraccion(0) for _ in range(q)] for _ in range(m)]
    pasos = []

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
