import math
from fraccion import Fraccion


def biseccion(f, a, b, tol=1e-6, max_iter=100, usar_error="absoluto"):
    """
    Método de bisección para encontrar raíces de f(x) = 0 en [a, b]

    Args:
        f: función continua
        a, b: extremos del intervalo
        tol: tolerancia
        max_iter: máximo número de iteraciones
        usar_error: "absoluto" o "relativo"

    Returns:
        raiz: aproximación de la raíz
        pasos: lista de diccionarios con información de cada iteración
        motivo: razón de terminación
    """

    # Validaciones iniciales
    if a >= b:
        raise ValueError("El intervalo debe cumplir a < b")

    fa = f(a)
    fb = f(b)

    if fa * fb > 0:
        raise ValueError("f(a) y f(b) deben tener signos opuestos (Teorema de Bolzano)")

    pasos = []
    iter_count = 0

    # Verificar si los extremos son raíces
    if abs(fa) < tol:
        return a, [{"k": 0, "a": a, "b": b, "c": a, "fa": fa, "fb": fb, "fc": fa, "error": 0.0}], "Raíz en extremo a"

    if abs(fb) < tol:
        return b, [{"k": 0, "a": a, "b": b, "c": b, "fa": fa, "fb": fb, "fc": fb, "error": 0.0}], "Raíz en extremo b"

    # Iteraciones de bisección
    c_prev = a
    for k in range(max_iter):
        iter_count = k + 1
        c = (a + b) / 2
        fc = f(c)

        # Calcular error
        if usar_error == "absoluto":
            error = abs(c - c_prev) if k > 0 else float('inf')
        else:  # relativo
            error = abs((c - c_prev) / c) if k > 0 and c != 0 else float('inf')

        # Guardar paso
        paso = {
            "k": iter_count,
            "a": a,
            "b": b,
            "c": c,
            "fa": fa,
            "fb": fb,
            "fc": fc,
            "error": error
        }
        pasos.append(paso)

        # Verificar convergencia
        if abs(fc) < tol:
            return c, pasos, f"Convergencia por |f(c)| < {tol}"

        if k > 0 and error < tol:
            return c, pasos, f"Convergencia por error < {tol}"

        # Actualizar intervalo
        if fa * fc < 0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc

        c_prev = c

    # Si llegamos aquí, máximo de iteraciones
    return c, pasos, f"Máximo de iteraciones ({max_iter}) alcanzado"


def evaluar_funcion(func_str, x):
    """
    Evalúa una función matemática en un punto x
    Soporta: +, -, *, /, **, sin, cos, tan, exp, log, sqrt, etc.
    """
    # Crear namespace seguro
    namespace = {
        'math': math,
        'x': x,
        'sin': math.sin,
        'cos': math.cos,
        'tan': math.tan,
        'exp': math.exp,
        'log': math.log,
        'log10': math.log10,
        'sqrt': math.sqrt,
        'pi': math.pi,
        'e': math.e
    }

    try:
        result = eval(func_str, {"__builtins__": {}}, namespace)
        return float(result)
    except Exception as e:
        raise ValueError(f"Error evaluando f({x}): {str(e)}")


def verificar_continuidad(f, a, b, puntos=1000):
    """
    Verifica aproximadamente la continuidad en [a, b]
    """
    import numpy as np

    x_vals = np.linspace(a, b, puntos)
    try:
        for x in x_vals:
            f(x)
        return True, "Función parece continua en el intervalo"
    except Exception as e:
        return False, f"Posible discontinuidad: {str(e)}"