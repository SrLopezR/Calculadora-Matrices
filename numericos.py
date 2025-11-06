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


def encontrar_intervalo_automatico(f, centro=0, rango_inicial=10, max_intentos=20):
    """
     Búsqueda automática de un intervalo [a, b] donde f(a)*f(b) < 0
    - Muestrea puntos en un rango creciente alrededor de `centro`
    - Si encuentra un par consecutivo con cambio de signo devuelve ese par (intervalo ajustado)
    - Si encuentra f(x)==0 devuelve un intervalo pequeño alrededor del punto y mensaje claro
    - Si no encuentra, devuelve (-10, 10) con mensaje indicando fallback
    """
    import numpy as np

    # Parámetros de muestreo
    puntos_inicial = 201   # muestreo denso para detectar cambios de signo
    for intento in range(max_intentos):
        rango_actual = rango_inicial * (2 ** intento)
        a = centro - rango_actual
        b = centro + rango_actual

        try:
            # muestrear puntos de a a b
            xs = np.linspace(a, b, puntos_inicial)
            fs = []
            for x in xs:
                try:
                    fs.append(float(f(x)))
                except Exception:
                    fs.append(float('nan'))

            # inspeccionar valores buscando cambio de signo entre puntos consecutivos
            for i in range(len(xs) - 1):
                xa, xb = xs[i], xs[i+1]
                fa, fb = fs[i], fs[i+1]

                # si alguno es NaN, saltar
                if fa != fa or fb != fb:
                    continue

                # si f(x) == 0 exactamente
                if abs(fa) == 0.0:
                    # devolver pequeño intervalo alrededor de xa
                    return xa - 1e-3, xa + 1e-3, f"Raíz encontrada exactamente en x = {xa:.6g}"
                if abs(fb) == 0.0:
                    return xb - 1e-3, xb + 1e-3, f"Raíz encontrada exactamente en x = {xb:.6g}"

                # cambio de signo entre fa y fb
                if fa * fb < 0:
                    # devolver intervalo ajustado [xa, xb]
                    return xa, xb, f"Intervalo encontrado: [{xa:.6g}, {xb:.6g}] (muestreo en intento {intento+1})"

            # si no hubo cambio de signo, repetir con rango mayor
        except Exception:
            # intentar siguiente intento
            continue

    # fallback si no se encontró intervalo útil
    return -10.0, 10.0, "No se encontró intervalo con cambio de signo. Usando fallback [-10, 10]."