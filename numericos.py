# numericos.py
import math

# ---------------------------
# Parser seguro para f(x)
# ---------------------------
_ALLOWED = {
    # constantes
    "pi": math.pi, "e": math.e,
    # funciones comunes
    "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "asin": math.asin, "acos": math.acos, "atan": math.atan,
    "sinh": math.sinh, "cosh": math.cosh, "tanh": math.tanh,
    "exp": math.exp, "log": math.log, "log10": math.log10,
    "sqrt": math.sqrt, "abs": abs, "ln": math.log,
}

class FuncionInvalida(Exception):
    pass

def parse_function(src: str):
    """
    Devuelve una función f(x) evaluable de forma segura.
    Uso: f = parse_function("x**3 - x - 2"); y = f(1.5)
    """
    if not isinstance(src, str) or not src.strip():
        raise FuncionInvalida("Ingresa una expresión para f(x).")
    code = compile(src, "<f>", "eval")
    # bloqueamos nombres no permitidos
    def f(x):
        env = {"x": float(x)}
        return float(eval(code, {"__builtins__": {}}, {**_ALLOWED, **env}))
    # prueba rápida
    try:
        _ = f(0.0)
    except Exception as e:
        raise FuncionInvalida(f"Expresión inválida para f(x): {e}")
    return f

# ---------------------------
# Errores numéricos
# ---------------------------
def error_absoluto(actual, anterior):
    return abs(actual - anterior)

def error_relativo(actual, anterior):
    ea = error_absoluto(actual, anterior)
    denom = abs(actual) if actual != 0 else 1.0
    return ea / denom

# ---------------------------
# Bisección
# ---------------------------
class IntervaloInvalido(Exception):
    pass

def biseccion(f, a, b, tol=1e-4, max_iter=200, usar_error="absoluto"):
    """
    Devuelve (raiz_aprox, pasos:list[dict], motivo_paro:str)
    pasos: [{'k':1,'a':..,'b':..,'c':..,'fa':..,'fb':..,'fc':..,'error':..}, ...]
    usar_error: 'absoluto' | 'relativo'
    """
    a = float(a); b = float(b)
    if a >= b:
        raise IntervaloInvalido("Se requiere a < b.")
    fa, fb = f(a), f(b)
    if fa == 0:  # por si justo cae en extremo
        return a, [{"k":0,"a":a,"b":b,"c":a,"fa":fa,"fb":fb,"fc":fa,"error":0.0}], "f(a)=0"
    if fb == 0:
        return b, [{"k":0,"a":a,"b":b,"c":b,"fa":fa,"fb":fb,"fc":fb,"error":0.0}], "f(b)=0"
    if fa * fb > 0:
        raise IntervaloInvalido("f(a) y f(b) deben tener signos opuestos (teorema del valor intermedio).")

    pasos = []
    c_prev = None
    motivo = "tol alcanzada"
    for k in range(1, max_iter+1):
        c = (a + b) / 2.0
        fc = f(c)
        # error
        if c_prev is None:
            err = float('nan')
        else:
            err = error_absoluto(c, c_prev) if usar_error == "absoluto" else error_relativo(c, c_prev)

        pasos.append({
            "k": k, "a": a, "b": b, "c": c,
            "fa": fa, "fb": fb, "fc": fc,
            "error": err
        })

        # criterios de paro
        if abs(fc) <= tol:
            motivo = "|f(c)| ≤ tol"
            break
        if c_prev is not None and err <= tol:
            motivo = "error ≤ tol"
            break

        # actualización de intervalo
        if fa * fb > 0:
            a, fa = c, fc
        else:
            b, fb = c, fc

        c_prev = c

    return c, pasos, motivo
