# fraccion.py
# fraccion.py
class Fraccion:
    def __init__(self, num, den=1):
        if isinstance(num, str) and den == 1:
            f = parse_token(num)
            self.num = f.num
            self.den = f.den
            return
        if den == 0:
            raise ZeroDivisionError("Denominador no puede ser cero")
        if den < 0:
            num, den = -num, -den
        g = self.mcd(abs(int(num)), abs(int(den)))
        self.num = int(num) // g
        self.den = int(den) // g

    @staticmethod
    def mcd(a, b):
        while b:
            a, b = b, a % b
        return a

    def _coaccionar(self, v):
        if isinstance(v, Fraccion):
            return v
        if isinstance(v, str):
            return parse_token(v)
        return Fraccion(v)

    def __add__(self, otro):
        o = self._coaccionar(otro)
        return Fraccion(self.num * o.den + o.num * self.den, self.den * o.den)

    def __sub__(self, otro):
        o = self._coaccionar(otro)
        return Fraccion(self.num * o.den - o.num * self.den, self.den * o.den)

    def __mul__(self, otro):
        o = self._coaccionar(otro)
        return Fraccion(self.num * o.num, self.den * o.den)

    def __truediv__(self, otro):
        o = self._coaccionar(otro)
        return Fraccion(self.num * o.den, self.den * o.num)

    def __neg__(self):
        return Fraccion(-self.num, self.den)

    def __eq__(self, otro):
        o = self._coaccionar(otro)
        return self.num == o.num and self.den == o.den

    def __abs__(self):
        return Fraccion(abs(self.num), self.den)

    def es_cero(self):
        return self.num == 0

    def es_uno(self):
        return self.num == 1 and self.den == 1

    def __float__(self):
        return self.num / self.den

    def __str__(self):
        return str(self.num) if self.den == 1 else f"{self.num}/{self.den}"

def parse_token(t: str) -> Fraccion:
    t = t.strip()
    if '/' in t:
        a, b = t.split('/')
        return Fraccion(int(a), int(b))
    else:
        return Fraccion(int(t))