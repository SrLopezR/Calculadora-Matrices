import math
class Fraccion:
    def __init__(self, numerador, denominador=1):
        if isinstance(numerador, str):
            if '/' in numerador:
                parts = numerador.split('/')
                if len(parts) == 2:
                    num = int(parts[0])
                    den = int(parts[1])
                else:
                    raise ValueError("Formato de fracción inválido")
            else:
                num = int(numerador)
                den = denominador
        else:
            num = numerador
            den = denominador

        if den == 0:
            raise ZeroDivisionError("Denominador no puede ser cero")

        # Simplificar
        gcd_val = math.gcd(abs(num), abs(den))
        self.numerador = num // gcd_val
        self.denominador = den // gcd_val

        # Asegurar que el denominador sea positivo
        if self.denominador < 0:
            self.numerador = -self.numerador
            self.denominador = -self.denominador

    def __add__(self, other):
        if isinstance(other, int):
            other = Fraccion(other)
        num = self.numerador * other.denominador + other.numerador * self.denominador
        den = self.denominador * other.denominador
        return Fraccion(num, den)

    def __sub__(self, other):
        if isinstance(other, int):
            other = Fraccion(other)
        num = self.numerador * other.denominador - other.numerador * self.denominador
        den = self.denominador * other.denominador
        return Fraccion(num, den)

    def __mul__(self, other):
        if isinstance(other, int):
            other = Fraccion(other)
        num = self.numerador * other.numerador
        den = self.denominador * other.denominador
        return Fraccion(num, den)

    def __truediv__(self, other):
        if isinstance(other, int):
            other = Fraccion(other)
        if other.numerador == 0:
            raise ZeroDivisionError("División por cero")
        num = self.numerador * other.denominador
        den = self.denominador * other.numerador
        return Fraccion(num, den)

    def __eq__(self, other):
        if isinstance(other, int):
            other = Fraccion(other)
        return (self.numerador * other.denominador ==
                other.numerador * self.denominador)

    def __lt__(self, other):
        if isinstance(other, int):
            other = Fraccion(other)
        return (self.numerador * other.denominador <
                other.numerador * self.denominador)

    def __le__(self, other):
        if isinstance(other, int):
            other = Fraccion(other)
        return (self.numerador * other.denominador <=
                other.numerador * self.denominador)

    def __gt__(self, other):
        return not self.__le__(other)

    def __ge__(self, other):
        return not self.__lt__(other)

    def __neg__(self):
        return Fraccion(-self.numerador, self.denominador)

    def __abs__(self):
        return Fraccion(abs(self.numerador), self.denominador)

    def __float__(self):
        return self.numerador / self.denominador

    def __str__(self):
        if self.denominador == 1:
            return str(self.numerador)
        else:
            return f"{self.numerador}/{self.denominador}"

    def __repr__(self):
        return f"Fraccion({self.numerador}, {self.denominador})"

    def es_cero(self):
        return self.numerador == 0

    def reciproco(self):
        return Fraccion(self.denominador, self.numerador)