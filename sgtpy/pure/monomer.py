
def amono(ahs, a1m, a2m, a3m, beta, ms):
    am = ms * (ahs + beta * a1m + beta**2 * a2m + beta**3 * a3m)
    return am


def damono_drho(ahs, a1m, a2m, a3m, beta, drho, ms):
    am = ms * drho * (ahs + beta * a1m + beta**2 * a2m + beta**3 * a3m)
    return am


def d2amono_drho(ahs, a1m, a2m, a3m, beta, drho, ms):

    am = ms * drho * (ahs + beta * a1m + beta**2 * a2m + beta**3 * a3m)
    return am
