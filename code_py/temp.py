

def calculer_indice_de_chaleur(T, phi, unite='C'):
    """
    Calcule l'indice de chaleur (Heat Index) en fonction de la température et de l'humidité relative.

    :param T: Température (en °C ou °F).
    :param phi: Humidité relative de l'air en %.
    :param unite: Unité de température ('C' pour Celsius, 'F' pour Fahrenheit).
    :return: Indice de chaleur (ressentie) calculé.
    """
    if unite == 'C':
        # Coefficients pour Celsius
        c1, c2, c3, c4 = -8.785, 1.611, 2.339, -0.146
        c5, c6, c7 = -1.231e-2, -1.642e-2, 2.212e-3
        c8, c9 = 7.255e-4, -3.582e-6
    elif unite == 'F':
        # Coefficients pour Fahrenheit
        c1, c2, c3, c4 = -42.379, 2.049, 10.143, -0.225
        c5, c6, c7 = -6.838e-3, -5.482e-2, 1.229e-3
        c8, c9 = 8.528e-4, -1.990e-6
    else:
        raise ValueError("Unité non supportée : veuillez utiliser 'C' pour Celsius ou 'F' pour Fahrenheit.")

    # Calcul de l'indice de chaleur
    HI = (c1 +
          c2 * T +
          c3 * phi +
          c4 * T * phi +
          c5 * T**2 +
          c6 * phi**2 +
          c7 * T**2 * phi +
          c8 * T * phi**2 +
          c9 * T**2 * phi**2)

    return round(HI, 2)



