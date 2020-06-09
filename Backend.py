import pandas as pd
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from scipy import optimize

# Assumptions
# Birth rate of the country is equal to the natural death rate, i.e. mu=nu
# Incubation period of 5 days
# The population of a country does not change significantly


class Country:
    """
    A class to contain the information about a country's model and interact with other Country objects to create a dynamic worldwide model for infection.

    Attributes:
        data(pd.DataFrame): A dataframe containing the infections, recoveries and deaths over time
        N(int): The population of the country
        mu(float): The birth rate of the country as a decimal
        nu(float): The natural death rate of the country as a decimal
    """
    def __init__(self, filepath: str, N: int, mu: float) -> None:
        """
        Constructor for the country class
        
        Arguments:
            filepath(str): The filepath of the data
            N(int): The population of the country
            mu(float): The birth rate of the country as a decimal
            nu(float): The natural death rate of the country as a decimal
        """
        self.data = pd.read_csv(filepath)
        self.N = N
        self.mu = mu

    def __deriv(self, y: np.ndarray, (im, em, t), beta: float, gamma: float, delta: float, epsilon: float) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray):
        S, E, I, R, D = y
        S_m, E_m, I_m, R_m = im[t] - em[t]
        theta = 1/5 # Because 5 is incubation period
        dSdt = - beta*S*I/self.N + epsilon*R + self.mu*(self.N - S) + S_m
        dEdt = beta*S*I/self.N - theta*E - self.mu*E + E_m
        dIdt = theta*E - gamma*I - delta*I - self.mu*I + I_m
        dRdt = gamma*I - epsilon*R - self.mu*R + R_m
        dDdt = delta*I
        return dSdt, dEdt, dIdt, dRdt, dDdt
    
    def __odesolve(self, t: np.ndarray, beta: float, gamma: float, delta: float, epsilon: float) -> np.ndarray:
        return odeint(self.__deriv, y0=(self.N-1, 0, 1, 0, 0), t=(im, em, t), args=(beta, gamma, delta, epsilon))

    def __cost(self, start: int, end: int) -> "func":
        return lambda params: mean_squared_error(self.__odesolve(range(start, end), params[0], params[1], params[2], params[3])[:,2:], self.data[start:end])

    def fit(self, start: int, end: int) -> (float, float, float, float):
        result = optimize.minimize(self.__cost(start, end), x0=[1, 1, 1, 1], method="L-BFGS-B", bounds=[(0,np.inf), (0, np.inf), (0, np.inf), (0, np.inf)])
        return result.x

class Migration:
    """
    Class to contain migration data between two countries

    Attributes:
        immigration: immigration from country1 to country2
    """
    def __init__(self, filepath: str, country1: Country, country2: Country) -> None:
        """
        Constructor

        Parameters:
            data
        """
        data = pd.read_csv(filepath)
        self.forward = data["Forward"]
        self.backward = data["Backward"]
        self.country1 = country1
        self.country2 = country2

class World:
    def __init__(self, country_list: list(Country), migration: list(Migration)) -> None:
        """
        Constructor for the World class

        Parameters:
            country_list: List of countries to evaluate the model for. Note that rest_of_world should be one of these countries if not all countries are listed.
            migration: List of migration data, this should be given as pairs for all pairs in the countries data (where data is not given, migration will be assumed to be 0 between those countries).
        """
        self.countries = country_list

    def odesolve(self, t: np.ndarray