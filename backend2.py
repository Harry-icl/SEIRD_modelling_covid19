import pandas as pd
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from scipy import optimize
import plotly.express as px
import sys

class World:
    def __init__(self, time) -> None:
        self.time = time
        self.num_countries = 0
        self.countries = np.ndarray(0)
        self.countries_data = np.ndarray((time, 0))
        self.N = np.ndarray(0)
        self.mu = np.ndarray(0)


    def add_country(self, country: str, filepath: str, N: int, mu: float) -> None:
        self.num_countries += 1
        self.countries = np.append(self.countries, [country])
        self.countries_data = np.append(self.countries_data, pd.read_csv(filepath), axis=1)
        self.N = np.append(self.N, [N])
        self.mu = np.append(self.mu, [mu])


    def country_list_complete(self) -> None:
        self.immigration_array = np.array([[np.zeros(self.num_countries)]*self.num_countries]*self.time)
        self.emigration_array = np.array([[np.zeros(self.num_countries)]*self.num_countries]*self.time)


    def add_migration(self, country_from: str, country_to: str, filepath: str) -> None:
        data = pd.read_csv(filepath)
        forward = data["Forward"]
        reverse = data["Backward"]

        country_from_index = np.where(self.countries == country_from)[0][0]
        country_to_index = np.where(self.countries == country_to)[0][0]

        from_im_prop = reverse/self.N[country_from_index]
        to_im_prop = forward/self.N[country_to_index]
        from_em_prop = forward/self.N[country_from_index]
        to_em_prop = reverse/self.N[country_to_index]

        for t in range(self.time):
            self.immigration_array[t][country_to_index][country_from_index] = from_im_prop[t]
            self.immigration_array[t][country_from_index][country_to_index] = to_im_prop[t]
            self.emigration_array[t][country_from_index][country_from_index] += from_em_prop[t]
            self.emigration_array[t][country_to_index][country_to_index] += to_em_prop[t]
    

    def __deriv(self, y: np.ndarray, t: float, args) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray):
        S, E, I, R, D = np.split(y, 5) # numpy doesn't accept 2D input so this is the simplest workaround
        beta, gamma, delta, epsilon = np.split(args, 4)
        theta = 1/5 # 5 day incubation period
        t = int(t) if t < 136 else 136
        dSdt = - beta*S*I/self.N + epsilon*R + self.mu*(self.N - S) + S@self.immigration_array[int(t)] - S@self.emigration_array[int(t)]
        dEdt = beta*S*I/self.N - theta*E - self.mu*E + S@self.immigration_array[int(t)] - S@self.emigration_array[int(t)]
        dIdt = theta*E - gamma*I - delta*I - self.mu*I + S@self.immigration_array[int(t)] - S@self.emigration_array[int(t)]
        dRdt = gamma*I - epsilon*R - self.mu*R + S@self.immigration_array[int(t)] - S@self.emigration_array[int(t)]
        dDdt = delta*I
        deriv = np.concatenate((dSdt, dEdt, dIdt, dRdt, dDdt)).flatten()
        return deriv

    
    def odesolve(self, y0, t, args):
        ode = odeint(self.__deriv, y0=y0, t=t, args=(args,)).T
        ode_dat = np.array([ode[i*5+2:i*5+5] for i in range(self.num_countries)])
        ode_dat = ode_dat.reshape((len(ode_dat[0])*self.num_countries, len(ode_dat[0][0]))).T
        return ode_dat


    def __cost(self, start, end, y0) -> "func":
        return lambda args: mean_squared_error(self.odesolve(y0, range(start, end), args), self.countries_data)
    

    def fit(self, start, end, y0) -> np.ndarray:
        x0 = np.ones(self.num_countries*4)
        bounds = np.array([(0, np.inf)]*self.num_countries*4)
        return optimize.minimize(self.__cost(start, end, y0), x0=x0, method="L-BFGS-B", bounds=bounds)

world = World(137)
world.add_country("gb", "Data/gb.csv", 66_650_000, 0.000027)
world.add_country("fr", "Data/fr.csv", 66_990_000, 0.000029)

world.country_list_complete()

world.add_migration("gb", "fr", "Data/gb-fr.csv")

y0 = np.array(np.concatenate(([world.N[i] for i in range(world.num_countries - 1)], [world.N[-1] - 1], [0]*world.num_countries, [0]*(world.num_countries - 1), [1], [0]*world.num_countries, [0]*world.num_countries)))

result = world.fit(0, len(world.countries_data), y0)

y = world.odesolve(y0, range(world.time), result.x)
plt.plot(x=range(137), y=y)
plt.show()