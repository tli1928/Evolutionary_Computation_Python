"""
Ryan Huang, Anthony Gemma, Eric Wu, Teng li
DS3500 / Resource Allocation with Evolutionary Computing
HW4
3/28
"""

import random as rnd
import copy
from functools import reduce
import time as tim
import numpy as np
import pandas as pd

class Evo:

    def __init__(self):
        self.pop = {}  # ((ob1, eval1), (obj2, eval2), ...) ==> solution
        self.fitness = {}  # name -> objective func
        self.agents = {}  # name -> (agent operator, # input solutions)

    def size(self):
        """ The size of the solution population """
        return len(self.pop)

    def add_fitness_criteria(self, name, f):
        """ Registering an objective with the Evo framework
        name - The name of the objective (string)
        f    - The objective function:   f(solution)--> a number """
        self.fitness[name] = f

    def add_agent(self, name, op, k=1):
        """ Registering an agent with the Evo framework
        name - The name of the agent
        op   - The operator - the function carried out by the agent  op(*solutions)-> new solution
        k    - the number of input solutions (usually 1) """
        self.agents[name] = (op, k)

    def get_random_solutions(self, k=1):
        """ Pick k random solutions from the population as a list of solutions
            We are returning DEEP copies of these solutions as a list """
        if self.size() == 0:  # No solutions in the populations
            # return []
            return np.array((0,))

        else:
            popvals = tuple(self.pop.values())
            # return [copy.deepcopy(rnd.choice(popvals)) for  in range(k)]
            return np.array([copy.deepcopy(rnd.choice(popvals)) for _ in range(k)])

    def add_solution(self, sol):
        """Add a new solution to the population """
        eval = tuple([(name, f(sol)) for name, f in self.fitness.items()])
        self.pop[eval] = sol

    def run_agent(self, name):
        """ Invoke an agent against the current population """
        op, k = self.agents[name]
        picks = self.get_random_solutions(k)
        new_solution = op(picks)
        self.add_solution(new_solution)

    def evolve(self, n=1, dom=100, status=100, time=600):
        """ To run n random agents against the population
        n - # of agent invocations
        dom - # of iterations between discarding the dominated solutions """

        agent_names = list(self.agents.keys())
        start = tim.time()

        while (tim.time() - start) <= time:
            for i in range(n):
                # print('i', i)
                # print(tim.time() - start)
                pick = rnd.choice(agent_names)  # pick an agent to run
                self.run_agent(pick)
                if i % dom == 0:
                    self.remove_dominated()

                if i % status == 0:  # print the population
                    self.remove_dominated()
                    print("Iteration: ", i)
                    print("Population Size: ", self.size())
                    # print(self)

                if (tim.time() - start) >= time:
                    # Check if elapsed time is close enough to specified time to terminate loop
                    print("Time's up!")
                    break

            # Clean up population
            self.remove_dominated()

        print("Algorithm finished.")

    @staticmethod
    def _dominates(p, q):
        pscores = [score for _, score in p]
        qscores = [score for _, score in q]
        score_diffs = list(map(lambda x, y: y - x, pscores, qscores))
        min_diff = min(score_diffs)
        max_diff = max(score_diffs)
        return min_diff >= 0.0 and max_diff > 0.0

    @staticmethod
    def _reduce_nds(S, p):
        return S - {q for q in S if Evo._dominates(p, q)}

    def remove_dominated(self):
        nds = reduce(Evo._reduce_nds, self.pop.keys(), self.pop.keys())
        self.pop = {k: self.pop[k] for k in nds}

    def __str__(self):
        """ Output the solutions in the population """
        rslt = ""
        for eval, sol in self.pop.items():
            rslt += str(dict(eval)) + ":\t" + str(sol) + "\n"
        return rslt

    def summarize(self, filename="results.csv"):
        """ Summarize the solutions in the population as a pandas DataFrame and save it to a CSV file """
        # create a new empty pandas DataFrame with columns for GroupName and all the fitness function keys
        df = pd.DataFrame(columns=["GroupName"] + list(self.fitness.keys()))

        # iterate through all evaluations and solutions in the population
        for eval, sol in self.pop.items():
            # extract the scores from the evaluation and add a row to the DataFrame with group name and scores
            scores = [score for _, score in eval]
            df = df.append(pd.Series(['group11'] + scores, index=df.columns), ignore_index=True)

        # save the DataFrame to a CSV file with the given filename
        df.to_csv(filename, index=False)

    def find_best_solution(self):
        """ Finds the best solution based on the fitness functions."""
        # initialize variables to hold the best evaluation and solution found so far
        best_eval = None
        best_sol = None

        # iterate through all evaluations and solutions in the population
        for eval, sol in self.pop.items():
            # if this is the first evaluation or all fitness scores are better than the previous best,
            # update the best solution
            if best_eval is None or all(eval[i][1] < best_eval[i][1] for i in range(len(eval))):
                best_eval = eval
                best_sol = sol

        # return a dictionary containing the best evaluation and the best solution
        return dict(best_eval), best_sol
