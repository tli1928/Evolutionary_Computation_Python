"""
Ryan Huang, Anthony Gemma, Eric Wu, Teng li
DS3500 / Resource Allocation with Evolutionary Computing
HW4
3/28
"""

from objectives import *
import numpy as np
from evo import Evo


def flatten(array):
    """ function that flattens an n-dimensional numpy array to a 2-dimensional numpy array

    Args:
        array (numpy array): an n-D Numpy Array

    Return:
        2-dimensional numpy array
    """
    # repeatedly index the first element of the input array until it is 3-dimensional
    while len(array.shape) != 3:
        array = array[0]

    # return the 2-dimensional array
    return array


def min_unpref(sol, n=25):
    """ function where n randomly selected TAs are given their exact working preferences

    Args:
        sol (2d-numpy array): current solution being tested in Evo
        n (int): number of TAs to switch

    Returns:
         2d-numpy array of a modified solution
    """
    # choose n random TAs to give their exact working preferences
    rand_tas = np.random.choice(np.array(range(43)), size=n, replace=False)

    # convert the TA data into a numpy array where their preferences are 1 and everything else is 0
    uw_tas = tas_data.drop(['ta_id', 'name', 'max_assigned'], axis=1).to_numpy()
    uw_tas[uw_tas == 'P'], uw_tas[uw_tas != 1] = 1, 0

    # flatten the input solution matrix to 2 dimensions
    sol = flatten(sol)

    # for each of the n selected TAs, assign them to their preferred sections in the modified solution
    for i in range(n):
        sol[0][rand_tas[i]] = uw_tas[rand_tas[i]]

    return sol


def min_undersupport(sol):
    """ function for a modified solution where all sections are given their exact minimum TA requirements

    Args:
        sol (2d-numpy array): current solution being tested in Evo

    Return:
        2d-numpy array of a modified solution
    """
    # minimum TA requirements per section as a numpy array
    min_ta = np.array(section_data['min_ta'])

    # flatten the input solution matrix to 2 dimensions
    sol = flatten(sol)

    # for each section, check if it is undersupported
    for col in range(17):
        # if it is undersupported, give that section the minimum number of TAs (TA chosen randomly)
        if np.sum(np.squeeze(sol)[:,col]) < min_ta[col]:
            n = min_ta[col] - np.sum(np.squeeze(sol)[:,col])
            rand_tas = np.random.choice(np.array(range(43)), size=n, replace=False)
            sol[:, rand_tas] = 1

    return sol


def min_overallocation(sol):
    """ function that removes section assignments from TAs until they are no longer overallocated

    Args:
        sol (2d-numpy array): current solution being tested in Evo

    Return:
        2d-numpy array of a modified solution
    """

    # maximum number of assignments per TA as an numpy array
    max_assign = np.array(tas_data['max_assigned'])

    # flatten the input solution matrix to 2 dimensions
    sol = flatten(sol)

    # for each TA, check if they are overallocated
    for ta in range(43):
        # if they are overallocated, choose random sections to remove from their assignments until
        # they are no longer overallocated
        if np.sum(np.squeeze(sol)[ta]) > max_assign[ta]:
            n = np.sum(np.squeeze(sol)[ta]) - max_assign[ta]
            rand_sections = np.random.choice(np.array(range(17)), size=n, replace=False)
            np.squeeze(sol)[ta][rand_sections] = 0

    return sol


def compress(sol):
    """ function that compresses to read 2D Numpy array format to a readable dictionary format

    Args:
        sol (2d-numpy array): current solution being tested in Evo

    Return:
        best_sol_dict (dict): dictionary with TA # as the key and their section assignments as the value
    """
    # create an empty dictionary named best_sol_dict
    best_sol_dict = {}

    # flatten the input solution matrix to 2 dimensions
    sol = flatten(sol)

    # iterate over each TA
    for ta in range(43):
        # check if the TA has no section assignments
        if np.where(np.squeeze(sol)[ta])[0].tolist() == []:
            # if the TA has no section assignments, set their value in the dictionary to "TA is not working"
            best_sol_dict[ta] = "TA is not working"
        else:
            # if the TA has section assignments, add their section assignments to the dictionary
            # the section assignments are added as a list of integers
            best_sol_dict[ta] = np.where(np.squeeze(sol)[ta])[0].tolist()

    return best_sol_dict


def main():

    # initialize the Evolver
    E = Evo()

    # add all the five objective functions
    E.add_fitness_criteria('Overallocation', overallocation)
    E.add_fitness_criteria('Conflicts', conflicts)
    E.add_fitness_criteria('Undersupport', undersupport)
    E.add_fitness_criteria('Unwilling', unwilling)
    E.add_fitness_criteria('Unpreferred', unpreferred)

    # register some (3) agents
    E.add_agent("min_unpref", min_unpref, k=1)
    E.add_agent("min_undersupport", min_undersupport, k=1)
    E.add_agent("min_overallocation", min_overallocation, k=1)

    # Seed the population with an initial random solution
    seed = np.random.randint(2, size=(43, 17))
    E.add_solution(seed)

    # run the evolver
    E.evolve(10000000, 10, 100, 600)

    # checking for the best results
    best_eval, best_sol = E.find_best_solution()

    # communicate best evaluation score
    print("Best Evaluation Score:", best_eval)
    print(best_sol)

    # communicate the best solution details
    best_sol_details = compress(best_sol)
    print(best_sol_details)


if __name__ == "__main__":
    main()
