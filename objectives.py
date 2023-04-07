"""
Ryan Huang, Anthony Gemma, Eric Wu, Teng li
DS3500 / Resource Allocation with Evolutionary Computing
HW4
3/28
"""

import numpy as np
import pandas as pd

# Section and TA files
section_data = pd.read_csv('sections.csv')
tas_data = pd.read_csv('tas.csv')

# Relevant columns for overallocation and undersupport
max_assign = np.array(tas_data['max_assigned'])
min_assign = np.array(section_data['min_ta'])


def overallocation(sol):
    """
    :param sol: Solution as a 2D Numpy array
    :param max_ta: 1D numpy array with maximum assignments
    :return Integer representing overallocations
    """
    max_ta = max_assign
    sol = np.squeeze(np.array(sol))

    # print('Sol', sol)

    return np.sum(np.maximum(np.subtract(np.sum(sol, axis=1), max_ta), 0))

def conflicts(sol):
    """
    :param sol: Solution as a 2D Numpy array
    :param sections: Dataframe of sections
    :return:
    """
    sections = section_data
    # Get the day and time for each section
    sections['day'], sections['time'] = sections['daytime'].str.extract(r'(\w)\s'), sections['daytime'].str.extract(r'\s(.*)')

    # Create a conflict matrix to represent the time conflicts between sections
    conflict_matrix = sections[['day', 'time']].apply(
        lambda x: sections['day'].eq(x['day']) & sections['time'].eq(x['time']), axis=1).values

    # Calculate the number of conflicts for each TA
    ta_conflict_matrix = np.dot(sol, conflict_matrix)
    ta_conflict_count = np.sum((ta_conflict_matrix >= 2).astype(int), axis=1)

    # Return the total number of TAs with one or more time conflicts
    return np.sum(ta_conflict_count >= 1)


def undersupport(sol):
    """
    :param sol: Solution as a 2D Numpy array
    :param min_assign: 1D numpy array with maximum assignments
    :return: Integer representing undersupported lab sections
    """
    min_ta = min_assign
    return np.sum(np.maximum(np.subtract(min_ta, np.sum(np.squeeze(sol), axis=0)), 0))


def unwilling(sol):
    """
    :param sol: Solution as a 2D Numpy array
    :param tas: Dataframe of tas.csv containing the unwilling sections
    :return: Integer representing the number of unwilling TAs
    """
    tas = tas_data
    uw_tas = tas.drop(['ta_id', 'name', 'max_assigned'], axis=1).to_numpy()
    uw_tas[uw_tas == 'U'], uw_tas[uw_tas != 1] = 1, 0
    return (np.add(uw_tas, sol) == 2).sum()


def unpreferred(sol):
    """
    :param sol: Solution as a 2D Numpy array
    :param tas: Dataframe of tas.csv containing the unwilling sections
    :return: Integer representing the number of unpreferring TAs
    """
    tas = tas_data
    uw_tas = tas.drop(['ta_id', 'name', 'max_assigned'], axis=1).to_numpy()
    uw_tas[uw_tas == 'W'], uw_tas[uw_tas != 1] = 1, 0
    return (np.add(uw_tas, sol) == 2).sum()














