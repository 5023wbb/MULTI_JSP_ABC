class FoodSource(object):
    trials = 0

    """docstring for FoodSource"""
    def __init__(self, initial_solution, initial_fitness, opIDsOnMchs, g_list):
        super(FoodSource, self).__init__()

        self.solution = initial_solution
        self.fitness = initial_fitness
        self.opIDsOnMchs = opIDsOnMchs
        self.g_list = g_list

    def __repr__(self):
        return f'<FoodSource s:{self.solution} f:{self.fitness} />'
