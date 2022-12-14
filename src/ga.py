import random


class GADiophant:
    survive_coef: int = 0.3
    productivity: int = 4

    def __init__(
        self,
        coefficients: list[int],
        answer: int,
        bounds=None,
        steps=100,
        stop_fitness=None,
        population_limit=25,
    ):
        self.coefficients = coefficients
        self.answer = answer
        self.steps = steps
        self.stop_fitness = stop_fitness
        self.population_limit = population_limit
        self.bounds = [(bounds[0], bounds[1], bounds[2])] * len(self.coefficients)
        self.best = []

    def evolve(self):
        newborns = []
        best = None
        for _ in range(self.steps):
            population = self.__generate_population(newborns)
            survivors = sorted(population, key=lambda i: -i[1])[: int(self.population_limit * self.survive_coef)]
            newborns = self.__cross(survivors)

            self.best.append(survivors[0])
            if not best:
                best = self.best[-1]
            else:
                best = max(best, self.best[-1], key=lambda i: i[1])

            if self.stop_fitness != None and best[-1] >= self.stop_fitness:
                break

        return best

    def __diophant(self, individs):
        ans = self.answer
        for i, coef in enumerate(self.coefficients):
            ans -= coef * individs[i]
        return -abs(ans)

    def __generate_population(self, newborns):
        population = []
        for individ in newborns:
            individ = self.__mutation(individ)
            fitness = self.__diophant(individ)
            population.append((individ, fitness))

        for _ in range(self.population_limit - len(newborns)):
            individs = []
            for bound in self.bounds:
                step = bound[2]
                gene = random.choice(self.__frange(bound[0], bound[1] + step, step))
                individs.append(gene)

            fitness = self.__diophant(individs)
            population.append((individs, fitness))
            newborns.append(individs)
        return population

    def __cross(self, best):
        newborns = []
        for _ in range(len(best) * self.productivity):
            dad, mom = random.sample(best, 2)
            dad, mom = dad[0], mom[0]
            split = len(dad) // 2
            child = dad[:split] + mom[split:]
            newborns.append(child)
        return newborns

    def __mutation(self, indiv):
        gene_id = random.randint(0, len(indiv) - 1)
        step = self.bounds[gene_id][2]
        step = random.choice([-step, step])
        while not (self.bounds[gene_id][0] <= indiv[gene_id] + step <= self.bounds[gene_id][1]):
            step = self.bounds[gene_id][2]
            step = random.choice([-step, step])

        indiv[gene_id] += step
        return indiv

    @staticmethod
    def __frange(start, stop, step):
        flist = []
        while start < stop:
            flist.append(start)
            start += step
        return flist





if __name__ == '__main__':
    ga = GADiophant(
        coefficients=(1, 2, 3, 4, 5, 6), # линейное уравнение вида: ax1 + bx2 + ... + zxn
        answer=123456,
        bounds=(-20000, 20000, 2),
        steps=1000,
        stop_fitness=0,
        population_limit=100,
    )
    result = ga.evolve()

    print_str = ''
    for i, x in enumerate(result[0]):
        print_str += f"{ga.coefficients[i]} * {x} + "
    print_str = print_str.strip()[:-1] + f"= {result[1]}"

    print("Solution:", print_str)
