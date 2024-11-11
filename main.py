import random
import csv
from math import ceil

# ------------------------------------------------ main parameters -----------------------------------------------------------

POPULATION_SIZE = 5
# the number of GPUs to include in a single gene
GENE_SIZE = 3
GENERATIONS = 30
MAX_COST = 50000
MIN_VRAM = 40
CROSSOVER_RATE = 0.7
MUTATION_RATE = 0.1

# a representation for each alele in the gene, or each gpu


class GpuAllele:
    def __init__(self, name, performance, cost, vram):
        self.name = name
        self.performance = performance
        self.cost = cost
        self.vram = vram


gpuList = []

# read all the gpus from the database
with open("gpus.csv", "r") as csvFile:
    csvReader = csv.reader(csvFile)

    names = next(csvReader)

    for line in csvReader:
        gpuName = line[0]
        performance = int(line[1])
        cost = int(line[2])
        vram = int(line[3])

        gpuList.append(GpuAllele(gpuName, performance, cost, vram))

# --------------------------------------------------- Utility Functions ----------------------------------------------------------


def initPopulation():
    population = []
    for _ in range(POPULATION_SIZE):
        gene = []
        for _ in range(GENE_SIZE):
            gene.append(random.choice(gpuList))
        population.append(gene)
    return population


def fitness(gene):
    geneFitness = 0
    geneCost = 0
    genevram = 0

    for gpu in gene:
        geneFitness += gpu.performance
        geneCost += gpu.cost
        genevram += gpu.vram

    # if this gene doesn't meet the constraints, return a very small fitness value
    if geneCost > MAX_COST and genevram < MIN_VRAM:
        return 1

    # a simple scaling factor
    return geneFitness + (MAX_COST - geneCost) / 15


def crossover(parent1, parent2):
    if random.random() < CROSSOVER_RATE:
        crossoverPoint = random.randint(1, GENE_SIZE - 1)
        return parent1[:crossoverPoint] + parent2[crossoverPoint:]
    return parent1


def mutate(gene):
    if random.random() < MUTATION_RATE:
        mutated_gpu = random.choice(gpuList)
        gene[random.randint(0, GENE_SIZE - 1)] = mutated_gpu
    return gene


# ===================================================== Main Loop =============================================================

if __name__ == "__main__":

    population = initPopulation()

    for _ in range(GENERATIONS):

        # find the fitness of each gene
        fitnessVals = []
        fitnessSum = 0
        for gene in population:
            fitnessVals.append(fitness(gene))
            fitnessSum += fitness(gene)

        # find the actual counts of each gene in the next generation now:
        nextGen = []
        for i, gene in enumerate(population):
            # for gpu in gene:
            #     print("perf= ", gpu.performance)
            #     print("cost= ", gpu.cost)
            #     print("vram= ", gpu.vram)

            probGenes = fitnessVals[i] / fitnessSum
            # print("prob genes= ", probGenes)
            expectedCount = probGenes * POPULATION_SIZE
            # print("exp count= ", expectedCount)
            actualCount = ceil(expectedCount)
            # print("actual count= ", actualCount)

            # add to the next generation
            for _ in range(actualCount):
                nextGen.append(gene)

        # find top 50% best genes in the next generation and perform crossover
        nextGen = sorted(nextGen, key=fitness, reverse=True)
        nextGen = nextGen[: POPULATION_SIZE // 2]

        while len(nextGen) < POPULATION_SIZE:
            parent1 = random.choice(nextGen)
            parent2 = random.choice(nextGen)
            nextGen.append(crossover(parent1, parent2))

        # perform mutation on all genes with MUTATION_RATE
        for i in range(POPULATION_SIZE):
            nextGen[i] = mutate(nextGen[i])

        # change over the generation to go to the next generation
        population = nextGen

    # print the best gene from the final population pool
    bestGene = max(population, key=fitness)

    print("The best solution is:")
    for gpu in bestGene:
        print(
            f"{gpu.name} with performance {gpu.performance}, cost {gpu.cost}, and VRAM {gpu.vram} GB"
        )
