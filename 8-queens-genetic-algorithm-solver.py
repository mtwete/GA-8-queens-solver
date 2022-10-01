"""
Matthew Twete

8-queens problem solver done using with genetic algorithm.
"""
#Import needed libraries
import numpy as np
import random
import matplotlib.pyplot as plt


#Class for running a genetic algorithm to solve the 8-queens problem and display the results.
#The solutions are represented as strings of length 8 with values between 1 and 8. Where each 
#value in the string represents the row number of the queen in that column. 
class QGA:
    
    
    #8-queens genetic algorithm class constructor. The arguments are:
    #populationSize, the size of the population to run the algorithm on. Please only use even numbers.
    #NumIteration, the number of generations (iterations) to run the algorithm for. 
    #MutationPerc, the gene mutation percentage the algorithm should use, please input in 
    #decimal form, not percentage. i.e. 0.01 for 1%. 
    def __init__(self, populationSize,numIteration,MutationPerc):
        #Population size
        self.popSize = populationSize
        #Number of generations (iterations) for the algorithm to run
        self.numIter = numIteration
        #Mutation percentage to use
        self.mutPerc = MutationPerc
        #Array to hold the average fitness of the population at a given generation
        self.avFitness = np.zeros(self.numIter)
        #List to hold random individuals sampled from different generations
        self.randomIndivid = []
        #List to hold the population of chromosome strings
        self.pop = []
        #Array to hold the fitness of the individuals in the population
        self.popFitness = np.zeros(self.popSize)
        
        
    #Function to generate the initial population, by randomly creating strings of 
    #length 8, with values between 1 and 8, so one queen per column. 
    def genPopulation(self):
        #Create popSize number of individuals as described above and add them to the population list
        for i in range(self.popSize):
            self.pop.append(''.join(str(random.randint(1,8)) for _ in range(8)))
            
            
    #Function to calculate the fitness of an individual. The fitness function I am using is the number 
    #of non mutually attacking pairs of queens. For each queen in the string, compares it with all the queens that
    #come after it (not before and after since that would double count) and for each other queen that isn't
    #mutually attacking with it, it adds one to the fitness score. The maximum score is 28, and the minimum is
    #0. The only argument is:
    #qstring, the 8-queen string to calculate the fitness of
    #The function will return the fitness score.
    def calcFitness(self, qstring):
        #Variable for the number of non-mutually attacking pairs of queens in the string
        count = 0
        #Loop over each queen
        for i in range(8):
            #Loop over remaining queens and add one to the count for each non-attacking pair
            for j in range(i+1,8):
                if (int(qstring[i]) != int(qstring[j]) and int(qstring[i]) != int(qstring[j]) + j - i and int(qstring[i]) != int(qstring[j]) - j +i):
                    count += 1
        
        #Return the count
        return count
    

    #Function to calculate the average fitness of the population at a given generation.
    #The only argument is:
    #genNumber, the generation number that the average population fitness is being calculated for
    def aveFitness(self, genNumber):
        #Calculate and store the average population fitness, which is simply the sum of 
        #inidivual fitnesses divided by the population size.
        self.avFitness[genNumber] = sum(self.popFitness)/self.popSize
        
        
    #Function to get a random sample individual from a generation for the written report.
    #Note, generation numbers are zero indexed. The only argument is:
    #genNumber, the generation number that the random sample individual is being taken from
    def getRandomInd(self,genNumber):
        #Pick a random individual from the population and store it in randomIndivid along with the generation number
        self.randomIndivid.append((random.choice(self.pop),genNumber))
        
        
    #Function to mutate a generated child if the probability occurs. If mutation occurs
    #it will randomly select a gene in the chromosome and then randomly generate a new 
    #value for that gene that is different than the old value. The only argument is:
    #child, the individual to mutate
    #The function will return the original child if no mutation occurs or the mutated child
    #if it does occur.
    def mutate(self, child):
        #Generate a random value between 0 and 1 to be the mutation probability
        mutationChance = random.random()
        #If the randomly generated value is less than the mutation percentage, then mutate
        if (mutationChance < self.mutPerc):
            #Randomly select a gene to mutate
            gene = random.randint(0,7)
            #Get the old value of the mutated gene
            oldvalue = int(child[gene])
            #Generate a new value for the mutated gene
            newvalue = random.randint(1,8)
            #Make sure the new value is different from the old value
            while (newvalue == oldvalue):
                newvalue = random.randint(1,8)
            #Since strings are immutable, create a mutated child that is the same as the original child
            #except for the one gene that was mutated
            mutatedChild = child[0:gene] + str(newvalue) + child[gene+1:]
            #return the mutated child
            return mutatedChild
        #Return the original child if no mutation occurs
        return child
    
    
    #Function to randomly select a parent for breeding from the population in proportion to their normalized 
    #fitness value as described in the assignment instructions. The function will return the index
    #of the selected parent in the pop list.
    def selectParent(self):
        #Generate a random value between 0 and 1
        chance = np.random.rand()
        #Variable to hold the index of the select parent
        parentIndex = 0
        #The fitness CDF essentially slices up the range 0 to 1 in proportion to the normalized 
        #fitness of the individuals. Parent index is incremented until the probability range of 
        #the individual that chance falls in is found. The index of the individual in self.pop is
        #the same as the index of the probability range of that individual in fitnessCDF. So once
        #the index in the right range is found, it will return it to get that inidividual for breeding
        while (chance > self.fitnessCDF[parentIndex]):
            parentIndex += 1
        #Return the index of the individual in self.pop to breed 
        return parentIndex
    
    
    #Function to crossover two parents and generate two children. It will crossover
    #by randomly selecting a crossover point and splicing the two parents before and after
    #that point to generate two children. The arguments are:
    #parent1, the first parent to breed with
    #parent2, the second parent to breed with
    #Returns, child1 and child2, the two children generated by the crossover
    def crossover(self,parent1,parent2):
        #Randomly select a crossover point
        crossPoint = random.randint(0,8)
        #Generate the two children by crossing over the parent chromosomes
        child1 = parent1[0:crossPoint] + parent2[crossPoint:]
        child2 = parent2[0:crossPoint] + parent1[crossPoint:]
        #return the children
        return child1, child2
        
    
    #Function to generate a new population of the same size after each generation.
    #It will randomly select parents to breed in proportion to their 
    #normalized fitness as described in the assignment instructions. 
    #The same individual is allowed to be selected to be both parents. 
    def repopulate(self):
        #Temporary list to hold the children generated for the new population
        newPop = []
        #Create children until the new population is the same size as the previous one
        while (len(newPop) != self.popSize):
            #Select two parents randomly in proportion to their normalized fitness
            parent1 = self.pop[self.selectParent()]
            parent2 = self.pop[self.selectParent()]
            #Crossover the parents to generate two children
            child1, child2 = self.crossover(parent1,parent2)
            #Mutate the children (if the probability occurs)
            child1 = self.mutate(child1)
            child2 = self.mutate(child2)
            #Add the new children to the temporary list
            newPop.append(child1)
            newPop.append(child2)
        #Set the population list to the newly generated population
        self.pop = newPop
        
        
    #Function to run the genetic algorithm, it will run for a set number of generations determined
    #by numIter. It will generate the initial population, determine the average population fitness
    #and normalized fitness then create a new population each generation. 
    def evolve(self):
        #Generate the initial population
        self.genPopulation()
        #Run the algorithm for numIter generations
        for i in range(self.numIter):
            #Calculate the fitness of each individual in the population
            for j in range(self.popSize):
                self.popFitness[j] = self.calcFitness(self.pop[j])
            #Select and store a random individual from the population every 500 generations (which will include 
            #the first generation) also if it is the last generation select a random individual from the population
            #as well. These individuals will be shown as examples when displaying the results.
            if (i % 500 == 0 or i == self.numIter - 1):
                self.getRandomInd(i)
            #Calculcate the average fitness of the population
            self.aveFitness(i)
            #Calculate the normalized fitness of each individual in the population
            self.normFitness = self.popFitness/sum(self.popFitness)
            #Set up a CDF of the population fitness to be used to select parents to breed randomly in proportion to 
            #their normalized fitness
            self.fitnessCDF = np.zeros(len(self.normFitness))
            for k in range(len(self.normFitness)):
                if (k == 0):
                    self.fitnessCDF[k] = self.normFitness[k]
                else:
                    self.fitnessCDF[k] = self.normFitness[k] + self.fitnessCDF[k-1]
            #Generate the next generation of individuals
            self.repopulate()
            
            
    #Function to display the results of the running of the algorithm
    #It will print out the random inidividuals sampled from the population from different
    #generations and then create and display the average population fitness plot.
    def displayResults(self):
        #Display the sample individuals taken from different generations of the genetic algorithm
        print("Example individuals sampled from the population at different generations: ")
        for samples in self.randomIndivid:
            print("Sample individual from generation ",samples[1] + 1," : ", samples[0])
            
        #Set up and display the average fitness plot
        #Create generation values for x-axis
        gens = np.arange(1,self.numIter + 1)
        #Plot the average population fitness against the generation number
        plt.plot(gens, self.avFitness)
        #Add text to the plot and display
        plt.xlabel('Generation number')
        plt.ylabel('Average population fitness')
        plt.title("Average population fitness against generation number for a population size of " + str(self.popSize) + " and mutation chance of " + str(self.mutPerc*100) + "%")
        plt.show()
            




#Run the algorithm with a population size of 500 and a mutation percentaeg of 0.1%
#for 3000 generations.    
a = QGA(500,3000,0.001)
a.evolve()
a.displayResults()

    
