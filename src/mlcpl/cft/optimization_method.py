import torch

class HeuristicForLinear():
    def __init__(self):
        pass
    
    def set_head(self, head):
        self.head = head

    def step(self, x, y):
        pass

    @staticmethod
    def encode(weight, bias):
        return torch.concatenate([weight.reshape(-1), bias.reshape(-1)], axis=0)
    
    @staticmethod
    def decode(sol):
        weight = sol[:-1].reshape(1, -1)
        bias = sol[-1].reshape(-1)
        return weight, bias

    @staticmethod
    def batch_fitness(population, x, y, metric):

        weight = population[:, :-1]
        bias = population[:, -1]
        preds = torch.nn.functional.linear(x, weight, bias=bias).transpose(0, 1).unsqueeze(-1)

        fitnesses = torch.zeros(preds.shape[0])
        for i in range(fitnesses.shape[0]):
            fitnesses[i] = metric(preds[i], y)

        return fitnesses

class GAForLinear(HeuristicForLinear):
    def __init__(
        self,
        head = None,
        device = 'cuda',
        metric = None,
        higher_is_better = True,
        init_pop_var = 0.0,
        num_pop = 50,
        Cr = 0.2,
        mutation_p = 0.5,
        mutation_percent_genes = 1,
        mutation_range = 0.001,
        elitism = 1,

        ):
        self.metric = metric
        self.device = device

        self.higher_is_better = higher_is_better
        self.init_pop_var = init_pop_var
        self.num_pop = num_pop
        self.Cr = Cr
        self.mutation_p = mutation_p
        self.mutation_percent_genes = mutation_percent_genes
        self.mutation_range = mutation_range
        self.elitism = elitism

        if head:
            self.set_head(head)

    def set_head(self, head):
        self.head = head
        self.population = torch.stack([self.encode(head.weight.clone().detach(), head.bias.clone().detach())] * self.num_pop).to(self.device)
        self.step_count = 0

    def step(self, x, y):
        with torch.no_grad():
            self.step_count += 1
            if (y==0.0).sum() == 0 or (y==1.0).sum() == 0:
                return

            pop_fitnesses = self.batch_fitness(self.population, x, y, self.metric)

            if self.higher_is_better:
                best_index = torch.argmax(pop_fitnesses)
                best_pop = self.population[best_index]
                best_fitness = torch.max(pop_fitnesses)
                avg_fitness = torch.mean(pop_fitnesses)
                parents_1_indices = torch.multinomial(pop_fitnesses, self.population.shape[0], replacement=True)
                parents_2_indices = torch.multinomial(pop_fitnesses, self.population.shape[0], replacement=True)
            else:
                best_index = torch.argmin(pop_fitnesses)
                best_pop = self.population[best_index]
                best_fitness = torch.min(pop_fitnesses)
                avg_fitness = torch.mean(pop_fitnesses)

                parents_1_indices = torch.multinomial(-pop_fitnesses, self.population.shape[0], replacement=True)
                parents_2_indices = torch.multinomial(-pop_fitnesses, self.population.shape[0], replacement=True)

            weight, bias = self.decode(best_pop)
            self.head.weight, self.bias = torch.nn.Parameter(weight), torch.nn.Parameter(bias)

            parents_1 = self.population[parents_1_indices]
            parents_2 = self.population[parents_2_indices]
            offsprings = torch.where(torch.rand_like((self.population)) < self.Cr, parents_1, parents_2)

            mutations = (torch.rand_like(offsprings) * 2 - 1) * self.mutation_range
            mutations = torch.where(torch.rand_like(mutations) < self.mutation_percent_genes, mutations, torch.zeros_like(mutations))
            mutations = torch.where((torch.rand_like(mutations)[:, 0] < self.mutation_p)[..., None], mutations, torch.zeros_like(mutations))
            offsprings = offsprings + mutations.to(self.device)

            if self.elitism > 0:
                sorted_population = self.population[torch.sort(pop_fitnesses, descending=self.higher_is_better)[1]]
                offsprings = torch.concat([sorted_population[: self.elitism], offsprings[self.elitism:]])

            self.population = offsprings

            log = {
                'best_fitness': best_fitness,
                'avg_fitness': avg_fitness,
            }

            return log

class BP():
    def __init__(
        self, 
        head = None,
        device = 'cuda',
        loss_fn = None,
        optimizer_class = None,
        optimizer_kwargs = None,
        ):
        self.loss_fn = loss_fn
        self.device = device
        self.optimizer_class = optimizer_class
        self.optimizer_kwargs = optimizer_kwargs

        if head:
            self.set_head(head)

    def set_head(self, head):
        self.head = head
        self.step_count = 0
        self.optimizer = self.optimizer_class(self.head.parameters(), **self.optimizer_kwargs)

    def step(self, x, y):
        self.step_count += 1
        if (y==0.0).sum() == 0 or (y==1.0).sum() == 0:
            return

        pred = self.head(x)
        loss = self.loss_fn(pred, y)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        log = {
            'loss': loss,
        }

        return log