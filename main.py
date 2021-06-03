import os, sys
import neat
import pygame
import numpy as np
import math
import random
import visualize
from pygame import draw
from classes.bodies import Body, Rocket, Planet
pygame.init()

# Global Constants
SCREEN_HEIGHT = 1000
SCREEN_WIDTH = 1000
FONT = pygame.font.Font('freesansbold.ttf', 20)
SCREEN = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
START_RADIUS = 100
START_VELOCITY = math.sqrt(10)
END_RADIUS = 200
END_VELOCITY = math.sqrt(5)
TOTAL_TIME_STEPS = 300

generation = 0

def reset():
    Rocket.rockets = []
    Body.bodies = []

def draw_orbits():
    # Starting orbit
    pygame.draw.circle(SCREEN, (0, 255, 0), (500, 500), START_RADIUS, 2)
    # Ending orbit
    pygame.draw.circle(SCREEN, (255, 0, 0), (500, 500), END_RADIUS, 2)

def eval_genomes(genomes, config):
    reset()
    clock = pygame.time.Clock()
    global generation
    time_step = 0

    rockets = []
    earth = Planet(np.array([500, 500], dtype=np.float64), 0, 1000, (168, 141, 50))
    ge = []
    nets = []

    def statistics():
        generation_text = FONT.render(f'Generation:  {generation}', True, (0, 0, 0))
        timestep_text = FONT.render(f'Time step:  {time_step}', True, (0, 0, 0))

        SCREEN.blit(generation_text, (50, 480))
        SCREEN.blit(timestep_text, (50, 510))

    starting_angle = 0
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)
        genome.fitness = 0
        ge.append(genome)
        rockets.append(Rocket(np.array([500 + START_RADIUS * math.cos(starting_angle), 500 + START_RADIUS * math.sin(starting_angle)], dtype=np.float64),
                              np.array([-START_VELOCITY * math.sin(starting_angle), START_VELOCITY * math.cos(starting_angle)], dtype=np.float64),
                              1, (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))))
        starting_angle += 2 * math.pi / len(genomes)
    
    run = True
    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        # Run for _ timesteps
        if time_step == TOTAL_TIME_STEPS:
            for rocket_id, rocket in enumerate(rockets):
                if (generation < 0):
                    ge[rocket_id].fitness -= rocket.get_cost(END_RADIUS, END_VELOCITY, earth, ["semimajor", "eccentricity"])
                else:
                    ge[rocket_id].fitness -= rocket.get_cost(END_RADIUS, END_VELOCITY, earth, ["semimajor", "eccentricity", "propulsion"])
            break

        # DRAWING
        SCREEN.fill((255, 255, 255))

        draw_orbits()
        earth.update()
        earth.draw(SCREEN)

        def get_data(rocket):
            r = rocket.position - earth.position
            radius_norm = (END_RADIUS - np.linalg.norm(r)) / END_RADIUS
            semimajor_norm = (END_RADIUS - rocket.semimajor_axis(earth)) / END_RADIUS
            eccentricity_norm = np.linalg.norm(rocket.eccentricity_vec(earth))
            velocity_norm = (END_VELOCITY - np.linalg.norm(rocket.velocity)) / END_VELOCITY
            theta_norm = np.arccos(np.dot(r, rocket.velocity) / np.linalg.norm(r) / np.linalg.norm(rocket.velocity)) /  (2 * math.pi)
            return (radius_norm, semimajor_norm, eccentricity_norm, theta_norm)
        
        for rocket_id, rocket in enumerate(rockets):
            output = nets[rocket_id].activate(get_data(rocket))
            
            i = output.index(max(output))
            if i == 0:
                rocket.propulsion = 1
            elif i == 1:
                rocket.propulsion = 0
            elif i == 2:
                rocket.propulsion = -1
            rocket.update()
            if rocket.semimajor_axis(earth) < 2 * END_RADIUS:
                rocket.draw_orbit(SCREEN, earth)
            rocket.draw(SCREEN)

        clock.tick(60)
        time_step += 1
        statistics()
        pygame.display.update()
    generation += 1

def run(config_path):
    config = neat.config.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path
    )

    pop = neat.Population(config)
    pop = neat.Checkpointer.restore_checkpoint("/Users/allengu/Documents/hohmann-transfer/runs/06-02-2021-20-36/neat-checkpoint-0")
    # pop = neat.Checkpointer.restore_checkpoint("/Users/allengu/Documents/hohmann-transfer/neat-checkpoint-43")
    pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.Checkpointer(1))
    best = pop.run(eval_genomes, 40)
    # eval_genomes(list(zip([1, 2, 3], stats.best_genomes(3))), config)
    eval_genomes([(1, stats.best_genome())], config)
    visualize.draw_net(config, best, True)
    visualize.plot_stats(stats, ylog=False, view=True)
    visualize.plot_species(stats, view=True)


if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config.txt')
    run(config_path)