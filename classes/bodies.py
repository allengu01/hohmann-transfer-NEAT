import pygame
import numpy as np
import math

class Body:
    bodies = []

    def __init__(self, position, velocity, mass, color, image):
        self.position = position
        self.velocity = velocity
        self.mass = mass
        self.color = color
        self.image = image
        self.bodies.append(self)

    def distance(self, other):
        diff_squares = [(a - b) ** 2 for a, b in zip(self.position, other.position)]
        return math.sqrt(sum(diff_squares))

    def net_gravitational_force(self, bodies):
        net_force = np.array([0, 0], dtype=np.float64)
        for body in bodies:
            if body is self or self.distance(body) < 1e-6:
                continue
            force_mag = body.mass * self.mass / self.distance(body) ** 2
            force_vec = force_mag * (body.position - self.position) / self.distance(body)
            net_force += force_vec
        return net_force

class Rocket(Body):
    rockets = []
    PROPULSION_FORCE = 0.01

    def __init__(self, position, velocity, mass, color, image=None):
        super().__init__(position, velocity, mass, color, image)
        self.rockets.append(self)
        self.propulsion = 0
        self.propulsion_history = np.array([], dtype=np.float64)

    def get_net_force(self):
        without_other_rockets = [body for body in self.bodies if body not in self.rockets]
        net_force = self.net_gravitational_force(without_other_rockets) # Rockets don't interact with each other

        # Rocket propulsion
        force_vec = self.propulsion * self.PROPULSION_FORCE * self.velocity / np.linalg.norm(self.velocity)
        np.append(self.propulsion_history, self.propulsion * self.PROPULSION_FORCE)
        net_force += force_vec
        return net_force

    def update(self):
        net_accel = self.get_net_force() / self.mass
        self.velocity += net_accel / 2
        self.position += self.velocity
        net_accel = self.get_net_force() / self.mass
        self.velocity += net_accel / 2

    def draw(self, surface):
        pygame.draw.circle(surface, self.color, self.position, 4)

    def semimajor_axis(self, planet):
        r = self.position - planet.position
        a = planet.mass * np.linalg.norm(r) / (2 * planet.mass - np.linalg.norm(r) * np.linalg.norm(self.velocity) ** 2) # semimajor axis
        return a

    def eccentricity_vec(self, planet):
        r = self.position - planet.position
        h = np.array([0, 0, np.cross(r, self.velocity)]) # angular momentum
        e = r / np.linalg.norm(r) - np.cross(self.velocity, h)[0:2] / planet.mass # eccentricity vector
        return e    

    def draw_orbit(self, surface, planet):
        r = self.position - planet.position
        a = self.semimajor_axis(planet)
        e = self.eccentricity_vec(planet)
        if not 0 <= np.linalg.norm(e) < 1 or np.linalg.norm(self.velocity) < 1e-6:
            return
        b = a * math.sqrt(1 - np.linalg.norm(e) ** 2) # semiminor axis
        center = planet.position + a * e
        rect = pygame.Rect(0, 0, a * 2, b * 2)
        ellipse_surf = pygame.Surface(rect.size, pygame.SRCALPHA)
        pygame.draw.ellipse(ellipse_surf, self.color, (0, 0, *rect.size), 1)
        rotated_surf = pygame.transform.rotate(ellipse_surf, -180 / math.pi * np.arctan2(e[1], e[0]))
        surface.blit(rotated_surf, rotated_surf.get_rect(center = center))

    def get_propulsion_history(self):
        return self.propulsion_history

    def get_cost(self, end_radius, end_velocity, planet, tags):
        cost = 0
        if ("velocity" in tags):
            cost += ((end_velocity - np.linalg.norm(self.velocity)) / end_velocity) ** 2
        if ("semimajor" in tags):
            cost += ((self.semimajor_axis(planet) - end_radius) / end_radius) ** 2
        if ("eccentricity" in tags):
            cost += np.linalg.norm(self.eccentricity_vec(planet)) ** 2
        if ("propulsion" in tags):
            cost += np.linalg.norm(self.propulsion_history / Rocket.PROPULSION_FORCE / self.propulsion_history.shape[0]) ** 2
        return cost

class Planet(Body):
    def __init__(self, position, velocity, mass, color, image=None):
        super().__init__(position, velocity, mass, color, image)

    def update(self):
        return

    def draw(self, surface):
        pygame.draw.circle(surface, self.color, self.position, self.mass ** 0.57 / 4)
    