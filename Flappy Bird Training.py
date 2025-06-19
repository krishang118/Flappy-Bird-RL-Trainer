import pygame
import random
import numpy as np
pygame.init()
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
BIRD_SIZE = 20
PIPE_WIDTH = 50
PIPE_GAP = 350  
PIPE_SPEED = 3
GRAVITY = 0.5
JUMP_STRENGTH = -8
FPS = 60
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
class Bird:
    def __init__(self):
        self.x = 100
        self.y = SCREEN_HEIGHT // 2
        self.velocity = 0
        self.size = BIRD_SIZE
    def jump(self):
        self.velocity = JUMP_STRENGTH
    def update(self):
        self.velocity += GRAVITY
        self.y += self.velocity
        if self.y < 0:
            self.y = 0
            self.velocity = 0
        elif self.y > SCREEN_HEIGHT - self.size:
            self.y = SCREEN_HEIGHT - self.size
            self.velocity = 0
    def get_rect(self):
        return pygame.Rect(self.x, self.y, self.size, self.size)
class Pipe:
    def __init__(self, x):
        self.x = x
        self.width = PIPE_WIDTH
        self.top_height = random.randint(80, SCREEN_HEIGHT - PIPE_GAP - 120)
        self.bottom_y = self.top_height + PIPE_GAP
    def update(self):
        self.x -= PIPE_SPEED
    def get_top_rect(self):
        return pygame.Rect(self.x, 0, self.width, self.top_height)
    def get_bottom_rect(self):
        return pygame.Rect(self.x, self.bottom_y, self.width, SCREEN_HEIGHT - self.bottom_y)
    def is_off_screen(self):
        return self.x + self.width < 0
class QLearningAgent:
    def __init__(self, state_size=6, action_size=2, learning_rate=0.1, 
                 discount_factor=0.95, epsilon=1.0, epsilon_decay=0.995, 
                 epsilon_min=0.01):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.q_table = {}
    def get_state(self, bird, pipes):
        if not pipes:
            return (int(bird.y // 50), 10, 10, int(bird.velocity // 2), 0, 7)        
        closest_pipe = None
        min_distance = float('inf')
        for pipe in pipes:
            if pipe.x + pipe.width > bird.x:
                distance = pipe.x - bird.x
                if distance < min_distance:
                    min_distance = distance
                    closest_pipe = pipe
        if closest_pipe is None:
            return (int(bird.y // 50), 10, 10, int(bird.velocity // 2), 0, 7)
        bird_y = int(bird.y // 50)  
        pipe_dist = int(min_distance // 80)  
        pipe_top = int(closest_pipe.top_height // 50)  
        bird_vel = int(bird.velocity // 2)  
        gap_center = closest_pipe.top_height + PIPE_GAP // 2
        relative_y = int((bird.y - gap_center) // 50)  
        gap_size = int(PIPE_GAP // 50)  
        return (bird_y, pipe_dist, pipe_top, bird_vel, relative_y, gap_size)    
    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.choice([0, 1]) 
        if state not in self.q_table:
            self.q_table[state] = [0.0, 0.0]
        return np.argmax(self.q_table[state])    
    def learn(self, state, action, reward, next_state, done):
        if state not in self.q_table:
            self.q_table[state] = [0.0, 0.0]
        if next_state not in self.q_table:
            self.q_table[next_state] = [0.0, 0.0]
        current_q = self.q_table[state][action]
        if done:
            target_q = reward
        else:
            target_q = reward + self.discount_factor * max(self.q_table[next_state])        
        self.q_table[state][action] = current_q + self.learning_rate * (target_q - current_q)        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
class FlappyBirdGame:
    def __init__(self):
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Flappy Bird Training")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        self.reset_game()
        self.agent = QLearningAgent()        
        self.episode = 0
        self.total_score = 0
        self.best_score = 0
        self.scores = []
    def reset_game(self):
        self.bird = Bird()
        self.pipes = []
        self.score = 0
        self.game_over = False
        self.pipe_timer = 0
    def spawn_pipe(self):
        self.pipes.append(Pipe(SCREEN_WIDTH))
    def update_game(self, action):
        if action == 1:
            self.bird.jump()
        self.bird.update()
        self.pipe_timer += 1
        if self.pipe_timer > 90:  
            self.spawn_pipe()
            self.pipe_timer = 0        
        for pipe in self.pipes[:]:
            pipe.update()
            if pipe.is_off_screen():
                self.pipes.remove(pipe)
                self.score += 1
        bird_rect = self.bird.get_rect()        
        for pipe in self.pipes:
            if bird_rect.colliderect(pipe.get_top_rect()) or bird_rect.colliderect(pipe.get_bottom_rect()):
                self.game_over = True        
        if self.bird.y <= 0 or self.bird.y >= SCREEN_HEIGHT - self.bird.size:
            self.game_over = True
        return self.game_over
    def get_reward(self):
        if self.game_over:
            return -100
        reward = 1
        if self.pipes:
            closest_pipe = min(self.pipes, key=lambda p: abs(p.x - self.bird.x))
            if closest_pipe.x + closest_pipe.width < self.bird.x:
                reward += 10  
        return reward
    def draw(self):
        self.screen.fill(WHITE)        
        pygame.draw.rect(self.screen, RED, self.bird.get_rect())        
        for pipe in self.pipes:
            pygame.draw.rect(self.screen, GREEN, pipe.get_top_rect())
            pygame.draw.rect(self.screen, GREEN, pipe.get_bottom_rect())
        score_text = self.font.render(f"Score: {self.score}", True, BLACK)
        episode_text = self.font.render(f"Episode: {self.episode}", True, BLACK)
        epsilon_text = self.font.render(f"Epsilon: {self.agent.epsilon:.3f}", True, BLACK)
        best_text = self.font.render(f"Best: {self.best_score}", True, BLACK)
        self.screen.blit(score_text, (10, 10))
        self.screen.blit(episode_text, (10, 50))
        self.screen.blit(epsilon_text, (10, 90))
        self.screen.blit(best_text, (10, 130))
        pygame.display.flip()
    def train(self, episodes=1000):
        for episode in range(episodes):
            self.episode = episode + 1
            self.reset_game()
            state = self.agent.get_state(self.bird, self.pipes)
            total_reward = 0            
            while not self.game_over:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return
                action = self.agent.choose_action(state)
                done = self.update_game(action)
                next_state = self.agent.get_state(self.bird, self.pipes)
                reward = self.get_reward()
                total_reward += reward                
                self.agent.learn(state, action, reward, next_state, done)
                state = next_state
                if episode % 50 == 0:
                    self.draw()
                    self.clock.tick(FPS)
            self.scores.append(self.score)
            if self.score > self.best_score:
                self.best_score = self.score
def main():
    game = FlappyBirdGame()
    game.train()
    pygame.quit()
if __name__ == "__main__":
    main()