import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import json
import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

class BuckshotRouletteEnv:
    def __init__(self):
        self.reset()

    def reset(self):
        hp_rand = random.choice([2, 3, 4])
        self.max_hp = hp_rand
        self.player_hp = hp_rand
        self.dealer_hp = hp_rand
        item_rand = random.randint(2, 5)
        self.items = self.random_items(item_rand)
        self.dealer_items = self.random_items(item_rand)
        self.clip = self.load_shotgun()
        self.turn = 'player'
        self.handcuffed = {'player': 0, 'dealer': 0}
        self.handsaw_active = {'player': 0, 'dealer': 0}
        self.player_clip_probs = self.initialize_clip_probs()
        self.dealer_clip_probs = self.initialize_clip_probs()
        self.inverter_final = False

        self.all_possible_items = ['cigarettes', 'handcuffs', 'adrenaline', 'magnifying_glass', 'beer', 'handsaw', 'expired_medicine', 'burner_phone', 'inverter']
        all_possible_actions_player =   [('player', 'use_cigarettes'), ('player', 'use_adrenaline_cigarettes'), \
                                         ('player', 'use_handcuffs'), ('player', 'use_adrenaline_handcuffs'), \
                                         ('player', 'use_magnifying_glass'), ('player', 'use_adrenaline_magnifying_glass'), \
                                         ('player', 'use_beer'), ('player', 'use_adrenaline_beer'), \
                                         ('player', 'use_handsaw'), ('player', 'use_adrenaline_handsaw'), \
                                         ('player', 'use_expired_medicine'), ('player', 'use_adrenaline_expired_medicine'), \
                                         ('player', 'use_burner_phone'), ('player', 'use_adrenaline_burner_phone'), \
                                         ('player', 'use_inverter'), ('player', 'use_adrenaline_inverter'), \
                                         ('player', 'shoot_self'), ('player', 'shoot_dealer')]
        all_possible_actions_dealer =   [('dealer', 'use_cigarettes'), ('dealer', 'use_adrenaline_cigarettes'), \
                                         ('dealer', 'use_handcuffs'), ('dealer', 'use_adrenaline_handcuffs'), \
                                         ('dealer', 'use_magnifying_glass'), ('dealer', 'use_adrenaline_magnifying_glass'), \
                                         ('dealer', 'use_beer'), ('dealer', 'use_adrenaline_beer'), \
                                         ('dealer', 'use_handsaw'), ('dealer', 'use_adrenaline_handsaw'), \
                                         ('dealer', 'use_expired_medicine'), ('dealer', 'use_adrenaline_expired_medicine'), \
                                         ('dealer', 'use_burner_phone'), ('dealer', 'use_adrenaline_burner_phone'), \
                                         ('dealer', 'use_inverter'), ('dealer', 'use_adrenaline_inverter'), \
                                         ('dealer', 'shoot_self'), ('dealer', 'shoot_dealer')]  # Define all possible actions
        self.all_possible_actions_player = all_possible_actions_player
        self.all_possible_actions_dealer = all_possible_actions_dealer

        

        return self.get_state()

    def all_possible_actions(self, player_name):
        if player_name == "dealer":
            return self.all_possible_actions_dealer
        elif player_name == "player":
            return self.all_possible_actions_player
        else:
            print("HUGE ERROR: ALL POSSIBLE ACTIONS FUNC")
            exit(0)
        
    def load_shotgun(self):
        # Choose a random combination of live and blank shells
        live_blank_combination = random.choice([
            [1, 1], [1, 2], [2, 2], [2, 3], [3, 3], [3, 4], [4, 4]
        ])
        live_count, blank_count = live_blank_combination

        # Create an array based on the number of live and blank shells
        shells = [1] * live_count + [0] * blank_count
        random.shuffle(shells)
        return shells


    def initialize_clip_probs(self):
        total_shells = len(self.clip)
        live_count = self.clip.count(1)
        blank_count = total_shells - live_count
        live_prob = live_count / total_shells
        blank_prob = blank_count / total_shells
        return [(live_prob, blank_prob) for _ in range(total_shells)]

    def update_clip_probs(self, player, index, is_live):
        probs = self.player_clip_probs if player == 'player' else self.dealer_clip_probs

        #print("Begin function")
        # Step 1 - Get live_count, blank_count, total rounds from self.clip
        live_count = self.clip.count(1)
        blank_count = self.clip.count(0)
        total_rounds = len(self.clip)
        #print("Live count: {} Blank count: {} Total rounds: {} Self.clip: {}".format(live_count, blank_count, total_rounds, self.clip))
        
        # Step 2 - Formulate an array based on if player knows the position of anything
        knowledge = []
        for prob in probs:
            if prob == (1, 0):
                knowledge.append('live')
            elif prob == (0, 1):
                knowledge.append('blank')
            else:
                knowledge.append('unknown')
        #print("Knowledge: {}".format(knowledge))
        
        # Step 3 - Update this array, changing 'index' to either live or blank depending on is_live
        knowledge[index] = 'live' if is_live else 'blank'
        #print("New knowledge: {}".format(knowledge))

        # Step 4 - Create an entirely new probability array based on this information
        unknown_count = knowledge.count('unknown')
        remaining_live = live_count - knowledge.count('live')
        remaining_blank = blank_count - knowledge.count('blank')

        #print("Unknown count: {} Remaining live: {} Remaining blank: {}".format(unknown_count, remaining_live, remaining_blank))
        
        new_probs = []
        for k in knowledge:
            if k == 'live':
                new_probs.append((1, 0))
            elif k == 'blank':
                new_probs.append((0, 1))
            else:
                if unknown_count > 0:
                    live_prob = remaining_live / unknown_count
                    blank_prob = remaining_blank / unknown_count
                    new_probs.append((live_prob, blank_prob))
                else:
                    new_probs.append((0, 0))  # Should not happen, but just in case

        #print("New probs: {}".format(new_probs))

        # Step 5 - Set old array to this new array
        if player == 'player':
            self.player_clip_probs = new_probs
        else:
            self.dealer_clip_probs = new_probs

    


    def random_items(self, num_items):
        return random.choices(['cigarettes', 'handcuffs', 'magnifying_glass', 'beer', 'handsaw', 'adrenaline', 'expired_medicine', 'burner_phone', 'inverter'], k=num_items)

    def get_state(self, player_view='player'):
        max_clip_length = 8  # Assuming the maximum clip length is 8

        # Flatten the state into a list
        state = [
            self.max_hp / 4,
            self.player_hp / 4,
            self.dealer_hp / 4,
            self.turn == 'player',
            self.handcuffed['player'] / 2,
            self.handcuffed['dealer'] / 2,
            self.handsaw_active['player'],
            self.handsaw_active['dealer'],
            self.inverter_final,
            len(self.clip) / 8
        ]

        # Flatten player and dealer items
        # self.all_possible_items = ['cigarettes', 'handcuffs', 'adrenaline', 'magnifying_glass', 'beer', 'handsaw', 'expired_medicine', 'burner_phone', 'inverter']
        state.extend([self.items.count(item) / 8 for item in self.all_possible_items])
        state.extend([self.dealer_items.count(item) / 8 for item in self.all_possible_items])


        # Flatten and normalize clip probabilities based on player view
        if player_view == 'player':
            clip_probs = self.player_clip_probs
        else:
            clip_probs = self.dealer_clip_probs

        for prob in clip_probs:
            #state.extend(prob) CHANGED!!!!!!!!!!!!!! TODO
            state.append(prob[0])

        # Pad the remaining probabilities to ensure consistent state length CHANGED TODO
        # while len(state) < (max_clip_length) + len(self.all_possible_items) * 2 + 10 + (max_clip_length):
        while len(state) < (max_clip_length) + len(self.all_possible_items) * 2 + 10:
            state.append(-1)

        return state

    
    def to_string(self):
        # state = self.get_state()
        # print("Player HP:\t {}".format(state[0]))
        # print("Dealer HP:\t {}".format(state[1]))
        # print("Player Items:\t {}".format(state[2]))
        # print("Dealer Items:\t {}".format(state[3]))
        # print("Clip:\t {}".format(state[4]))
        # print("Turn:\t {}".format(state[5]))
        # print("Handcuffed:\t {}".format(state[6]))
        # print("Handsaw:\t {}".format(state[7]))
        # print("Player Probs:\t {}".format(state[8]))
        # print("Dealer Probs:\t {}".format(state[9]))
        pass


    def copy(self):
        new_env = BuckshotRouletteEnv()
        new_env.max_hp = self.max_hp
        new_env.player_hp = self.player_hp
        new_env.dealer_hp = self.dealer_hp
        new_env.items = self.items.copy()
        new_env.dealer_items = self.dealer_items.copy()
        new_env.clip = self.clip.copy()
        new_env.turn = self.turn
        new_env.handcuffed = self.handcuffed.copy()
        new_env.handsaw_active = self.handsaw_active.copy()
        new_env.player_clip_probs = self.player_clip_probs.copy()
        new_env.dealer_clip_probs = self.dealer_clip_probs.copy()
        new_env.inverter_final = self.inverter_final
        new_env.all_possible_actions_player = self.all_possible_actions_player.copy()
        new_env.all_possible_actions_dealer = self.all_possible_actions_dealer.copy()
        return new_env


    def step(self, action):
        player, move = action
        done = False
        turn_over = False

        if move == 'use_cigarettes':
            if player == 'player':
                self.items.remove('cigarettes')
                if self.player_hp < self.max_hp:
                    self.player_hp += 1
            elif player == 'dealer':
                self.dealer_items.remove('cigarettes')
                if self.dealer_hp < self.max_hp:
                    self.dealer_hp += 1

        if move == 'use_adrenaline_cigarettes':
            if player == 'player':
                self.items.remove('adrenaline')
                self.dealer_items.remove('cigarettes')
                if self.player_hp < self.max_hp:
                    self.player_hp += 1
            elif player == 'dealer':
                self.dealer_items.remove('adrenaline')
                self.items.remove('cigarettes')
                if self.dealer_hp < self.max_hp:
                    self.dealer_hp += 1

        elif move == 'use_handcuffs':
            we_have_handcuffs = True
            if player == 'dealer':
                if not 'handcuffs' in self.dealer_items:
                    we_have_handcuffs = False
                else:
                    self.dealer_items.remove('handcuffs')
            else:
                if not 'handcuffs' in self.items:
                    we_have_handcuffs = False
                else:
                    self.items.remove('handcuffs')
            if we_have_handcuffs:
                self.handcuffed['dealer' if player == 'player' else 'player'] = 2
            else:
                print("Exceptional case in use_handcuffs")
                print("Begin Debug -----------")
                print(self.get_state())
                print(self.get_possible_actions())
                print("END --------------")

        elif move == 'use_adrenaline_handcuffs':
            we_have_handcuffs = True
            if player == 'dealer':
                if not 'handcuffs' in self.items or not 'adrenaline' in self.dealer_items:
                    we_have_handcuffs = False
                else:
                    self.dealer_items.remove('adrenaline')
                    self.items.remove('handcuffs')
            else:
                if not 'handcuffs' in self.dealer_items or not 'adrenaline' in self.items:
                    we_have_handcuffs = False
                else:
                    self.items.remove('adrenaline')
                    self.dealer_items.remove('handcuffs')
            if we_have_handcuffs:
                self.handcuffed['dealer' if player == 'player' else 'player'] = 2
            else:
                print("Exceptional case in use_adrenaline_handcuffs")

        elif move == 'use_magnifying_glass':
            # Reveal the next shell in the chamber
            if player == 'dealer':
                self.dealer_items.remove('magnifying_glass')
            else:
                self.items.remove('magnifying_glass')

            is_live = self.clip[0] == 1
            self.update_clip_probs(player, 0, is_live)

        elif move == 'use_adrenaline_magnifying_glass':
            # Reveal the next shell in the chamber
            if player == 'dealer':
                self.dealer_items.remove('adrenaline')
                self.items.remove('magnifying_glass')
            else:
                self.items.remove('adrenaline')
                self.dealer_items.remove('magnifying_glass')

            is_live = self.clip[0] == 1
            self.update_clip_probs(player, 0, is_live)


        elif move == 'use_adrenaline_beer':
            #self.clip.pop(0)  # Eject the current shell
            # Update probabilities for both players
            if player == 'dealer':
                self.dealer_items.remove('adrenaline')
                self.items.remove('beer')
            else:
                self.items.remove('adrenaline')
                self.dealer_items.remove('beer')

            self.update_probabilities_after_shot()

        elif move == 'use_beer':
            #self.clip.pop(0)  # Eject the current shell
            # Update probabilities for both players
            if player == 'dealer':
                self.dealer_items.remove('beer')
            else:
                self.items.remove('beer')

            self.update_probabilities_after_shot()

        elif move == 'use_handsaw':
            if player == 'dealer':
                self.dealer_items.remove('handsaw')
            else:
                self.items.remove('handsaw')
            if player == 'player':
                self.handsaw_active['player'] = 1
            else:
                self.handsaw_active['dealer'] = 1

        elif move == 'use_adrenaline_handsaw':
            if player == 'dealer':
                self.dealer_items.remove('adrenaline')
                self.items.remove('handsaw')
            else:
                self.items.remove('adrenaline')
                self.dealer_items.remove('handsaw')
            if player == 'player':
                self.handsaw_active['player'] = 1
            else:
                self.handsaw_active['dealer'] = 1

        elif move == 'use_expired_medicine':

            if player == 'dealer':
                self.dealer_items.remove('expired_medicine')
            else:
                self.items.remove('expired_medicine')

            if random.choice([True, False]):
                if player == 'player' and self.player_hp < self.max_hp:
                    self.player_hp += 2
                elif player == 'dealer' and self.dealer_hp < self.max_hp:
                    self.dealer_hp += 2
                if self.player_hp > self.max_hp:
                    self.player_hp = self.max_hp
                if self.dealer_hp > self.max_hp:
                    self.dealer_hp = self.max_hp
            else:
                if player == 'player':
                    self.player_hp -= 1
                else:
                    self.dealer_hp -= 1
            done = self.player_hp <= 0 or self.dealer_hp <= 0
            turn_over = self.player_hp <= 0 or self.dealer_hp <= 0

        elif move == 'use_adrenaline_expired_medicine':

            if player == 'dealer':
                self.dealer_items.remove('adrenaline')
                self.items.remove('expired_medicine')
            else:
                self.items.remove('adrenaline')
                self.dealer_items.remove('expired_medicine')


            if random.choice([True, False]):
                if player == 'player' and self.player_hp < self.max_hp:
                    self.player_hp += 2
                elif player == 'dealer' and self.dealer_hp < self.max_hp:
                    self.dealer_hp += 2
                if self.player_hp > self.max_hp:
                    self.player_hp = self.max_hp
                if self.dealer_hp > self.max_hp:
                    self.dealer_hp = self.max_hp
            else:
                if player == 'player':
                    self.player_hp -= 1
                else:
                    self.dealer_hp -= 1
            done = self.player_hp <= 0 or self.dealer_hp <= 0
            turn_over = self.player_hp <= 0 or self.dealer_hp <= 0

        elif move == 'use_burner_phone':
            if player == 'dealer':
                self.dealer_items.remove('burner_phone')
            else:
                self.items.remove('burner_phone')
            # Reveal the state of one future shell at random
            if len(self.clip) == 1:
                future_index = 0
            else:
                future_index = random.randint(1, len(self.clip) - 1)
            future_shell = self.clip[future_index]
            is_live = future_shell == 1
            self.update_clip_probs(player, future_index, is_live)

        elif move == 'use_adrenaline_burner_phone':
            if player == 'dealer':
                self.dealer_items.remove('adrenaline')
                self.items.remove('burner_phone')
            else:
                self.items.remove('adrenaline')
                self.dealer_items.remove('burner_phone')
            # Reveal the state of one future shell at random
            if len(self.clip) == 1:
                future_index = 0
            else:
                future_index = random.randint(1, len(self.clip) - 1)
            future_shell = self.clip[future_index]
            is_live = future_shell == 1
            self.update_clip_probs(player, future_index, is_live)

        elif move == 'use_inverter':
            if player == 'dealer':
                self.dealer_items.remove('inverter')
            else:
                self.items.remove('inverter')
            self.inverter_final = True
            if self.clip[0] == 0:
                self.clip[0] = 1
            elif self.clip[0] == 1:
                self.clip[0] = 0
        elif move == 'use_adrenaline_inverter':
            if player == 'dealer':
                self.dealer_items.remove('adrenaline')
                self.items.remove('inverter')
            else:
                self.items.remove('adrenaline')
                self.dealer_items.remove('inverter')
            self.inverter_final = True
            if self.clip[0] == 0:
                self.clip[0] = 1
            elif self.clip[0] == 1:
                self.clip[0] = 0
        elif move == 'shoot_self':
            is_live = self.clip[0] == 1
            if is_live:
                damage = 2 if self.handsaw_active[player] else 1
                if player == 'player':
                    self.player_hp -= damage
                else:
                    self.dealer_hp -= damage
                self.handsaw_active[player] = 0
                done = self.player_hp <= 0 or self.dealer_hp <= 0
                turn_over = True
            else:
                self.handsaw_active[player] = 0
                pass  # Blank, no damage, player continues turn
            #self.clip.pop(0)
            self.update_probabilities_after_shot()

        elif move == 'shoot_dealer':
            is_live = self.clip[0] == 1
            if is_live:
                damage = 2 if self.handsaw_active[player] else 1
                if player == 'player':
                    self.dealer_hp -= damage
                else:
                    self.player_hp -= damage
            self.handsaw_active[player] = 0
            #self.clip.pop(0)
            done = self.player_hp <= 0 or self.dealer_hp <= 0
            turn_over = True
            self.update_probabilities_after_shot()

        if turn_over:
            self.inverter_final = False

            if self.handcuffed['player'] == 2:
                self.handcuffed['player'] = 1
                self.turn = 'dealer'
            elif self.handcuffed['dealer'] == 2:
                self.handcuffed['dealer'] = 1
                self.turn = 'player'
            elif self.handcuffed['player'] == 1:
                self.handcuffed['player'] = 0
                self.turn = 'player'
            elif self.handcuffed['dealer'] == 1:
                self.handcuffed['dealer'] = 0
                self.turn = 'dealer'
            elif self.turn == 'dealer':
                self.turn = 'player'
            elif self.turn == 'player':
                self.turn = 'dealer'

        if len(self.clip) == 0:
            self.clip = self.load_shotgun()
            rand_items = random.randint(2, 5)

            #self.items = self.random_items(rand_items)
            #self.dealer_items = self.random_items(rand_items)

            new_player_items = self.random_items(rand_items)
            new_dealer_items = self.random_items(rand_items)

            while len(self.items) < 8 and len(new_player_items) > 0:
                new_item = new_player_items.pop(0)
                self.items.append(new_item)

            while len(self.dealer_items) < 8 and len(new_dealer_items) > 0:
                new_item = new_dealer_items.pop(0)
                self.dealer_items.append(new_item)

            self.handcuffed = {'player': 0, 'dealer': 0}
            self.handsaw_active = {'player': 0, 'dealer': 0}

            self.player_clip_probs = self.initialize_clip_probs()
            self.dealer_clip_probs = self.initialize_clip_probs()
            self.turn = 'player'
        
        if done:
            #reward = -1 if self.player_hp <= 0 else 1
            reward = -1 if self.player_hp <= 0 else 0.1
        else:
            reward = 0

        return self.get_state(), reward, done, {}

    def update_probabilities_after_shot(self):
        is_live = self.clip[0] == 1
        if is_live:
            self.update_clip_probs('player', 0, True)
            self.update_clip_probs('dealer', 0, True)
        else:
            self.update_clip_probs('player', 0, False)
            self.update_clip_probs('dealer', 0, False)

        self.clip.pop(0)
        self.player_clip_probs.pop(0)
        self.dealer_clip_probs.pop(0)

    def get_possible_actions(self, state=None):

        if state is not None: # Rebuild from state
            new_env = self.copy()
            #print(state)
            #exit(0)
            new_env.max_hp = state[0] * 4
            new_env.player_hp = state[1] * 4
            new_env.dealer_hp = state[2] * 4
            items_state = copy.deepcopy(state[10:19])
            new_env.items = []
            for i, j in enumerate(items_state):
                while j > 0:
                    if i == 0:
                        new_env.items.append('cigarettes')
                        j -= 0.125 #self.all_possible_items = ['cigarettes', 'handcuffs', 'adrenaline', 'magnifying_glass', 'beer', 'handsaw', 'expired_medicine', 'burner_phone', 'inverter']
                    elif i == 1:
                        new_env.items.append('handcuffs')
                        j -= 0.125
                    elif i == 2:
                        new_env.items.append('adrenaline')
                        j -= 0.125
                    elif i == 3:
                        new_env.items.append('magnifying_glass')
                        j -= 0.125
                    elif i == 4:
                        new_env.items.append('beer')
                        j -= 0.125
                    elif i == 5:
                        new_env.items.append('handsaw')
                        j -= 0.125
                    elif i == 6:
                        new_env.items.append('expired_medicine')
                        j -= 0.125
                    elif i == 7:
                        new_env.items.append('burner_phone')
                        j -= 0.125
                    elif i == 8:
                        new_env.items.append('inverter')
                        j -= 0.125
            dealer_items_state = copy.deepcopy(state[19:28])
            new_env.dealer_items = []
            for i, j in enumerate(dealer_items_state):
                while j > 0:
                    if i == 0:
                        new_env.dealer_items.append('cigarettes')
                        j -= 0.125 #self.all_possible_items = ['cigarettes', 'handcuffs', 'adrenaline', 'magnifying_glass', 'beer', 'handsaw', 'expired_medicine', 'burner_phone', 'inverter']
                    elif i == 1:
                        new_env.dealer_items.append('handcuffs')
                        j -= 0.125
                    elif i == 2:
                        new_env.dealer_items.append('adrenaline')
                        j -= 0.125
                    elif i == 3:
                        new_env.dealer_items.append('magnifying_glass')
                        j -= 0.125
                    elif i == 4:
                        new_env.dealer_items.append('beer')
                        j -= 0.125
                    elif i == 5:
                        new_env.dealer_items.append('handsaw')
                        j -= 0.125
                    elif i == 6:
                        new_env.dealer_items.append('expired_medicine')
                        j -= 0.125
                    elif i == 7:
                        new_env.dealer_items.append('burner_phone')
                        j -= 0.125
                    elif i == 8:
                        new_env.dealer_items.append('inverter')
                        j -= 0.125

            new_env.clip = self.clip.copy()
            if state[3] == 1:
                new_env.turn = 'player'
            elif state[3] == 0:
                new_env.turn = 'dealer'
            else:
                print("CRITICAL EXCEPTION get_possible_actions() state[3] is not 1 or 0")
                exit(0)
            new_env.handcuffed = self.handcuffed.copy()
            new_env.handcuffed['player'] = state[4]
            new_env.handcuffed['dealer'] = state[5]
            new_env.handsaw_active = self.handsaw_active.copy()
            new_env.handsaw_active['player'] = state[6]
            new_env.handsaw_active['dealer'] = state[7]
            new_env.player_clip_probs = self.player_clip_probs.copy()
            new_env.dealer_clip_probs = self.dealer_clip_probs.copy()
            if state[8] == 1:
                new_env.inverter_final = True
            elif state[8] == 0:
                new_env.inverter_final = False
            else:
                print("CRITICAL EXCEPTION get_possible_actions() state[8] is not 1 or 0")
                exit(0)
            new_env.all_possible_actions_player = self.all_possible_actions_player.copy()
            new_env.all_possible_actions_dealer = self.all_possible_actions_dealer.copy()
            
            # DEBUG
            #result = new_env.get_possible_actions()
            #print("STATE: {}".format(state))
            #print("RESULT/POSSIBLE ACTIONS: {}".format(result))
            #exit(0)

            return new_env.get_possible_actions()

        actions = []
        if self.turn == 'player':
            if self.inverter_final: # workaround to force inverter to be final action to prevent knowledge leak exploits from my poor implementation
                actions.append(('player', 'shoot_self'))
                actions.append(('player', 'shoot_dealer'))
                return actions
            for item in self.items:
                if item != 'adrenaline':
                    actions.append(('player', f'use_{item}'))
            # Add specific adrenaline actions based on dealer's items if player has adrenaline
            if 'adrenaline' in self.items:
                for item in self.dealer_items:
                    if item != 'adrenaline':
                        actions.append(('player', f'use_adrenaline_{item}'))
            if self.handcuffed['dealer'] > 0:
                while ('player', 'use_handcuffs') in actions:
                    actions.remove(('player', 'use_handcuffs'))
                while ('player', 'use_adrenaline_handcuffs') in actions:
                    actions.remove(('player', 'use_adrenaline_handcuffs'))
            actions.append(('player', 'shoot_self'))
            actions.append(('player', 'shoot_dealer'))
        else:
            if self.inverter_final: # workaround to force inverter to be final action to prevent knowledge leak exploits from my poor implementation
                actions.append(('dealer', 'shoot_self'))
                actions.append(('dealer', 'shoot_dealer'))
                return actions
            for item in self.dealer_items:
                if item != 'adrenaline':
                    actions.append(('dealer', f'use_{item}'))
            # Add specific adrenaline actions based on player's items if dealer has adrenaline
            if 'adrenaline' in self.dealer_items:
                for item in self.items:
                    if item != 'adrenaline':
                        actions.append(('dealer', f'use_adrenaline_{item}'))
            if self.handcuffed['player'] > 0:
                while ('dealer', 'use_handcuffs') in actions:
                    actions.remove(('dealer', 'use_handcuffs'))
                while ('dealer', 'use_adrenaline_handcuffs') in actions:
                    actions.remove(('dealer', 'use_adrenaline_handcuffs'))
            actions.append(('dealer', 'shoot_self'))
            actions.append(('dealer', 'shoot_dealer'))
        actions = list(set(actions))
        if ('player', 'use_adrenaline') in actions or ('dealer', 'use_adrenaline') in actions:
            print("CRITICAL ERROR get_possible_actions")
            print(actions)
            exit(0)
        return actions




class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, output_dim)

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.98, epsilon=1.0, epsilon_decay=0.99995, epsilon_min=0.05):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.memory = deque(maxlen=2000)
        
        # COMMENT OUT if using CPU - also del .to(self.device)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQN(state_dim, action_dim).to(self.device)
        self.target_model = DQN(state_dim, action_dim).to(self.device)


        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.update_target_model()

    def update_target_model(self, tau=0.05):
        #self.target_model.load_state_dict(self.model.state_dict())
        for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


    def remember(self, state, action, reward, next_state, done, unhang, force_unhang=False):

        if force_unhang:
            try:
                (oldstate, oldaction) = self.memory.pop()
            except:
                return
            # CHECK IF THIS IS NOT NOTHING
            self.memory.append((oldstate, oldaction, reward, next_state, done))
            return

        if not unhang:
            if state[0][3] != next_state[0][3]:
                #print("State turn: {}".format(state[0][3]))
                #print("Next State turn: {}".format(next_state[0][3]))
                #print("Appending state/action, hanging memory")
                self.memory.append((state, action))
                #return True # Hang memory
            else:
                #print("Normal samestate->samestate, appending everything")
                self.memory.append((state, action, reward, next_state, done))
                #return False # Normal, unhang memory
        else: # Memory hanged, called again to unhang
            if state[0][3] != next_state[0][3]:
                #print("Unhanging now")
                #print("State turn: {}".format(state[0][3]))
                #print("Next State turn: {}".format(next_state[0][3]))
                try:
                    (oldstate, oldaction) = self.memory.pop()
                except:
                    return
                # CHECK IF THIS IS NOT NOTHING
                self.memory.append((oldstate, oldaction, reward, next_state, done))
                #return False # Unhang memory
            else:
                print("Critical error in remember()")
                print("States: {} {}".format(state[0][3], next_state[0][3]))
                print(self.memory)
                exit(0)
                return False

    # def act(self, state, action_mask):
    #     if np.random.rand() <= self.epsilon:
    #         valid_actions = np.where(action_mask)[0]
    #         return np.random.choice(valid_actions)
    #     state = torch.FloatTensor(state).unsqueeze(0)
    #     act_values = self.model(state)
    #     action_mask_tensor = torch.tensor(action_mask, dtype=torch.float32).unsqueeze(0)
    #     masked_act_values = act_values + (action_mask_tensor - 1) * 1e9  # Apply mask
    #     return torch.argmax(masked_act_values, dim=1).item()

    def act(self, state, action_mask, dimm=2):
        if np.random.rand() <= self.epsilon:
            valid_actions = np.where(action_mask)[0]
            selected_action = np.random.choice(valid_actions)
            return selected_action
        
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        act_values = self.model(state)
        #print("Act values:")
        #print(act_values)
        
        action_mask_tensor = torch.tensor(action_mask, dtype=torch.float32).unsqueeze(0).to(self.device)
        #print("Action_mask_tensor:")
        #print(action_mask_tensor)
        
        masked_act_values = act_values + (1 - action_mask_tensor.unsqueeze(1)) * -1e9  # Apply mask correctly
        #print("Masked_act_values:")
        #print(masked_act_values)
        
        selected_action = torch.argmax(masked_act_values, dim=dimm).item()  # Correct dimension
        #if selected_action == 2 and np.random.rand() < 0.05:
        if np.random.rand() < 0.0001:
            print("DEBUG PRINT: Selected Action")
            print(f"Action values: {act_values}, Masked action values: {masked_act_values}, Selected action: {selected_action}")
            print(f"State: {state}")
        
        return selected_action

    # def replay(self, batch_size):
    #     minibatch = random.sample(self.memory, batch_size)
    #     for state, action, reward, next_state, done in minibatch:
    #         state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
    #         next_state = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
    #         reward = torch.tensor(reward).to(self.device)

    #         with torch.no_grad():
    #             target = self.model(state).detach()
    #             if done:
    #                 target[0, 0, action] = reward
    #             else:
    #                 next_action = self.model(next_state).argmax(dim=2).item()
    #                 t = self.target_model(next_state)[0, 0, next_action]
    #                 target[0, 0, action] = reward + self.gamma * t

    #         self.optimizer.zero_grad()
    #         loss = nn.MSELoss()(self.model(state), target)
    #         loss.backward()
    #         torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
    #         self.optimizer.step()

    #     if self.epsilon > self.epsilon_min:
    #         self.epsilon *= self.epsilon_decay

    def replay(self, batch_size, env):
        if np.random.rand() < 0.001: # Rare debug print of memory
            print("DEBUG PRINT: MEMORY DUMP!")
            print("Length: {}".format(len(self.memory)))
            #for t in self.memory:
            #    #print(t[1])
            #    print("State:\t{}".format(t[0][0][3]))
            #    print("Action:\t{}".format(t[1]))
            #    print("Reward:\t{}".format(t[2]))
            #    print("Done:\t{}".format(t[4]))
            #    print("NextState:\t{}".format(t[3][0][3]))
        minibatch = random.sample(self.memory, batch_size)

        for state, action, reward, next_state, done in minibatch:
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            next_state = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
            reward = torch.tensor(reward).to(self.device)

            with torch.no_grad():
                target = self.model(state).detach()
                if done:
                    target[0, 0, action] = reward
                else:
                    #next_action = self.model(next_state).argmax(dim=2).item()
                    if next_state.cpu()[0][0][3]:
                        action_mask = generate_action_mask(env, 'player', next_state.cpu()[0][0])
                    else:
                        action_mask = generate_action_mask(env, 'dealer', next_state.cpu()[0][0])
                    next_action = self.act(next_state.cpu().numpy(), action_mask, dimm=3)
                    t = self.target_model(next_state)[0, 0, next_action]
                    target[0, 0, action] = reward + self.gamma * t

            self.optimizer.zero_grad()
            loss = nn.MSELoss()(self.model(state), target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


            
    def load(self, name):
        self.model.load_state_dict(torch.load(name))
        self.model.to(self.device) # Apparently necessary 

    def save(self, name):
        torch.save(self.model.state_dict(), name)

def generate_action_mask(env, player_type, state=None):
    all_actions = env.all_possible_actions(player_type)
    if state is not None:
        possible_actions = env.get_possible_actions(state)
    else:
        possible_actions = env.get_possible_actions()
    action_mask = np.zeros(len(all_actions))
    for action in possible_actions:
        action_index = all_actions.index(action)
        action_mask[action_index] = 1
    return action_mask


def train_agents(env, player_agent, dealer_agent, episodes, batch_size):
    update_interval = 10  # Update target network every 10 episodes
    save_interval = 1000
    for e in range(episodes):
        state = env.reset()
        state = np.array(env.get_state(player_view='player')).flatten()
        state = torch.FloatTensor(state).unsqueeze(0).to(player_agent.device) # Apparently necessary for GPU
        done = False
        replay_log = []


        while not done:
            if env.turn == 'player':
                memory_hanging = False
                action_mask = generate_action_mask(env, 'player')
                action = player_agent.act(state.cpu().numpy(), action_mask)
                action_tuple = env.all_possible_actions('player')[action]
                next_state, reward, done, _ = env.step(action_tuple)
                next_state = np.array(env.get_state(player_view='player')).flatten()
                next_state = torch.FloatTensor(next_state).unsqueeze(0).to(player_agent.device) # Apparently necessary for GPU
                player_agent.remember(state.cpu().numpy(), action, reward, next_state.cpu().numpy(), done, False)  # Reward is 0 during the game
                if env.turn == 'dealer':
                    #print("Player -> Dealer turn")
                    dealer_agent.remember(state.cpu().numpy(), action, -reward, next_state.cpu().numpy(), done, True, force_unhang=True)
                    memory_hanging = True
                replay_log.append({
                    'state': state.cpu().numpy().tolist(),
                    'action': int(action),
                    'reward': reward,
                    'next_state': next_state.cpu().numpy().tolist()
                })
                if not done:
                    state = next_state
                elif memory_hanging: # Game over/player->dealer, Player is the one who's hung, unhang
                    player_agent.remember(state.cpu().numpy(), action, reward, next_state.cpu().numpy(), done, True, force_unhang=True)
                if done and not memory_hanging: # Game over/player->player, Dealer is the one who's hung, unhang
                    dealer_agent.remember(state.cpu().numpy(), action, -reward, next_state.cpu().numpy(), done, True, force_unhang=True)
                if len(player_agent.memory) > batch_size and memory_hanging == False:
                    player_agent.replay(batch_size, env)
            elif env.turn == 'dealer':
                memory_hanging = False
                action_mask = generate_action_mask(env, 'dealer')
                action = dealer_agent.act(state.cpu().numpy(), action_mask)
                action_tuple = env.all_possible_actions('dealer')[action]
                next_state, reward, done, _ = env.step(action_tuple)
                next_state = np.array(env.get_state(player_view='dealer')).flatten()
                next_state = torch.FloatTensor(next_state).unsqueeze(0).to(dealer_agent.device) # Apparently necessary for GPU
                dealer_agent.remember(state.cpu().numpy(), action, -reward, next_state.cpu().numpy(), done, False)  # Reward is 0 during the game
                if env.turn == 'player':
                    #print("Dealer -> Player turn")
                    player_agent.remember(state.cpu().numpy(), action, reward, next_state.cpu().numpy(), done, True, force_unhang=True)
                    memory_hanging = True
                replay_log.append({
                    'state': state.cpu().numpy().tolist(),
                    'action': int(action),
                    'reward': reward,
                    'next_state': next_state.cpu().numpy().tolist()
                })
                if not done:
                    state = next_state
                elif memory_hanging: # Game over/dealer->player, Dealer is the one who's hung, unhang
                    dealer_agent.remember(state.cpu().numpy(), action, -reward, next_state.cpu().numpy(), done, True, force_unhang=True)
                if done and not memory_hanging: # Game over/dealer->dealer, Player is the one who's hung, unhang
                    player_agent.remember(state.cpu().numpy(), action, reward, next_state.cpu().numpy(), done, True, force_unhang=True)
                if len(dealer_agent.memory) > batch_size and memory_hanging == False:
                    dealer_agent.replay(batch_size, env)

            if done:
                replay_log.append({
                    #'state': state.tolist(),
                    #'action': int(action),
                    #'next_state': next_state.tolist(),
                    'result': {
                        'player_score': reward,
                        'dealer_score': -reward
                    }
                })

                if e % update_interval == 0:
                    player_agent.update_target_model()
                    dealer_agent.update_target_model()
                if e % save_interval == 0:
                    player_agent.save(f"player_model_{e}.pth")
                    dealer_agent.save(f"dealer_model_{e}.pth")
                print(f"episode: {e}/{episodes}, player_score: {reward}, dealer_score: {-reward}, epsilon_player: {player_agent.epsilon:.2}, epsilon_dealer: {dealer_agent.epsilon:.2}")
                
                # Save replay log to a file
                with open(f'Replays\\replay_episode_{e}.json', 'w') as f:
                    json.dump(replay_log, f, indent=2)
                
                break





if __name__ == "__main__":
    env = BuckshotRouletteEnv()
    state_dim = np.prod(np.array(env.get_state()).shape)  # Flatten state shape
    print("Dim Len: {}".format(state_dim))
    #exit(0)
    
    action_dim = len(env.all_possible_actions('player'))

    player_agent = DQNAgent(state_dim, action_dim)
    dealer_agent = DQNAgent(state_dim, action_dim)

    # Load the pre-trained models
    player_agent.load("player_model_53000_v1.pth")
    dealer_agent.load("dealer_model_53000_v1.pth")
    ## NOTE for this run - changed rewards from +1 -1 to +0.1 -1 in accordance with ~90%wr.
    ## Also gonna give it more freedom to do whatever it wants to do, min experimental value will be 0.01 instead of 0.05 (gradual decreasing)
    
    # Set the starting epsilon values
    player_agent.epsilon = 0.05
    player_agent.epsilon_decay = 0.99998
    player_agent.epsilon_min = 0.03
    dealer_agent.epsilon = 0.05
    dealer_agent.epsilon_decay = 0.99998
    dealer_agent.epsilon_min = 0.03

    train_agents(env, player_agent, dealer_agent, episodes=100000, batch_size=64)

    player_agent.save("player_model_final.pth")
    dealer_agent.save("dealer_model_final.pth")
    print("Models saved successfully.")