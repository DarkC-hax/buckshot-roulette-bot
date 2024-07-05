import unittest
from buckshot_bot import BuckshotRouletteEnv
import random

class TestBuckshotRouletteEnv(unittest.TestCase):
    def test_reset(self):
        env = BuckshotRouletteEnv()
        for _ in range(1):
            state = env.reset()
            print("\nSample Game State:")
            env.to_string()
    
    def test_load_shotgun(self):
        env = BuckshotRouletteEnv()
        for _ in range(3):
            clip = env.load_shotgun()
            self.assertGreaterEqual(len(clip), 2)
            self.assertLessEqual(len(clip), 8)
            self.assertTrue(all(round in [0, 1] for round in clip))
            num_ones = clip.count(1)
            num_zeros = clip.count(0)
            self.assertLessEqual(num_ones, num_zeros, "Number of 1s is greater than number of 0s")
            #print("\nLoaded Clip:")
            #print(clip)
    
    def test_initialize_clip_probs(self):
        env = BuckshotRouletteEnv()
        
        env.clip = [1, 0]
        probs = env.initialize_clip_probs()
        expected_probs = [(0.5, 0.5), (0.5, 0.5)]
        self.assertEqual(probs, expected_probs)
        
        env.clip = [1, 1, 0, 0, 0]
        probs = env.initialize_clip_probs()
        expected_probs = [(0.4, 0.6), (0.4, 0.6), (0.4, 0.6), (0.4, 0.6), (0.4, 0.6)]
        self.assertEqual(probs, expected_probs)
        
        env.clip = [1, 1, 1, 0, 0, 0, 0]
        probs = env.initialize_clip_probs()
        expected_probs = [(0.42857142857142855, 0.5714285714285714)] * 7
        self.assertEqual(probs, expected_probs)
    
    def test_random_items(self):
        env = BuckshotRouletteEnv()
        for _ in range(1):
            rand_items = random.randint(2, 5)
            items = env.random_items(rand_items)
            print("\nRandom Items:")
            print(items)
    
    # def test_step(self):
    #     env = BuckshotRouletteEnv()
        
    #     # Initial state
    #     state = env.reset()
    #     print("\nInitial State:")
    #     print(state)
        
    #     # Test use_burner_phone
    #     state = env.reset()
    #     state, reward, done, _ = env.step(('player', 'use_burner_phone'))
    #     print("\nState after use_burner_phone:")
    #     print(state, reward, done)
    

    

    def test_update_probabilities_after_shot(self):
        env = BuckshotRouletteEnv()
        
        env.player_clip_probs = [(0.33, 0.66), (1, 0), (0.33, 0.66), (0.33, 0.66)]
        env.dealer_clip_probs = [(0.33, 0.66), (1, 0), (0.33, 0.66), (0.33, 0.66)]
        env.clip = [0, 1, 0, 1]
        env.update_probabilities_after_shot()
        
        expected_probs = [(1, 0), (0.5, 0.5), (0.5, 0.5)]
        self.assertEqual(env.player_clip_probs, expected_probs)
        self.assertEqual(env.dealer_clip_probs, expected_probs)
        
        env.player_clip_probs = [(0.5, 0.5), (0.5, 0.5), (1, 0)]
        env.dealer_clip_probs = [(1, 0), (0.5, 0.5), (0.5, 0.5)]
        env.clip = [1, 0, 1]
        env.update_probabilities_after_shot()
        
        expected_probs_dealer = [(0.5, 0.5), (0.5, 0.5)]
        expected_probs_player = [(0, 1), (1, 0)]
        self.assertEqual(env.player_clip_probs, expected_probs_player)
        self.assertEqual(env.dealer_clip_probs, expected_probs_dealer)
    
    def test_get_possible_actions(self):
        env = BuckshotRouletteEnv()
        
        env.turn = 'player'
        env.items = ['cigarettes', 'handcuffs']
        actions = env.get_possible_actions()
        expected_actions = [('player', 'use_cigarettes'), ('player', 'use_handcuffs'), ('player', 'shoot_self'), ('player', 'shoot_dealer')]
        self.assertEqual(set(actions), set(expected_actions))
        
        env.turn = 'dealer'
        env.dealer_items = ['magnifying_glass', 'beer']
        actions = env.get_possible_actions()
        expected_actions = [('dealer', 'use_magnifying_glass'), ('dealer', 'use_beer'), ('dealer', 'shoot_self'), ('dealer', 'shoot_dealer')]
        self.assertEqual(set(actions), set(expected_actions))

        ## TODO - test adrenaline (no need for handcuffs - that's tested later)

    def test_cigarettes(self):
        env = BuckshotRouletteEnv()
        # Initial state
        env.max_hp = 4
        env.player_hp = 1
        env.dealer_hp = 1
        env.items = ['cigarettes', 'adrenaline']
        env.dealer_items = ['cigarettes']
        env.turn = 'player'
        
        # Test use_cigarettes
        state, reward, done, _ = env.step(('player', 'use_cigarettes'))
        #print("\nState after use_cigarettes:")
        #env.to_string()
        self.assertEqual(env.player_hp, 2)
        self.assertEqual(env.items, ['adrenaline'])
        self.assertEqual(env.dealer_items, ['cigarettes'])

        # Test use_adrenaline_cigarettes
        state, reward, done, _ = env.step(('player', 'use_adrenaline_cigarettes'))
        #print("\nState after use_adrenaline_cigarettes:")
        #env.to_string()
        self.assertEqual(env.player_hp, 3)
        self.assertEqual(env.items, [])
        self.assertEqual(env.dealer_items, [])

        env.max_hp = 4
        env.player_hp = 1
        env.dealer_hp = 1
        env.items = ['cigarettes']
        env.dealer_items = ['cigarettes', 'adrenaline']
        env.turn = 'dealer'
        
        # Test use_cigarettes
        state, reward, done, _ = env.step(('dealer', 'use_cigarettes'))
        #print("\nState after use_cigarettes:")
        #env.to_string()
        self.assertEqual(env.dealer_hp, 2)
        self.assertEqual(env.dealer_items, ['adrenaline'])
        self.assertEqual(env.items, ['cigarettes'])

        # Test use_adrenaline_cigarettes
        state, reward, done, _ = env.step(('dealer', 'use_adrenaline_cigarettes'))
        #print("\nState after use_adrenaline_cigarettes:")
        #env.to_string()
        self.assertEqual(env.dealer_hp, 3)
        self.assertEqual(env.dealer_items, [])
        self.assertEqual(env.items, [])

    def test_handcuffs_and_shooting(self):
        env = BuckshotRouletteEnv()

        #################
        # Test Normally #
        #################

        env.reset()
        # Initial state
        env.max_hp = 4
        env.player_hp = 4
        env.dealer_hp = 4
        env.clip = [1, 1, 1, 1, 1, 1]
        env.items = ['handcuffs', 'handcuffs', 'handcuffs', 'adrenaline']
        env.dealer_items = ['handcuffs', 'handcuffs', 'handcuffs', 'adrenaline']
        env.turn = 'player'
        env.handcuffed = {'player': 0, 'dealer': 0}
        env.player_clip_probs = env.initialize_clip_probs()
        env.dealer_clip_probs = env.initialize_clip_probs()

        actions = env.get_possible_actions()
        self.assertTrue(('player', 'use_handcuffs') in actions)
        self.assertTrue(('player', 'use_adrenaline_handcuffs') in actions)

        # Test use_handcuffs
        state, reward, done, _ = env.step(('player', 'use_handcuffs'))
        #env.to_string()
        self.assertEqual(env.items, ['handcuffs', 'handcuffs', 'adrenaline'])
        self.assertEqual(env.dealer_items, ['handcuffs', 'handcuffs', 'handcuffs', 'adrenaline'])
        self.assertEqual(env.handcuffed, {'player': 0, 'dealer': 2})

        actions = env.get_possible_actions()
        self.assertFalse(('player', 'use_handcuffs') in actions)
        self.assertFalse(('player', 'use_adrenaline_handcuffs') in actions)

        # Test shooting
        state, reward, done, _ = env.step(('player', 'shoot_dealer'))
        self.assertEqual(env.dealer_hp, 3)
        self.assertEqual(env.turn, 'player')
        self.assertEqual(env.handcuffed, {'player': 0, 'dealer': 1})
        
        actions = env.get_possible_actions()
        self.assertFalse(('player', 'use_handcuffs') in actions)
        self.assertFalse(('player', 'use_adrenaline_handcuffs') in actions)

        state, reward, done, _ = env.step(('player', 'shoot_dealer'))
        self.assertEqual(env.dealer_hp, 2)
        self.assertEqual(env.turn, 'dealer')
        self.assertEqual(env.handcuffed, {'player': 0, 'dealer': 0})

        #### Test for dealer

        actions = env.get_possible_actions()
        self.assertTrue(('dealer', 'use_handcuffs') in actions)
        self.assertTrue(('dealer', 'use_adrenaline_handcuffs') in actions)

        # Test use_handcuffs
        state, reward, done, _ = env.step(('dealer', 'use_handcuffs'))
        #env.to_string()
        self.assertEqual(env.items, ['handcuffs', 'handcuffs', 'adrenaline'])
        self.assertEqual(env.dealer_items, ['handcuffs', 'handcuffs', 'adrenaline'])
        self.assertEqual(env.handcuffed, {'player': 2, 'dealer': 0})

        actions = env.get_possible_actions()
        self.assertFalse(('dealer', 'use_handcuffs') in actions)
        self.assertFalse(('dealer', 'use_adrenaline_handcuffs') in actions)

        # Test shooting
        state, reward, done, _ = env.step(('dealer', 'shoot_dealer'))
        self.assertEqual(env.player_hp, 3)
        self.assertEqual(env.turn, 'dealer')
        self.assertEqual(env.handcuffed, {'player': 1, 'dealer': 0})
        
        actions = env.get_possible_actions()
        self.assertFalse(('dealer', 'use_handcuffs') in actions)
        self.assertFalse(('dealer', 'use_adrenaline_handcuffs') in actions)

        state, reward, done, _ = env.step(('dealer', 'shoot_dealer'))
        self.assertEqual(env.player_hp, 2)
        self.assertEqual(env.turn, 'player')
        self.assertEqual(env.handcuffed, {'player': 0, 'dealer': 0})

        
        #env.to_string()
        
        #######################
        # Test for Adrenaline #
        #######################

        env.reset()
        # Initial state
        env.max_hp = 4
        env.player_hp = 4
        env.dealer_hp = 4
        env.clip = [1, 1, 1, 1, 1, 1]
        env.items = ['handcuffs', 'handcuffs', 'handcuffs', 'adrenaline']
        env.dealer_items = ['handcuffs', 'handcuffs', 'handcuffs', 'adrenaline']
        env.turn = 'player'
        env.handcuffed = {'player': 0, 'dealer': 0}
        env.player_clip_probs = env.initialize_clip_probs()
        env.dealer_clip_probs = env.initialize_clip_probs()

        actions = env.get_possible_actions()
        self.assertTrue(('player', 'use_handcuffs') in actions)
        self.assertTrue(('player', 'use_adrenaline_handcuffs') in actions)

        # Test use_handcuffs
        state, reward, done, _ = env.step(('player', 'use_adrenaline_handcuffs'))
        #env.to_string()
        self.assertEqual(env.items, ['handcuffs', 'handcuffs', 'handcuffs'])
        self.assertEqual(env.dealer_items, ['handcuffs', 'handcuffs', 'adrenaline'])
        self.assertEqual(env.handcuffed, {'player': 0, 'dealer': 2})

        actions = env.get_possible_actions()
        self.assertFalse(('player', 'use_handcuffs') in actions)
        self.assertFalse(('player', 'use_adrenaline_handcuffs') in actions)

        # Test shooting
        state, reward, done, _ = env.step(('player', 'shoot_dealer'))
        self.assertEqual(env.dealer_hp, 3)
        self.assertEqual(env.turn, 'player')
        self.assertEqual(env.handcuffed, {'player': 0, 'dealer': 1})
        
        actions = env.get_possible_actions()
        self.assertFalse(('player', 'use_handcuffs') in actions)
        self.assertFalse(('player', 'use_adrenaline_handcuffs') in actions)

        state, reward, done, _ = env.step(('player', 'shoot_dealer'))
        self.assertEqual(env.dealer_hp, 2)
        self.assertEqual(env.turn, 'dealer')
        self.assertEqual(env.handcuffed, {'player': 0, 'dealer': 0})

        #### Test for dealer

        actions = env.get_possible_actions()
        self.assertTrue(('dealer', 'use_handcuffs') in actions)
        self.assertTrue(('dealer', 'use_adrenaline_handcuffs') in actions)

        # Test use_handcuffs
        state, reward, done, _ = env.step(('dealer', 'use_adrenaline_handcuffs'))
        #env.to_string()
        self.assertEqual(env.items, ['handcuffs', 'handcuffs'])
        self.assertEqual(env.dealer_items, ['handcuffs', 'handcuffs'])
        self.assertEqual(env.handcuffed, {'player': 2, 'dealer': 0})

        actions = env.get_possible_actions()
        self.assertFalse(('dealer', 'use_handcuffs') in actions)
        self.assertFalse(('dealer', 'use_adrenaline_handcuffs') in actions)

        # Test shooting
        state, reward, done, _ = env.step(('dealer', 'shoot_dealer'))
        self.assertEqual(env.player_hp, 3)
        self.assertEqual(env.turn, 'dealer')
        self.assertEqual(env.handcuffed, {'player': 1, 'dealer': 0})
        
        actions = env.get_possible_actions()
        self.assertFalse(('dealer', 'use_handcuffs') in actions)
        self.assertFalse(('dealer', 'use_adrenaline_handcuffs') in actions)

        state, reward, done, _ = env.step(('dealer', 'shoot_dealer'))
        self.assertEqual(env.player_hp, 2)
        self.assertEqual(env.turn, 'player')
        self.assertEqual(env.handcuffed, {'player': 0, 'dealer': 0})

    def test_magnifying_glass(self):
        env = BuckshotRouletteEnv()

        #################
        # Test Normally #
        #################

        env.reset()
        # Initial state
        env.max_hp = 4
        env.player_hp = 4
        env.dealer_hp = 4
        env.clip = [1, 0, 1, 0, 1, 0]
        env.items = ['magnifying_glass', 'magnifying_glass', 'magnifying_glass', 'adrenaline']
        env.dealer_items = ['magnifying_glass', 'magnifying_glass', 'magnifying_glass', 'adrenaline']
        env.turn = 'player'
        env.player_clip_probs = env.initialize_clip_probs()
        env.dealer_clip_probs = env.initialize_clip_probs()

        self.assertEqual(env.player_clip_probs, [(0.5, 0.5), (0.5, 0.5), (0.5, 0.5), (0.5, 0.5), (0.5, 0.5), (0.5, 0.5)])
        self.assertEqual(env.dealer_clip_probs, [(0.5, 0.5), (0.5, 0.5), (0.5, 0.5), (0.5, 0.5), (0.5, 0.5), (0.5, 0.5)])

        actions = env.get_possible_actions()
        self.assertTrue(('player', 'use_magnifying_glass') in actions)
        self.assertTrue(('player', 'use_adrenaline_magnifying_glass') in actions)

        # Test use_magnifying_glass
        state, reward, done, _ = env.step(('player', 'use_magnifying_glass'))
        self.assertEqual(env.items, ['magnifying_glass', 'magnifying_glass', 'adrenaline'])
        self.assertEqual(env.dealer_items, ['magnifying_glass', 'magnifying_glass', 'magnifying_glass', 'adrenaline'])
        self.assertEqual(env.player_clip_probs, [(1, 0), (0.4, 0.6), (0.4, 0.6), (0.4, 0.6), (0.4, 0.6), (0.4, 0.6)])
        self.assertEqual(env.dealer_clip_probs, [(0.5, 0.5), (0.5, 0.5), (0.5, 0.5), (0.5, 0.5), (0.5, 0.5), (0.5, 0.5)])

        state, reward, done, _ = env.step(('player', 'shoot_dealer'))
        self.assertEqual(env.turn, 'dealer')
        state, reward, done, _ = env.step(('dealer', 'use_magnifying_glass'))

        self.assertEqual(env.items, ['magnifying_glass', 'magnifying_glass', 'adrenaline'])
        self.assertEqual(env.dealer_items, ['magnifying_glass', 'magnifying_glass', 'adrenaline'])
        self.assertEqual(env.player_clip_probs, [(0.4, 0.6), (0.4, 0.6), (0.4, 0.6), (0.4, 0.6), (0.4, 0.6)])
        self.assertEqual(env.dealer_clip_probs, [(0, 1), (0.5, 0.5), (0.5, 0.5), (0.5, 0.5), (0.5, 0.5)])

        state, reward, done, _ = env.step(('dealer', 'shoot_self'))

        self.assertEqual(env.player_clip_probs, [(0.5, 0.5), (0.5, 0.5), (0.5, 0.5), (0.5, 0.5)])
        self.assertEqual(env.dealer_clip_probs, [(0.5, 0.5), (0.5, 0.5), (0.5, 0.5), (0.5, 0.5)])

        self.assertEqual(env.turn, 'dealer')

        #######################
        # Test for Adrenaline #
        #######################

        env.reset()
        # Initial state
        env.max_hp = 4
        env.player_hp = 4
        env.dealer_hp = 4
        env.clip = [1, 0, 1, 0, 1, 0]
        env.items = ['magnifying_glass', 'magnifying_glass', 'magnifying_glass', 'adrenaline']
        env.dealer_items = ['magnifying_glass', 'magnifying_glass', 'magnifying_glass', 'adrenaline']
        env.turn = 'player'
        env.player_clip_probs = env.initialize_clip_probs()
        env.dealer_clip_probs = env.initialize_clip_probs()

        self.assertEqual(env.player_clip_probs, [(0.5, 0.5), (0.5, 0.5), (0.5, 0.5), (0.5, 0.5), (0.5, 0.5), (0.5, 0.5)])
        self.assertEqual(env.dealer_clip_probs, [(0.5, 0.5), (0.5, 0.5), (0.5, 0.5), (0.5, 0.5), (0.5, 0.5), (0.5, 0.5)])

        actions = env.get_possible_actions()
        self.assertTrue(('player', 'use_magnifying_glass') in actions)
        self.assertTrue(('player', 'use_adrenaline_magnifying_glass') in actions)

        # Test use_magnifying_glass
        state, reward, done, _ = env.step(('player', 'use_adrenaline_magnifying_glass'))
        self.assertEqual(env.items, ['magnifying_glass', 'magnifying_glass', 'magnifying_glass'])
        self.assertEqual(env.dealer_items, ['magnifying_glass', 'magnifying_glass', 'adrenaline'])
        self.assertEqual(env.player_clip_probs, [(1, 0), (0.4, 0.6), (0.4, 0.6), (0.4, 0.6), (0.4, 0.6), (0.4, 0.6)])
        self.assertEqual(env.dealer_clip_probs, [(0.5, 0.5), (0.5, 0.5), (0.5, 0.5), (0.5, 0.5), (0.5, 0.5), (0.5, 0.5)])

        state, reward, done, _ = env.step(('player', 'shoot_dealer'))
        self.assertEqual(env.turn, 'dealer')
        state, reward, done, _ = env.step(('dealer', 'use_adrenaline_magnifying_glass'))

        self.assertEqual(env.items, ['magnifying_glass', 'magnifying_glass'])
        self.assertEqual(env.dealer_items, ['magnifying_glass', 'magnifying_glass'])
        self.assertEqual(env.player_clip_probs, [(0.4, 0.6), (0.4, 0.6), (0.4, 0.6), (0.4, 0.6), (0.4, 0.6)])
        self.assertEqual(env.dealer_clip_probs, [(0, 1), (0.5, 0.5), (0.5, 0.5), (0.5, 0.5), (0.5, 0.5)])

        state, reward, done, _ = env.step(('dealer', 'shoot_self'))

        self.assertEqual(env.clip, [1, 0, 1, 0])
        self.assertEqual(env.player_clip_probs, [(0.5, 0.5), (0.5, 0.5), (0.5, 0.5), (0.5, 0.5)])
        self.assertEqual(env.dealer_clip_probs, [(0.5, 0.5), (0.5, 0.5), (0.5, 0.5), (0.5, 0.5)])

        self.assertEqual(env.turn, 'dealer')

    def test_use_beer(self):
        env = BuckshotRouletteEnv()

        #################
        # Test Normally #
        #################

        env.reset()
        # Initial state
        env.max_hp = 4
        env.player_hp = 4
        env.dealer_hp = 4
        env.clip = [1, 0, 1, 0, 1, 0]
        env.items = ['beer', 'beer', 'beer', 'adrenaline']
        env.dealer_items = ['beer', 'beer', 'beer', 'adrenaline']
        env.turn = 'player'
        env.player_clip_probs = env.initialize_clip_probs()
        env.dealer_clip_probs = env.initialize_clip_probs()

        self.assertEqual(env.clip, [1, 0, 1, 0, 1, 0])
        self.assertEqual(env.player_clip_probs, [(0.5, 0.5), (0.5, 0.5), (0.5, 0.5), (0.5, 0.5), (0.5, 0.5), (0.5, 0.5)])
        self.assertEqual(env.dealer_clip_probs, [(0.5, 0.5), (0.5, 0.5), (0.5, 0.5), (0.5, 0.5), (0.5, 0.5), (0.5, 0.5)])

        state, reward, done, _ = env.step(('player', 'use_beer'))
        self.assertEqual(env.items, ['beer', 'beer', 'adrenaline'])
        self.assertEqual(env.dealer_items, ['beer', 'beer', 'beer', 'adrenaline'])

        self.assertEqual(env.clip, [0, 1, 0, 1, 0])
        self.assertEqual(env.player_clip_probs, [(0.4, 0.6), (0.4, 0.6), (0.4, 0.6), (0.4, 0.6), (0.4, 0.6)])
        self.assertEqual(env.dealer_clip_probs, [(0.4, 0.6), (0.4, 0.6), (0.4, 0.6), (0.4, 0.6), (0.4, 0.6)])

        state, reward, done, _ = env.step(('player', 'shoot_dealer'))
        self.assertEqual(env.dealer_hp, 4)
        self.assertEqual(env.turn, 'dealer')

        self.assertEqual(env.clip, [1, 0, 1, 0])
        self.assertEqual(env.player_clip_probs, [(0.5, 0.5), (0.5, 0.5), (0.5, 0.5), (0.5, 0.5)])
        self.assertEqual(env.dealer_clip_probs, [(0.5, 0.5), (0.5, 0.5), (0.5, 0.5), (0.5, 0.5)])

        state, reward, done, _ = env.step(('dealer', 'use_beer'))
        state, reward, done, _ = env.step(('dealer', 'use_beer'))
        self.assertEqual(env.items, ['beer', 'beer', 'adrenaline'])
        self.assertEqual(env.dealer_items, ['beer', 'adrenaline'])

        self.assertEqual(env.clip, [1, 0])
        self.assertEqual(env.player_clip_probs, [(0.5, 0.5), (0.5, 0.5)])
        self.assertEqual(env.dealer_clip_probs, [(0.5, 0.5), (0.5, 0.5)])

        #######################
        # Test for Adrenaline #
        #######################

        env.reset()
        # Initial state
        env.max_hp = 4
        env.player_hp = 4
        env.dealer_hp = 4
        env.clip = [1, 0, 1, 0, 1, 0]
        env.items = ['beer', 'beer', 'beer', 'adrenaline']
        env.dealer_items = ['beer', 'beer', 'beer', 'adrenaline']
        env.turn = 'player'
        env.player_clip_probs = env.initialize_clip_probs()
        env.dealer_clip_probs = env.initialize_clip_probs()

        self.assertEqual(env.clip, [1, 0, 1, 0, 1, 0])
        self.assertEqual(env.player_clip_probs, [(0.5, 0.5), (0.5, 0.5), (0.5, 0.5), (0.5, 0.5), (0.5, 0.5), (0.5, 0.5)])
        self.assertEqual(env.dealer_clip_probs, [(0.5, 0.5), (0.5, 0.5), (0.5, 0.5), (0.5, 0.5), (0.5, 0.5), (0.5, 0.5)])

        state, reward, done, _ = env.step(('player', 'use_adrenaline_beer'))
        self.assertEqual(env.items, ['beer', 'beer', 'beer'])
        self.assertEqual(env.dealer_items, ['beer', 'beer', 'adrenaline'])

        self.assertEqual(env.clip, [0, 1, 0, 1, 0])
        self.assertEqual(env.player_clip_probs, [(0.4, 0.6), (0.4, 0.6), (0.4, 0.6), (0.4, 0.6), (0.4, 0.6)])
        self.assertEqual(env.dealer_clip_probs, [(0.4, 0.6), (0.4, 0.6), (0.4, 0.6), (0.4, 0.6), (0.4, 0.6)])

        state, reward, done, _ = env.step(('player', 'shoot_dealer'))
        self.assertEqual(env.dealer_hp, 4)
        self.assertEqual(env.turn, 'dealer')

        self.assertEqual(env.clip, [1, 0, 1, 0])
        self.assertEqual(env.player_clip_probs, [(0.5, 0.5), (0.5, 0.5), (0.5, 0.5), (0.5, 0.5)])
        self.assertEqual(env.dealer_clip_probs, [(0.5, 0.5), (0.5, 0.5), (0.5, 0.5), (0.5, 0.5)])

        state, reward, done, _ = env.step(('dealer', 'use_adrenaline_beer'))
        state, reward, done, _ = env.step(('dealer', 'use_beer'))
        self.assertEqual(env.items, ['beer', 'beer'])
        self.assertEqual(env.dealer_items, ['beer'])

        self.assertEqual(env.clip, [1, 0])
        self.assertEqual(env.player_clip_probs, [(0.5, 0.5), (0.5, 0.5)])
        self.assertEqual(env.dealer_clip_probs, [(0.5, 0.5), (0.5, 0.5)])

    def test_handsaw(self):
        env = BuckshotRouletteEnv()

        #################
        # Test Normally #
        #################

        env.reset()
        # Initial state
        env.max_hp = 4
        env.player_hp = 4
        env.dealer_hp = 4
        env.clip = [1, 1, 1, 1, 1, 1]
        env.items = ['handsaw', 'handsaw', 'handsaw', 'adrenaline']
        env.dealer_items = ['handsaw', 'handsaw', 'handsaw', 'adrenaline']
        env.turn = 'player'
        env.player_clip_probs = env.initialize_clip_probs()
        env.dealer_clip_probs = env.initialize_clip_probs()
        env.handsaw_active = {'player': 0, 'dealer': 0}

        self.assertEqual(env.player_hp, 4)
        self.assertEqual(env.dealer_hp, 4)
        self.assertEqual(env.handsaw_active, {'player': 0, 'dealer': 0})

        state, reward, done, _ = env.step(('player', 'use_handsaw'))
        self.assertEqual(env.items, ['handsaw', 'handsaw', 'adrenaline'])
        self.assertEqual(env.dealer_items, ['handsaw', 'handsaw', 'handsaw', 'adrenaline'])
        self.assertEqual(env.handsaw_active, {'player': 1, 'dealer': 0})
        self.assertEqual(env.player_hp, 4)
        self.assertEqual(env.dealer_hp, 4)

        state, reward, done, _ = env.step(('player', 'shoot_dealer'))
        self.assertEqual(env.handsaw_active, {'player': 0, 'dealer': 0})
        self.assertEqual(env.player_hp, 4)
        self.assertEqual(env.dealer_hp, 2)

        state, reward, done, _ = env.step(('dealer', 'use_handsaw'))
        self.assertEqual(env.items, ['handsaw', 'handsaw', 'adrenaline'])
        self.assertEqual(env.dealer_items, ['handsaw', 'handsaw', 'adrenaline'])
        self.assertEqual(env.handsaw_active, {'player': 0, 'dealer': 1})
        self.assertEqual(env.player_hp, 4)
        self.assertEqual(env.dealer_hp, 2)

        state, reward, done, _ = env.step(('dealer', 'shoot_dealer'))
        self.assertEqual(env.handsaw_active, {'player': 0, 'dealer': 0})
        self.assertEqual(env.player_hp, 2)
        self.assertEqual(env.dealer_hp, 2)

        #######################
        # Test for Adrenaline #
        #######################

        env.reset()
        # Initial state
        env.max_hp = 4
        env.player_hp = 4
        env.dealer_hp = 4
        env.clip = [1, 1, 1, 1, 1, 1]
        env.items = ['handsaw', 'handsaw', 'handsaw', 'adrenaline']
        env.dealer_items = ['handsaw', 'handsaw', 'handsaw', 'adrenaline']
        env.turn = 'player'
        env.player_clip_probs = env.initialize_clip_probs()
        env.dealer_clip_probs = env.initialize_clip_probs()
        env.handsaw_active = {'player': 0, 'dealer': 0}

        self.assertEqual(env.player_hp, 4)
        self.assertEqual(env.dealer_hp, 4)
        self.assertEqual(env.handsaw_active, {'player': 0, 'dealer': 0})

        state, reward, done, _ = env.step(('player', 'use_adrenaline_handsaw'))
        self.assertEqual(env.items, ['handsaw', 'handsaw', 'handsaw'])
        self.assertEqual(env.dealer_items, ['handsaw', 'handsaw', 'adrenaline'])
        self.assertEqual(env.handsaw_active, {'player': 1, 'dealer': 0})
        self.assertEqual(env.player_hp, 4)
        self.assertEqual(env.dealer_hp, 4)

        state, reward, done, _ = env.step(('player', 'shoot_dealer'))
        self.assertEqual(env.handsaw_active, {'player': 0, 'dealer': 0})
        self.assertEqual(env.player_hp, 4)
        self.assertEqual(env.dealer_hp, 2)

        state, reward, done, _ = env.step(('dealer', 'use_adrenaline_handsaw'))
        self.assertEqual(env.items, ['handsaw', 'handsaw'])
        self.assertEqual(env.dealer_items, ['handsaw', 'handsaw'])
        self.assertEqual(env.handsaw_active, {'player': 0, 'dealer': 1})
        self.assertEqual(env.player_hp, 4)
        self.assertEqual(env.dealer_hp, 2)

        state, reward, done, _ = env.step(('dealer', 'shoot_dealer'))
        self.assertEqual(env.handsaw_active, {'player': 0, 'dealer': 0})
        self.assertEqual(env.player_hp, 2)
        self.assertEqual(env.dealer_hp, 2)

    def test_expired_medicine(self):
        for i in range(10): # should deal with RNG
            env = BuckshotRouletteEnv()

            #################
            # Test Normally #
            #################

            env.reset()
            # Initial state
            env.max_hp = 4
            env.player_hp = 2
            env.dealer_hp = 2
            env.clip = [1, 1, 1, 1, 1, 1]
            env.items = ['expired_medicine', 'expired_medicine', 'expired_medicine', 'adrenaline']
            env.dealer_items = ['expired_medicine', 'expired_medicine', 'expired_medicine', 'adrenaline']
            env.turn = 'player'
            env.player_clip_probs = env.initialize_clip_probs()
            env.dealer_clip_probs = env.initialize_clip_probs()

            self.assertEqual(env.player_hp, 2)
            self.assertEqual(env.dealer_hp, 2)

            state, reward, done, _ = env.step(('player', 'use_expired_medicine'))
            self.assertEqual(env.items, ['expired_medicine', 'expired_medicine', 'adrenaline'])
            self.assertEqual(env.dealer_items, ['expired_medicine', 'expired_medicine', 'expired_medicine', 'adrenaline'])
            self.assertTrue(env.player_hp == 4 or env.player_hp == 1)

            env.reset()
            # Initial state
            env.max_hp = 4
            env.player_hp = 2
            env.dealer_hp = 2
            env.clip = [1, 1, 1, 1, 1, 1]
            env.items = ['expired_medicine', 'expired_medicine', 'expired_medicine', 'adrenaline']
            env.dealer_items = ['expired_medicine', 'expired_medicine', 'expired_medicine', 'adrenaline']
            env.turn = 'dealer'
            env.player_clip_probs = env.initialize_clip_probs()
            env.dealer_clip_probs = env.initialize_clip_probs()

            self.assertEqual(env.player_hp, 2)
            self.assertEqual(env.dealer_hp, 2)

            state, reward, done, _ = env.step(('dealer', 'use_expired_medicine'))
            self.assertEqual(env.items, ['expired_medicine', 'expired_medicine', 'expired_medicine', 'adrenaline'])
            self.assertEqual(env.dealer_items, ['expired_medicine', 'expired_medicine', 'adrenaline'])
            self.assertTrue(env.dealer_hp == 4 or env.dealer_hp == 1)

            #######################
            # Test for Adrenaline #
            #######################

            env.reset()
            # Initial state
            env.max_hp = 4
            env.player_hp = 2
            env.dealer_hp = 2
            env.clip = [1, 1, 1, 1, 1, 1]
            env.items = ['expired_medicine', 'expired_medicine', 'expired_medicine', 'adrenaline']
            env.dealer_items = ['expired_medicine', 'expired_medicine', 'expired_medicine', 'adrenaline']
            env.turn = 'player'
            env.player_clip_probs = env.initialize_clip_probs()
            env.dealer_clip_probs = env.initialize_clip_probs()

            self.assertEqual(env.player_hp, 2)
            self.assertEqual(env.dealer_hp, 2)

            state, reward, done, _ = env.step(('player', 'use_adrenaline_expired_medicine'))
            self.assertEqual(env.items, ['expired_medicine', 'expired_medicine', 'expired_medicine'])
            self.assertEqual(env.dealer_items, ['expired_medicine', 'expired_medicine', 'adrenaline'])
            self.assertTrue(env.player_hp == 4 or env.player_hp == 1)

            env.reset()
            # Initial state
            env.max_hp = 4
            env.player_hp = 2
            env.dealer_hp = 2
            env.clip = [1, 1, 1, 1, 1, 1]
            env.items = ['expired_medicine', 'expired_medicine', 'expired_medicine', 'adrenaline']
            env.dealer_items = ['expired_medicine', 'expired_medicine', 'expired_medicine', 'adrenaline']
            env.turn = 'dealer'
            env.player_clip_probs = env.initialize_clip_probs()
            env.dealer_clip_probs = env.initialize_clip_probs()

            self.assertEqual(env.player_hp, 2)
            self.assertEqual(env.dealer_hp, 2)

            state, reward, done, _ = env.step(('dealer', 'use_adrenaline_expired_medicine'))
            self.assertEqual(env.items, ['expired_medicine', 'expired_medicine', 'adrenaline'])
            self.assertEqual(env.dealer_items, ['expired_medicine', 'expired_medicine', 'expired_medicine'])
            self.assertTrue(env.dealer_hp == 4 or env.dealer_hp == 1)

        ########################
        # Test for Done (Loss) #
        ########################

        for i in range(20):
            if i == 19:
                self.assertFalse(True, 'RNG unlikely, check test_expired_medicine')
            env = BuckshotRouletteEnv()

            env.reset()
            # Initial state
            env.max_hp = 4
            env.player_hp = 1
            env.dealer_hp = 1
            env.clip = [1, 1, 1, 1, 1, 1]
            env.items = ['expired_medicine', 'expired_medicine', 'expired_medicine', 'adrenaline']
            env.dealer_items = ['expired_medicine', 'expired_medicine', 'expired_medicine', 'adrenaline']
            env.turn = 'player'
            env.player_clip_probs = env.initialize_clip_probs()
            env.dealer_clip_probs = env.initialize_clip_probs()

            self.assertEqual(env.player_hp, 1)
            self.assertEqual(env.dealer_hp, 1)

            state, reward, done, _ = env.step(('player', 'use_expired_medicine'))
            if done == True:
                self.assertLess(reward, 0)
                break

        ###################
        # Test for Max HP #
        ###################

        for i in range(20):
            if i == 19:
                self.assertFalse(True, 'RNG unlikely, check test_expired_medicine')
            
            env.reset()
            # Initial state
            env.max_hp = 4
            env.player_hp = 3
            env.dealer_hp = 3
            env.clip = [1, 1, 1, 1, 1, 1]
            env.items = ['expired_medicine', 'expired_medicine', 'expired_medicine', 'adrenaline']
            env.dealer_items = ['expired_medicine', 'expired_medicine', 'expired_medicine', 'adrenaline']
            env.turn = 'player'
            env.player_clip_probs = env.initialize_clip_probs()
            env.dealer_clip_probs = env.initialize_clip_probs()

            self.assertEqual(env.player_hp, 3)
            self.assertEqual(env.dealer_hp, 3)

            state, reward, done, _ = env.step(('player', 'use_expired_medicine'))
            if env.player_hp == 4:
                break

    def test_burner_phone(self):
        env = BuckshotRouletteEnv()

        #################
        # Test Normally #
        #################

        env.reset()
        # Initial state
        env.max_hp = 4
        env.player_hp = 4
        env.dealer_hp = 4
        env.clip = [1, 0, 1, 0, 1]
        env.items = ['burner_phone', 'burner_phone', 'burner_phone', 'adrenaline']
        env.dealer_items = ['burner_phone', 'burner_phone', 'burner_phone', 'adrenaline']
        env.turn = 'player'
        env.player_clip_probs = env.initialize_clip_probs()
        env.dealer_clip_probs = env.initialize_clip_probs()

        self.assertEqual(env.player_clip_probs, [(0.6, 0.4), (0.6, 0.4), (0.6, 0.4), (0.6, 0.4), (0.6, 0.4)])
        self.assertEqual(env.dealer_clip_probs, [(0.6, 0.4), (0.6, 0.4), (0.6, 0.4), (0.6, 0.4), (0.6, 0.4)])

        actions = env.get_possible_actions()
        self.assertTrue(('player', 'use_burner_phone') in actions)
        self.assertTrue(('player', 'use_adrenaline_burner_phone') in actions)

        # Test use_burner_phone
        state, reward, done, _ = env.step(('player', 'use_burner_phone'))
        self.assertEqual(env.items, ['burner_phone', 'burner_phone', 'adrenaline'])
        self.assertEqual(env.dealer_items, ['burner_phone', 'burner_phone', 'burner_phone', 'adrenaline'])
        
        self.assertTrue(env.player_clip_probs == [(1, 0), (0.5, 0.5), (0.5, 0.5), (0.5, 0.5), (0.5, 0.5)] or \
                        env.player_clip_probs == [(0.5, 0.5), (0.5, 0.5), (1, 0), (0.5, 0.5), (0.5, 0.5)] or \
                        env.player_clip_probs == [(0.5, 0.5), (0.5, 0.5), (0.5, 0.5), (0.5, 0.5), (1, 0)] or \
                        env.player_clip_probs == [(0.75, 0.25), (0, 1), (0.75, 0.25), (0.75, 0.25), (0.75, 0.25)] or \
                        env.player_clip_probs == [(0.75, 0.25), (0.75, 0.25), (0.75, 0.25), (0, 1), (0.75, 0.25)])


        self.assertEqual(env.dealer_clip_probs, [(0.6, 0.4), (0.6, 0.4), (0.6, 0.4), (0.6, 0.4), (0.6, 0.4)])

        env.reset()
        # Initial state
        env.max_hp = 4
        env.player_hp = 4
        env.dealer_hp = 4
        env.clip = [1, 0, 1, 0, 1]
        env.items = ['burner_phone', 'burner_phone', 'burner_phone', 'adrenaline']
        env.dealer_items = ['burner_phone', 'burner_phone', 'burner_phone', 'adrenaline']
        env.turn = 'dealer'
        env.player_clip_probs = env.initialize_clip_probs()
        env.dealer_clip_probs = env.initialize_clip_probs()

        self.assertEqual(env.player_clip_probs, [(0.6, 0.4), (0.6, 0.4), (0.6, 0.4), (0.6, 0.4), (0.6, 0.4)])
        self.assertEqual(env.dealer_clip_probs, [(0.6, 0.4), (0.6, 0.4), (0.6, 0.4), (0.6, 0.4), (0.6, 0.4)])

        actions = env.get_possible_actions()
        self.assertTrue(('dealer', 'use_burner_phone') in actions)
        self.assertTrue(('dealer', 'use_adrenaline_burner_phone') in actions)

        # Test use_burner_phone
        state, reward, done, _ = env.step(('dealer', 'use_burner_phone'))
        self.assertEqual(env.items, ['burner_phone', 'burner_phone', 'burner_phone', 'adrenaline'])
        self.assertEqual(env.dealer_items, ['burner_phone', 'burner_phone', 'adrenaline'])
        
        self.assertTrue(env.dealer_clip_probs == [(1, 0), (0.5, 0.5), (0.5, 0.5), (0.5, 0.5), (0.5, 0.5)] or \
                        env.dealer_clip_probs == [(0.5, 0.5), (0.5, 0.5), (1, 0), (0.5, 0.5), (0.5, 0.5)] or \
                        env.dealer_clip_probs == [(0.5, 0.5), (0.5, 0.5), (0.5, 0.5), (0.5, 0.5), (1, 0)] or \
                        env.dealer_clip_probs == [(0.75, 0.25), (0, 1), (0.75, 0.25), (0.75, 0.25), (0.75, 0.25)] or \
                        env.dealer_clip_probs == [(0.75, 0.25), (0.75, 0.25), (0.75, 0.25), (0, 1), (0.75, 0.25)])


        self.assertEqual(env.player_clip_probs, [(0.6, 0.4), (0.6, 0.4), (0.6, 0.4), (0.6, 0.4), (0.6, 0.4)])

        #######################
        # Test for Adrenaline #
        #######################

        env.reset()
        # Initial state
        env.max_hp = 4
        env.player_hp = 4
        env.dealer_hp = 4
        env.clip = [1, 0, 1, 0, 1]
        env.items = ['burner_phone', 'burner_phone', 'burner_phone', 'adrenaline']
        env.dealer_items = ['burner_phone', 'burner_phone', 'burner_phone', 'adrenaline']
        env.turn = 'player'
        env.player_clip_probs = env.initialize_clip_probs()
        env.dealer_clip_probs = env.initialize_clip_probs()

        self.assertEqual(env.player_clip_probs, [(0.6, 0.4), (0.6, 0.4), (0.6, 0.4), (0.6, 0.4), (0.6, 0.4)])
        self.assertEqual(env.dealer_clip_probs, [(0.6, 0.4), (0.6, 0.4), (0.6, 0.4), (0.6, 0.4), (0.6, 0.4)])

        actions = env.get_possible_actions()
        self.assertTrue(('player', 'use_burner_phone') in actions)
        self.assertTrue(('player', 'use_adrenaline_burner_phone') in actions)

        # Test use_burner_phone
        state, reward, done, _ = env.step(('player', 'use_adrenaline_burner_phone'))
        self.assertEqual(env.items, ['burner_phone', 'burner_phone', 'burner_phone'])
        self.assertEqual(env.dealer_items, ['burner_phone', 'burner_phone', 'adrenaline'])
        
        self.assertTrue(env.player_clip_probs == [(1, 0), (0.5, 0.5), (0.5, 0.5), (0.5, 0.5), (0.5, 0.5)] or \
                        env.player_clip_probs == [(0.5, 0.5), (0.5, 0.5), (1, 0), (0.5, 0.5), (0.5, 0.5)] or \
                        env.player_clip_probs == [(0.5, 0.5), (0.5, 0.5), (0.5, 0.5), (0.5, 0.5), (1, 0)] or \
                        env.player_clip_probs == [(0.75, 0.25), (0, 1), (0.75, 0.25), (0.75, 0.25), (0.75, 0.25)] or \
                        env.player_clip_probs == [(0.75, 0.25), (0.75, 0.25), (0.75, 0.25), (0, 1), (0.75, 0.25)])


        self.assertEqual(env.dealer_clip_probs, [(0.6, 0.4), (0.6, 0.4), (0.6, 0.4), (0.6, 0.4), (0.6, 0.4)])

        env.reset()
        # Initial state
        env.max_hp = 4
        env.player_hp = 4
        env.dealer_hp = 4
        env.clip = [1, 0, 1, 0, 1]
        env.items = ['burner_phone', 'burner_phone', 'burner_phone', 'adrenaline']
        env.dealer_items = ['burner_phone', 'burner_phone', 'burner_phone', 'adrenaline']
        env.turn = 'dealer'
        env.player_clip_probs = env.initialize_clip_probs()
        env.dealer_clip_probs = env.initialize_clip_probs()

        self.assertEqual(env.player_clip_probs, [(0.6, 0.4), (0.6, 0.4), (0.6, 0.4), (0.6, 0.4), (0.6, 0.4)])
        self.assertEqual(env.dealer_clip_probs, [(0.6, 0.4), (0.6, 0.4), (0.6, 0.4), (0.6, 0.4), (0.6, 0.4)])

        actions = env.get_possible_actions()
        self.assertTrue(('dealer', 'use_burner_phone') in actions)
        self.assertTrue(('dealer', 'use_adrenaline_burner_phone') in actions)

        # Test use_burner_phone
        state, reward, done, _ = env.step(('dealer', 'use_adrenaline_burner_phone'))
        self.assertEqual(env.items, ['burner_phone', 'burner_phone', 'adrenaline'])
        self.assertEqual(env.dealer_items, ['burner_phone', 'burner_phone', 'burner_phone'])
        
        self.assertTrue(env.dealer_clip_probs == [(1, 0), (0.5, 0.5), (0.5, 0.5), (0.5, 0.5), (0.5, 0.5)] or \
                        env.dealer_clip_probs == [(0.5, 0.5), (0.5, 0.5), (1, 0), (0.5, 0.5), (0.5, 0.5)] or \
                        env.dealer_clip_probs == [(0.5, 0.5), (0.5, 0.5), (0.5, 0.5), (0.5, 0.5), (1, 0)] or \
                        env.dealer_clip_probs == [(0.75, 0.25), (0, 1), (0.75, 0.25), (0.75, 0.25), (0.75, 0.25)] or \
                        env.dealer_clip_probs == [(0.75, 0.25), (0.75, 0.25), (0.75, 0.25), (0, 1), (0.75, 0.25)])


        self.assertEqual(env.player_clip_probs, [(0.6, 0.4), (0.6, 0.4), (0.6, 0.4), (0.6, 0.4), (0.6, 0.4)])

    def test_inverter(self):
        env = BuckshotRouletteEnv()

        #################
        # Test Normally #
        #################

        env.reset()
        # Initial state
        env.max_hp = 4
        env.player_hp = 4
        env.dealer_hp = 4
        env.clip = [1, 0, 1, 0, 1]
        env.items = ['inverter', 'inverter', 'inverter', 'adrenaline']
        env.dealer_items = ['inverter', 'inverter', 'inverter', 'adrenaline']
        env.turn = 'player'
        env.player_clip_probs = env.initialize_clip_probs()
        env.dealer_clip_probs = env.initialize_clip_probs()

        actions = env.get_possible_actions()
        self.assertTrue(('player', 'use_inverter') in actions)
        self.assertTrue(('player', 'use_adrenaline_inverter') in actions)

        # Test use_inverter
        state, reward, done, _ = env.step(('player', 'use_inverter'))
        self.assertEqual(env.items, ['inverter', 'inverter', 'adrenaline'])
        self.assertEqual(env.dealer_items, ['inverter', 'inverter', 'inverter', 'adrenaline'])
        self.assertEqual(env.clip, [0, 0, 1, 0, 1])
        actions = env.get_possible_actions()
        self.assertFalse(('player', 'use_inverter') in actions)
        self.assertFalse(('player', 'use_adrenaline_inverter') in actions)
        self.assertTrue(('player', 'shoot_dealer') in actions)
        self.assertTrue(('player', 'shoot_self') in actions)

        state, reward, done, _ = env.step(('player', 'shoot_dealer'))
        self.assertEqual(env.dealer_hp, 4)
        actions = env.get_possible_actions()
        self.assertTrue(('dealer', 'use_inverter') in actions)
        self.assertTrue(('dealer', 'use_adrenaline_inverter') in actions)

        # Test use_inverter
        state, reward, done, _ = env.step(('dealer', 'use_inverter'))
        self.assertEqual(env.items, ['inverter', 'inverter', 'adrenaline'])
        self.assertEqual(env.dealer_items, ['inverter', 'inverter', 'adrenaline'])
        self.assertEqual(env.clip, [1, 1, 0, 1])
        actions = env.get_possible_actions()
        self.assertFalse(('dealer', 'use_inverter') in actions)
        self.assertFalse(('dealer', 'use_adrenaline_inverter') in actions)
        self.assertTrue(('dealer', 'shoot_dealer') in actions)
        self.assertTrue(('dealer', 'shoot_self') in actions)

        #######################
        # Test for Adrenaline #
        #######################

        env.reset()
        # Initial state
        env.max_hp = 4
        env.player_hp = 4
        env.dealer_hp = 4
        env.clip = [1, 0, 1, 0, 1]
        env.items = ['inverter', 'inverter', 'inverter', 'adrenaline']
        env.dealer_items = ['inverter', 'inverter', 'inverter', 'adrenaline']
        env.turn = 'player'
        env.player_clip_probs = env.initialize_clip_probs()
        env.dealer_clip_probs = env.initialize_clip_probs()

        actions = env.get_possible_actions()
        self.assertTrue(('player', 'use_inverter') in actions)
        self.assertTrue(('player', 'use_adrenaline_inverter') in actions)

        # Test use_inverter
        state, reward, done, _ = env.step(('player', 'use_adrenaline_inverter'))
        self.assertEqual(env.items, ['inverter', 'inverter', 'inverter'])
        self.assertEqual(env.dealer_items, ['inverter', 'inverter', 'adrenaline'])
        self.assertEqual(env.clip, [0, 0, 1, 0, 1])
        actions = env.get_possible_actions()
        self.assertFalse(('player', 'use_inverter') in actions)
        self.assertFalse(('player', 'use_adrenaline_inverter') in actions)
        self.assertTrue(('player', 'shoot_dealer') in actions)
        self.assertTrue(('player', 'shoot_self') in actions)

        state, reward, done, _ = env.step(('player', 'shoot_dealer'))
        self.assertEqual(env.dealer_hp, 4)
        actions = env.get_possible_actions()
        self.assertTrue(('dealer', 'use_inverter') in actions)
        self.assertTrue(('dealer', 'use_adrenaline_inverter') in actions)

        # Test use_inverter
        state, reward, done, _ = env.step(('dealer', 'use_adrenaline_inverter'))
        self.assertEqual(env.items, ['inverter', 'inverter'])
        self.assertEqual(env.dealer_items, ['inverter', 'inverter'])
        self.assertEqual(env.clip, [1, 1, 0, 1])
        actions = env.get_possible_actions()
        self.assertFalse(('dealer', 'use_inverter') in actions)
        self.assertFalse(('dealer', 'use_adrenaline_inverter') in actions)
        self.assertTrue(('dealer', 'shoot_dealer') in actions)
        self.assertTrue(('dealer', 'shoot_self') in actions)

    def test_end_round(self):
        # test end-of-round functionality (item refills (8 maximum - can start w/6 and test for 8), win/loss, rewards, etc.)
        env = BuckshotRouletteEnv()

        #################
        # Test Normally #
        #################

        env.reset()
        # Initial state
        env.max_hp = 4
        env.player_hp = 4
        env.dealer_hp = 4
        env.clip = [1, 0]
        env.items = ['inverter', 'cigarettes', 'handsaw', 'adrenaline', 'cigarettes', 'cigarettes', 'cigarettes', 'cigarettes',]
        env.dealer_items = ['burner_phone', 'magnifying_glass', 'handcuffs', 'adrenaline', 'cigarettes', 'cigarettes', 'cigarettes', 'cigarettes']
        env.turn = 'player'
        env.player_clip_probs = env.initialize_clip_probs()
        env.dealer_clip_probs = env.initialize_clip_probs()

        actions = env.get_possible_actions()
        expected_actions =  [('player', 'use_cigarettes'), ('player', 'use_adrenaline_cigarettes'), ('player', 'use_adrenaline_magnifying_glass'), ('player', 'use_adrenaline'), ('player', 'use_adrenaline_burner_phone'), ('player', 'shoot_self'), \
                              ('player', 'use_cigarettes'), ('player', 'shoot_dealer'), ('player', 'use_adrenaline_handcuffs'), ('player', 'use_inverter'), ('player', 'use_handsaw')]
        self.assertEqual(set(actions), set(expected_actions))
        env.turn = 'dealer'
        actions = env.get_possible_actions()
        expected_actions =  [('dealer', 'use_burner_phone'), ('dealer', 'use_adrenaline_inverter'), ('dealer', 'use_magnifying_glass'), ('dealer', 'shoot_dealer'), \
                             ('dealer', 'use_adrenaline_handsaw'), ('dealer', 'shoot_self'), ('dealer', 'use_cigarettes'), ('dealer', 'use_adrenaline_cigarettes'), ('dealer', 'use_handcuffs'), ('dealer', 'use_adrenaline')]
        self.assertEqual(set(actions), set(expected_actions))
        env.turn = 'player'

        state, reward, done, _ = env.step(('player', 'use_handsaw'))
        state, reward, done, _ = env.step(('player', 'shoot_self'))
        self.assertEqual(env.player_hp, 2)
        self.assertEqual(env.turn, 'dealer')
        self.assertEqual(env.clip, [0])
        state, reward, done, _ = env.step(('dealer', 'use_handcuffs'))
        state, reward, done, _ = env.step(('dealer', 'use_adrenaline_inverter'))
        self.assertEqual(len(env.items), 6)
        self.assertEqual(len(env.dealer_items), 6)
        state, reward, done, _ = env.step(('dealer', 'shoot_self'))
        self.assertEqual(len(env.items), 8)
        self.assertEqual(len(env.dealer_items), 8)
        self.assertGreater(len(env.clip), 1)
        self.assertEqual(env.handcuffed, {'player': 0, 'dealer': 0})

    def test_end_game(self):
        # Model end of a game/lose conditions etc. don't worry about expired pills
        env = BuckshotRouletteEnv()

        #################
        # Test Normally #
        #################

        env.reset()
        # Initial state
        env.max_hp = 4
        env.player_hp = 2
        env.dealer_hp = 2
        env.clip = [1, 0]
        env.items = ['inverter', 'cigarettes', 'handsaw', 'adrenaline', 'cigarettes', 'cigarettes', 'cigarettes', 'cigarettes',]
        env.dealer_items = ['burner_phone', 'magnifying_glass', 'handcuffs', 'adrenaline', 'cigarettes', 'cigarettes', 'cigarettes', 'cigarettes']
        env.turn = 'player'
        env.player_clip_probs = env.initialize_clip_probs()
        env.dealer_clip_probs = env.initialize_clip_probs()

        state, reward, done, _ = env.step(('player', 'use_handsaw'))
        state, reward, done, _ = env.step(('player', 'shoot_self'))
        self.assertTrue(done)
        self.assertLess(reward, 0)

        env.reset()
        # Initial state
        env.max_hp = 4
        env.player_hp = 2
        env.dealer_hp = 2
        env.clip = [1, 0]
        env.items = ['inverter', 'cigarettes', 'handsaw', 'adrenaline', 'cigarettes', 'cigarettes', 'cigarettes', 'cigarettes',]
        env.dealer_items = ['burner_phone', 'magnifying_glass', 'handcuffs', 'adrenaline', 'cigarettes', 'cigarettes', 'cigarettes', 'cigarettes']
        env.turn = 'player'
        env.player_clip_probs = env.initialize_clip_probs()
        env.dealer_clip_probs = env.initialize_clip_probs()

        state, reward, done, _ = env.step(('player', 'use_handsaw'))
        state, reward, done, _ = env.step(('player', 'shoot_dealer'))
        self.assertTrue(done)
        self.assertGreater(reward, 0)

    def test_full_game(self):
        # MODEL A FULL GAME
        pass

if __name__ == '__main__':
    unittest.main()

