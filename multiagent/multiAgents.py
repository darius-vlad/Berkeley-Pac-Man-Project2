# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        ghostPositions = successorGameState.getGhostPositions()
        newFoodList = newFood.asList()

        # Initialize score from the current game state
        totalScore = successorGameState.getScore()

        # Calculate distances to ghosts and food
        distance_to_ghosts = [manhattanDistance(ghostPos, newPos) for ghostPos in ghostPositions]
        distance_to_food = [manhattanDistance(foodPos, newPos) for foodPos in newFoodList]

        # Penalize staying in the same position
        if newPos == currentGameState.getPacmanPosition():
            totalScore -= 10

        # Penalize proximity to active (non-scared) ghosts
        for i, dist in enumerate(distance_to_ghosts):
            if newScaredTimes[i] == 0:  # Ghost is not scared
                if dist < 2:
                    totalScore -= 200  # Large penalty for being close to a ghost
                else:
                    totalScore += 10 / (dist + 1)  # Slight encouragement to keep a safe distance

        # Encourage eating scared ghosts
        for i, dist in enumerate(distance_to_ghosts):
            if newScaredTimes[i] > 0:  # Ghost is scared
                totalScore += 200 / (dist + 1)  # Incentive to approach scared ghosts

        # Reward eating food
        if len(newFoodList) < len(currentGameState.getFood().asList()):
            totalScore += 100  # Reward for eating food

        # Encourage proximity to food
        if distance_to_food:
            closest_food_distance = min(distance_to_food)
            totalScore += 50 / (closest_food_distance + 1)  # Higher reward for being closer to food

        # Reward for eating power pellets
        capsules = currentGameState.getCapsules()
        if newPos in capsules:
            totalScore += 120  # High reward for eating a power pellet

        return totalScore

def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.
        """

        def minimax(agentIndex, depth, state):
            if state.isWin() or state.isLose() or depth == 0:
                return self.evaluationFunction(state)

            if agentIndex == 0:  # Pacman's turn (maximize)
                return maxValue(state, depth, agentIndex)
            else:  # Ghosts' turn (minimize)
                return minValue(state, depth, agentIndex)

        def maxValue(state, depth, agentIndex):
            max_eval = float("-inf")
            legal_actions = state.getLegalActions(agentIndex)

            if not legal_actions:
                return self.evaluationFunction(state)

            for action in legal_actions:
                successor = state.generateSuccessor(agentIndex, action)
                eval = minimax(agentIndex + 1, depth, successor)
                max_eval = max(max_eval, eval)
            return max_eval

        def minValue(state, depth, agentIndex):
            min_eval = float("inf")
            legal_actions = state.getLegalActions(agentIndex)
            if not legal_actions:  # Terminal state
                return self.evaluationFunction(state)

            next_agent = (agentIndex + 1) % state.getNumAgents()
            for action in legal_actions:
                successor = state.generateSuccessor(agentIndex, action)
                if next_agent == 0:  # Back to Pacman, decrement depth
                    eval = minimax(next_agent, depth - 1, successor)
                else:
                    eval = minimax(next_agent, depth, successor)
                min_eval = min(min_eval, eval)
            return min_eval

        legal_actions = gameState.getLegalActions(0)
        best_score = float("-inf")
        best_action = None
        for action in legal_actions:
            successor = gameState.generateSuccessor(0, action)
            score = minimax(1, self.depth, successor)
            if score > best_score:
                best_score = score
                best_action = action

        return best_action

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """

        def alphaBetaPruning(agentIndex, depth, state, alpha, beta):
            if state.isWin() or state.isLose() or depth == 0:
                return self.evaluationFunction(state)

            if agentIndex == 0:  # Pacman's turn (maximize)
                return maxValue(state, depth, alpha, beta)
            else:  # Ghosts' turn (minimize)
                return minValue(state, depth, agentIndex, alpha, beta)

        def maxValue(state, depth, alpha, beta):
            maxEval = float("-inf")
            legal_actions = state.getLegalActions(0)
            if not legal_actions:
                return self.evaluationFunction(state)

            for action in legal_actions:
                successor = state.generateSuccessor(0, action)
                eval = alphaBetaPruning(1, depth, successor, alpha, beta)
                maxEval = max(maxEval, eval)
                if maxEval > beta:
                    return maxEval
                alpha = max(alpha, maxEval)

            return maxEval

        def minValue(state, depth, agentIndex, alpha, beta):
            minEval = float("inf")
            legal_actions = state.getLegalActions(agentIndex)
            if not legal_actions:
                return self.evaluationFunction(state)

            next_agent = (agentIndex + 1) % state.getNumAgents()
            for action in legal_actions:
                successor = state.generateSuccessor(agentIndex, action)
                if next_agent == 0:  # Next is Pacman
                    eval = alphaBetaPruning(next_agent, depth - 1, successor, alpha, beta)
                else:
                    eval = alphaBetaPruning(next_agent, depth, successor, alpha, beta)
                minEval = min(minEval, eval)
                if minEval < alpha:
                    return minEval
                beta = min(beta, minEval)

            return minEval


        alpha = float("-inf")
        beta = float("inf")
        best_action = None
        legal_actions = gameState.getLegalActions(0)

        if not legal_actions:
            return None

        maxEval = float("-inf")
        for action in legal_actions:
            successor = gameState.generateSuccessor(0, action)
            eval = alphaBetaPruning(1, self.depth, successor, alpha, beta)
            if eval > maxEval:
                maxEval = eval
                best_action = action
            alpha = max(alpha, eval)

        return best_action


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction
        """

        def expectimax(agentIndex, depth, state):
            # If game is over or depth is 0, return evaluation
            if state.isWin() or state.isLose() or depth == 0:
                return self.evaluationFunction(state)

            if agentIndex == 0:  # Pacman's turn (maximize)
                return maxValue(state, depth)
            else:  # Ghost's turn (expectation)
                return expectValue(state, depth, agentIndex)

        def maxValue(state, depth):
            maxEval = float("-inf")
            legal_actions = state.getLegalActions(0)  # Pacman is agent 0
            if not legal_actions:
                return self.evaluationFunction(state)

            for action in legal_actions:
                successor = state.generateSuccessor(0, action)
                eval = expectimax(1, depth, successor)  # Go to first ghost (agent 1)
                maxEval = max(maxEval, eval)

            return maxEval

        def expectValue(state, depth, agentIndex):
            legal_actions = state.getLegalActions(agentIndex)
            if not legal_actions:
                return self.evaluationFunction(state)

            num_ghosts = state.getNumAgents() - 1
            next_agent = (agentIndex + 1) % state.getNumAgents()

            expectedValue = 0
            prob = 1 / len(legal_actions)  # Assume uniform distribution over actions

            for action in legal_actions:
                successor = state.generateSuccessor(agentIndex, action)
                if next_agent == 0:  # Back to Pacman
                    expectedValue += prob * expectimax(0, depth - 1, successor)
                else:
                    expectedValue += prob * expectimax(next_agent, depth, successor)

            return expectedValue

        # Find the best action for Pacman
        legal_actions = gameState.getLegalActions(0)  # Pacman's actions
        best_action = None
        maxEval = float("-inf")

        for action in legal_actions:
            successor = gameState.generateSuccessor(0, action)
            eval = expectimax(1, self.depth, successor)  # Start with first ghost (agent 1)
            if eval > maxEval:
                maxEval = eval
                best_action = action

        return best_action


def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
