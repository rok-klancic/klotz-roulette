import random

class EuropeanRoulette:
    # Class representing a European roulette
    # It also contains methods for calculating the payout

    def __init__(self):
        self.numbers = [
            (0, "green"),
            (1, "red"),
            (2, "black"),
            (3, "red"),
            (4, "black"),
            (5, "red"),
            (6, "black"),
            (7, "red"),
            (8, "black"),
            (9, "red"),
            (10, "black"),
            (11, "black"),
            (12, "red"),
            (13, "black"),
            (14, "red"),
            (15, "black"),
            (16, "red"),
            (17, "black"),
            (18, "red"),
            (19, "red"),
            (20, "black"),
            (21, "red"),
            (22, "black"),
            (23, "red"),
            (24, "black"),
            (25, "red"),
            (26, "black"),
            (27, "red"),
            (28, "black"),
            (29, "black"),
            (30, "red"),
            (31, "black"),
            (32, "red"),
            (33, "black"),
            (34, "red"),
            (35, "black"),
            (36, "red")
        ]

    def _spin(self):
        # Returns a random number and its color
        return random.choice(self.numbers)

    def get_payout(self, bet_type, bet_amount, selection):
        # Returns the payout and the number/color combination of the spin

        if bet_type == "red_black":
            spin_result = self._spin()
            spin_color = spin_result[1]

            if (selection != "red") and (selection != "black"):
                raise ValueError("Selection must be 'red' or 'black'")

            if spin_color == selection:
                # print(f"You bet on {selection} and the spin result was {spin_result}")
                # print(f"You won {bet_amount}!")
                return bet_amount, spin_result

            else:
                # print(f"You bet on {selection} and the spin result was {spin_result}")
                # print(f"You lost {bet_amount}!")
                return -bet_amount, spin_result

        elif bet_type == "straight_up":
            spin_result = self._spin()
            spin_number = spin_result[0]

            if not isinstance(selection, list):
                raise ValueError("Selection must be a list of numbers")
            if not isinstance(bet_amount, list):
                raise ValueError("Bet amount must be a list")
            if len(selection) != len(bet_amount):
                raise ValueError("Selection and bet amount must have the same length")

            # print(f"You bet on {list(zip(selection, bet_amount))} and the spin result was {spin_result}")
            amount_won = sum([(amount * 35) if (spin_number == selection_number) else (- amount) for selection_number, amount in zip(selection, bet_amount)])
            if amount_won > 0:
                # print(f"You won {amount_won}!")
                pass
            else:
                # print(f"You lost {amount_won}!")
                pass
            return amount_won, spin_result

        else:
            raise ValueError(f"Invalid bet type: {bet_type}")


class KlotzRoulette:
    # Class representing a Klotz roulette
    # It also contains methods for calculating the payout

    def __init__(self):
        self.numbers = [
            (0, "green"),
            (1, "red"),
            (2, "black"),
            (3, "red"),
            (4, "black"),
            (5, "red"),
            (6, "black"),
            (7, "red"),
            (8, "black"),
            (9, "red"),
            (10, "black"),
            (11, "black"),
            (12, "red"),
            (13, "black"),
            (14, "red"),
            (15, "black"),
            (16, "red"),
            (17, "black"),
            (18, "red"),
            (19, "red"),
            (20, "black"),
            (21, "red"),
            (22, "black"),
            (23, "red"),
            (24, "black"),
            (25, "red"),
            (26, "black"),
            (27, "red"),
            (28, "black"),
            (29, "black"),
            (30, "red"),
            (31, "black"),
            (32, "red"),
            (33, "black"),
            (34, "red"),
            (35, "black"),
            (36, "red")
        ]

        self.probabilities = self._define_probabilities()

    def _define_probabilities(self):
        # Defines the probabilities of the numbers
        weights = []
        for _ in self.numbers:
            weights.append(random.random())

        total_weight = sum(weights)

        probabilities = [weight / total_weight for weight in weights]
        return probabilities

    def spin(self):
        # Returns a random number and its color
        return random.choices(self.numbers, weights=self.probabilities, k=1)[0]

    def get_payout(self, bet_type, bet_amount, selection):
        # Returns the payout and the number/color combination of the spin

        if bet_type == "red_black":
            spin_result = self.spin()
            spin_color = spin_result[1]

            if (selection != "red") and (selection != "black"):
                raise ValueError("Selection must be 'red' or 'black'")

            if spin_color == selection:
                # print(f"You bet on {selection} and the spin result was {spin_result}")
                # print(f"You won {bet_amount}!")
                return bet_amount, spin_result

            else:
                # print(f"You bet on {selection} and the spin result was {spin_result}")
                # print(f"You lost {bet_amount}!")
                return -bet_amount, spin_result

        elif bet_type == "straight_up":
            spin_result = self.spin()
            spin_number = spin_result[0]

            if not isinstance(selection, list):
                raise ValueError("Selection must be a list of numbers")
            if not isinstance(bet_amount, list):
                raise ValueError("Bet amount must be a list")
            if len(selection) != len(bet_amount):
                raise ValueError("Selection and bet amount must have the same length")

            # print(f"You bet on {list(zip(selection, bet_amount))} and the spin result was {spin_result}")
            amount_won = sum(
                (amount * 35) if (spin_number == selection_number) else (-amount)
                for selection_number, amount in zip(selection, bet_amount)
            )
            if amount_won > 0:
                # print(f"You won {amount_won}!")
                pass
            else:
                # print(f"You lost {amount_won}!")
                pass
            return amount_won, spin_result

        else:
            raise ValueError(f"Invalid bet type: {bet_type}")

    def get_probabilities(self):
        return self.probabilities

