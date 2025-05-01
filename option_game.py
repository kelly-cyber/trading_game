import numpy as np
from flask import current_app

# ---------------------------------------------------------------------------
# constants you can tune once at startup
C_DELTA      = 0.001   # ¢ per unit delta
C_VEGA       = 0.0001  # ¢ per unit vega
HALF_SPREAD  = 0.02   # 2¢ wide market   ( = 4¢ total width )
MIN_PRICE    = 0.00   # bids never below zero
# ---------------------------------------------------------------------------


class Option:
    """Base class for all options."""
    def __init__(self, strike):
        self.strike = strike
        
    def calculate_probabilities(self, first_roll=None):
        """
        Calculate probability distribution for two dice.
        
        Args:
            first_roll: Value of the first die if already rolled
        """
        # Create a 6x6 grid of all possible dice combinations
        outcomes = np.zeros(13)  # Index 0 will be unused, outcomes[2] through outcomes[12]
        
        if first_roll is None:
            # No dice rolled yet - standard distribution
            for die1 in range(1, 7):
                for die2 in range(1, 7):
                    total = die1 + die2
                    outcomes[total] += 1/36  # 36 possible combinations
        else:
            # First die already rolled - conditional distribution
            for die2 in range(1, 7):
                total = first_roll + die2
                outcomes[total] += 1/6  # 6 possible outcomes for second die
                
        return outcomes
        
<<<<<<< HEAD
    # def delta(self, first_roll=None):
    #     """Delta is the probability of being in the money."""
    #     return self.probability_in_the_money(first_roll)
        
=======
>>>>>>> upstream/master
    def vega(self, first_roll=None):
        """Vega is the probability that the roll equals the strike * 36."""
        probs = self.calculate_probabilities(first_roll)
        if 2 <= self.strike <= 12:
            # If first die is rolled, multiply by 6 instead of 36
            multiplier = 6 if first_roll is not None else 36
            return probs[self.strike] * multiplier
        return 0


class Call(Option):
    """Call option - pays off when roll is greater than or equal to strike."""
    def __init__(self, strike):
        super().__init__(strike)
        
    def delta(self, first_roll=None):
        """Probability that roll >= strike."""
        probs = self.calculate_probabilities(first_roll)
        return sum(probs[self.strike:])
        
    def __str__(self):
        return f"{self.strike} call"


class Put(Option):
    """Put option - pays off when roll is less than or equal to strike."""
    def __init__(self, strike):
        super().__init__(strike)
        
    def delta(self, first_roll=None):
        """Probability that roll <= strike."""
        probs = self.calculate_probabilities(first_roll)
        return sum(probs[:self.strike])-1
        
    def __str__(self):
        return f"{self.strike} put"


class RiskReversal:
    """Long a put and short a call."""
    def __init__(self, put_strike, call_strike):
        if put_strike >= call_strike:
            raise ValueError("Put strike must be lower than call strike")
        self.put = Put(put_strike)
        self.call = Call(call_strike)
        
    def delta(self, first_roll=None):
        """Delta of risk reversal is put delta minus call delta."""
        return self.put.delta(first_roll) - self.call.delta(first_roll)
        
    def vega(self, first_roll=None):
        """Vega of risk reversal is put vega minus call vega."""
        return self.put.vega(first_roll) - self.call.vega(first_roll)
        
    def __str__(self):
        return f"{self.put.strike}-{self.call.strike} risk reversal"


class CallSpread:
    """Long a call, short a call with a higher strike."""
    def __init__(self, lower_strike, higher_strike):
        if lower_strike >= higher_strike:
            raise ValueError("First strike must be lower than second strike")
        self.long_call = Call(lower_strike)
        self.short_call = Call(higher_strike)
        
    def delta(self, first_roll=None):
        """Delta of call spread is long call delta minus short call delta."""
        return self.long_call.delta(first_roll) - self.short_call.delta(first_roll)
        
    def vega(self, first_roll=None):
        """Vega of call spread is long call vega minus short call vega."""
        return self.long_call.vega(first_roll) - self.short_call.vega(first_roll)
        
    def __str__(self):
        return f"{self.long_call.strike}-{self.short_call.strike} call spread"


class PutSpread:
    """Long a put, short a put with a lower strike."""
    def __init__(self, higher_strike, lower_strike):
        if higher_strike <= lower_strike:
            raise ValueError("First strike must be higher than second strike")
        self.long_put = Put(higher_strike)
        self.short_put = Put(lower_strike)
        
    def delta(self, first_roll=None):
        """Delta of put spread is long put delta minus short put delta."""
        return self.long_put.delta(first_roll) - self.short_put.delta(first_roll)
        
    def vega(self, first_roll=None):
        """Vega of put spread is long put vega minus short put vega."""
        return self.long_put.vega(first_roll) - self.short_put.vega(first_roll)
        
    def __str__(self):
        return f"{self.long_put.strike}-{self.short_put.strike} put spread"


class Straddle:
    """Long a call and a put with the same strike."""
    def __init__(self, strike):
        self.call = Call(strike)
        self.put = Put(strike)
        
    def delta(self, first_roll=None):
        """Delta of straddle is call delta plus put delta."""
        return self.call.delta(first_roll) + self.put.delta(first_roll) 
        
    def vega(self, first_roll=None):
        """Vega of straddle is call vega plus put vega."""
        return self.call.vega(first_roll) + self.put.vega(first_roll)
        
    def __str__(self):
        return f"{self.call.strike} straddle"


class Strangle:
    """Long a call and a put with different strikes."""
    def __init__(self, put_strike, call_strike):
        if put_strike >= call_strike:
            raise ValueError("Put strike must be lower than call strike")
        self.put = Put(put_strike)
        self.call = Call(call_strike)
        
    def delta(self, first_roll=None):
        """Delta of strangle is call delta plus put delta."""
        return self.call.delta(first_roll) + self.put.delta(first_roll)
        
    def vega(self, first_roll=None):
        """Vega of strangle is call vega plus put vega."""
        return self.call.vega(first_roll) + self.put.vega(first_roll)
        
    def __str__(self):
        return f"{self.put.strike}-{self.call.strike} strangle"


class Portfolio:
    """A portfolio of options and option strategies."""
    def __init__(self):
        self.positions = []
        
    def add_position(self, position, quantity=1, entry_price=None):
        """
        Add a position to the portfolio.
        
        Args:
            position: An option or option strategy
            quantity: Number of contracts (positive for long, negative for short)
            entry_price: Price paid for the position (optional)
        """
        self.positions.append((position, quantity, entry_price))
        
    def delta(self, first_roll=None):
        """
        Calculate the total delta of the portfolio.
        
        Args:
            first_roll: Value of the first die if already rolled
        """
        return sum(position.delta(first_roll) * quantity for position, quantity, _ in self.positions)
        
    def vega(self, first_roll=None):
        """
        Calculate the total vega of the portfolio.
        
        Args:
            first_roll: Value of the first die if already rolled
        """
        return sum(position.vega(first_roll) * quantity for position, quantity, _ in self.positions)
        
    def calculate_pnl(self, current_values):
        """
        Calculate the PNL for each position and the total.
        
        Args:
            current_values: Dictionary mapping positions to their current values
            
        Returns:
            tuple: (total_pnl, list of position PNLs)
        """
        position_pnls = []
        total_pnl = 0
        
        for position, quantity, entry_price in self.positions:
            current_value = current_values[position]
            
            if entry_price is None:
                # Skip positions without an entry price for PNL calculation
                # but still include them in the results with None for PNL
                position_pnls.append((position, quantity, entry_price, current_value, None))
                continue
            
            position_pnl = (current_value - entry_price) * quantity
            position_pnls.append((position, quantity, entry_price, current_value, position_pnl))
            total_pnl += position_pnl
            
        return total_pnl, position_pnls
        
    def __str__(self):
        """String representation of the portfolio."""
        if not self.positions:
            return "No positions in portfolio"
            
        result = []
        for position, quantity, entry_price in self.positions:
            prefix = "+" if quantity > 0 else ""
            price_info = f" (entry: {entry_price:.2f})" if entry_price is not None else ""
            result.append(f"{prefix}{quantity} x {position}{price_info}")
            
        return "\n".join(result)


class DiceSimulator:
    """Simulates dice rolls and tracks option portfolio performance."""
    def __init__(self):
        self.rolls = []
        self.portfolio = Portfolio()
        # self.bid_ask_spread = 0.05  # 5% spread by default
        
    def roll_die(self, value=None):
        """
        Roll a single die and add to roll history.
        If value is provided, use that value instead of random.
        Returns None if maximum rolls (2) have been reached.
        """
        if len(self.rolls) >= 2:
            print("Maximum number of rolls (2) reached.")
            return None
            
        if value is None:
            value = np.random.randint(1, 7)
        
        self.rolls.append(value)
        print(f"Roll {len(self.rolls)}: {value}")
        return value
        
    def roll_two_dice(self, value1=None, value2=None):
        """Roll two dice at once."""
        if len(self.rolls) >= 2:
            print("Maximum number of rolls (2) reached.")
            return None
            
        die1 = self.roll_die(value1)
        if die1 is None:  # This shouldn't happen due to the check above, but just in case
            return None
            
        die2 = self.roll_die(value2)
        if die2 is None:
            return None
            
        return die1 + die2
    
    def reset_rolls(self):
        """Reset roll history."""
        self.rolls = []
        print("Roll history reset.")
    
    def get_total(self):
        """Get the sum of all rolls."""
        return sum(self.rolls)
    
    def calculate_option_value(self, option):
        """
        Calculate the current value of an option based on rolls so far.
        
        Returns:
            float: Expected value if not all dice rolled, actual payoff if all rolled
        """
        if isinstance(option, Call):
            if len(self.rolls) == 0:
                # Expected value before any rolls
                return sum(max(i - option.strike, 0) * prob 
                          for i, prob in enumerate(option.calculate_probabilities()) 
                          if i >= 2)
            elif len(self.rolls) == 1:
                # Expected value after one roll
                first_roll = self.rolls[0]
                return sum(max(first_roll + i - option.strike, 0) * (1/6) 
                          for i in range(1, 7))
            else:
                # Actual payoff after both rolls
                return max(sum(self.rolls) - option.strike, 0)
                
        elif isinstance(option, Put):
            if len(self.rolls) == 0:
                # Expected value before any rolls
                return sum(max(option.strike - i, 0) * prob 
                          for i, prob in enumerate(option.calculate_probabilities()) 
                          if i >= 2)
            elif len(self.rolls) == 1:
                # Expected value after one roll
                first_roll = self.rolls[0]
                return sum(max(option.strike - (first_roll + i), 0) * (1/6) 
                          for i in range(1, 7))
            else:
                # Actual payoff after both rolls
                return max(option.strike - sum(self.rolls), 0)
        
        # Handle strategy objects by calculating their component options
        elif hasattr(option, 'call') and hasattr(option, 'put'):
            # For Straddle and Strangle
            call_value = self.calculate_option_value(option.call)
            put_value = self.calculate_option_value(option.put)
            return call_value + put_value
            
        elif hasattr(option, 'long_call') and hasattr(option, 'short_call'):
            # For CallSpread
            long_value = self.calculate_option_value(option.long_call)
            short_value = self.calculate_option_value(option.short_call)
            return long_value - short_value
            
        elif hasattr(option, 'long_put') and hasattr(option, 'short_put'):
            # For PutSpread
            long_value = self.calculate_option_value(option.long_put)
            short_value = self.calculate_option_value(option.short_put)
            return long_value - short_value
            
        elif hasattr(option, 'put') and hasattr(option, 'call'):
            # For RiskReversal
            put_value = self.calculate_option_value(option.put)
            call_value = self.calculate_option_value(option.call)
            return put_value - call_value
            
        else:
            raise ValueError(f"Unsupported option type: {type(option)}")
    
    def add_to_portfolio(self, option, quantity=1, entry_price=None):
        """Add an option position to the portfolio."""
        self.portfolio.add_position(option, quantity, entry_price)
        price_info = f" at price {entry_price:.2f}" if entry_price is not None else ""
        print(f"Added {quantity} x {option}{price_info} to portfolio")
    
    def calculate_portfolio_value(self):
        """Calculate the current value of the entire portfolio."""
        total_value = 0
        for position, quantity, _ in self.portfolio.positions:
            position_value = self.calculate_option_value(position) * quantity
            total_value += position_value
        return total_value
    
    def print_portfolio_status(self):
        """Print the current status of the portfolio."""
        print("\nPortfolio Status:")
        print(self.portfolio)
        
        current_value = self.calculate_portfolio_value()
        print(f"Current Portfolio Value: {current_value:.2f}")
        print(f"Portfolio Delta: {self.portfolio.delta():.4f}")
        print(f"Portfolio Vega: {self.portfolio.vega():.4f}")
        
        if len(self.rolls) > 0:
            print(f"\nCurrent Rolls: {self.rolls}")
            print(f"Current Total: {self.get_total()}")

    def calculate_portfolio_pnl(self):
        """Calculate the PNL for the entire portfolio."""
        # Only calculate PNL if we have completed two rolls
        if len(self.rolls) < 2:
            return None, []
        
        # Calculate current values for all positions
        current_values = {}
        for position, _, _ in self.portfolio.positions:
            current_values[position] = self.calculate_option_value(position)
        
        # Calculate PNL
        return self.portfolio.calculate_pnl(current_values)

    def get_option_analytics(self, option, quantity=1):
        """
        Calculate and return analytics for an option before adding to portfolio.
        
        Args:
            option: The option to analyze
            quantity: Number of contracts (positive for buy, negative for sell)
        
        Returns:
            dict: Contains fair values, deltas, vegas, bid/ask for total position
        """
        fair_value_per = self.calculate_option_value(option)
        fair_value_total = fair_value_per * abs(quantity)
        
        # Pass the first roll to delta and vega calculations if available
        first_roll = self.rolls[0] if len(self.rolls) == 1 else None
        delta_per_contract = option.delta(first_roll)
        vega_per_contract = option.vega(first_roll)
        
        # Calculate total delta and vega based on quantity
        total_delta = delta_per_contract * quantity
        total_vega = vega_per_contract * quantity
        
        # Calculate portfolio-adjusted bid and ask prices for the total position
        bid_per, ask_per = self.calculate_portfolio_adjusted_prices(fair_value_per, delta_per_contract, vega_per_contract, quantity)
        bid_total = bid_per * abs(quantity)
        ask_total = ask_per * abs(quantity)
        
        # Calculate delta-neutral quantity (negative of inverse delta)
        # If delta is zero, we can't be delta neutral with this option
        portfolio_delta = self.portfolio.delta(first_roll)
        delta_neutral_quantity = -portfolio_delta/delta_per_contract if delta_per_contract != 0 else 0
        
        return {
            'fair_value_per': fair_value_per,
            'fair_value_total': fair_value_total,
            'bid_per': bid_per,
            'ask_per': ask_per,
            'bid_total': bid_total,
            'ask_total': ask_total,
            'delta_per_contract': delta_per_contract,
            'vega_per_contract': vega_per_contract,
            'total_delta': total_delta,
            'total_vega': total_vega,
            'delta_neutral_quantity': delta_neutral_quantity
        }


    def calculate_portfolio_adjusted_prices(
        self, fair_value, leg_delta, leg_vega, quantity=1
    ):
        """
        Risk-aware bid / ask around fair value, adjusted for how the trade
        changes *absolute* delta and vega of the book.
        """
        first_roll      = self.rolls[0] if len(self.rolls) == 1 else None
        delta_old       = self.portfolio.delta(first_roll)
        vega_old        = self.portfolio.vega(first_roll)

        # portfolio after filling ONE lot ( *quantity* already signed )
        delta_new = delta_old + leg_delta * quantity
        vega_new  = vega_old  + leg_vega  * quantity

        # +ve => trade improves risk, –ve => trade worsens risk
        risk_benefit = (
            C_DELTA * (abs(delta_old) - abs(delta_new))
            + C_VEGA * (abs(vega_old)  - abs(vega_new))
        )

        mid_price = fair_value + risk_benefit      # shift mid toward risk benefit
        bid_price = max(MIN_PRICE, mid_price - HALF_SPREAD)
        ask_price = mid_price + HALF_SPREAD

        return bid_price, ask_price
