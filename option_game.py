import numpy as np

class Option:
    """Base class for all options."""
    def __init__(self, strike):
        self.strike = strike
        
    def calculate_probabilities(self):
        """Calculate probability distribution for two dice."""
        # Create a 6x6 grid of all possible dice combinations
        outcomes = np.zeros(13)  # Index 0 will be unused, outcomes[2] through outcomes[12]
        
        for die1 in range(1, 7):
            for die2 in range(1, 7):
                total = die1 + die2
                outcomes[total] += 1/36  # 36 possible combinations
                
        return outcomes
        
    def probability_in_the_money(self):
        """Abstract method to be implemented by subclasses."""
        raise NotImplementedError
        
    def delta(self):
        """Delta is the probability of being in the money."""
        return self.probability_in_the_money()
        
    def vega(self):
        """Vega is the probability that the roll equals the strike * 36."""
        probs = self.calculate_probabilities()
        if 2 <= self.strike <= 12:
            return probs[self.strike] * 36
        return 0


class Call(Option):
    """Call option - pays off when roll is greater than or equal to strike."""
    def __init__(self, strike):
        super().__init__(strike)
        
    def probability_in_the_money(self):
        """Probability that roll >= strike."""
        probs = self.calculate_probabilities()
        return sum(probs[self.strike:])
        
    def __str__(self):
        return f"{self.strike} call"


class Put(Option):
    """Put option - pays off when roll is less than or equal to strike."""
    def __init__(self, strike):
        super().__init__(strike)
        
    def probability_in_the_money(self):
        """Probability that roll <= strike."""
        probs = self.calculate_probabilities()
        return sum(probs[2:self.strike+1])
        
    def __str__(self):
        return f"{self.strike} put"


class RiskReversal:
    """Long a put and short a call."""
    def __init__(self, put_strike, call_strike):
        if put_strike >= call_strike:
            raise ValueError("Put strike must be lower than call strike")
        self.put = Put(put_strike)
        self.call = Call(call_strike)
        
    def delta(self):
        """Delta of risk reversal is put delta minus call delta."""
        return self.put.delta() - self.call.delta()
        
    def vega(self):
        """Vega of risk reversal is put vega plus call vega."""
        return self.put.vega() + self.call.vega()
        
    def __str__(self):
        return f"{self.put.strike}-{self.call.strike} risk reversal"


class CallSpread:
    """Long a call, short a call with a higher strike."""
    def __init__(self, lower_strike, higher_strike):
        if lower_strike >= higher_strike:
            raise ValueError("First strike must be lower than second strike")
        self.long_call = Call(lower_strike)
        self.short_call = Call(higher_strike)
        
    def delta(self):
        """Delta of call spread is long call delta minus short call delta."""
        return self.long_call.delta() - self.short_call.delta()
        
    def vega(self):
        """Vega of call spread is long call vega plus short call vega."""
        return self.long_call.vega() + self.short_call.vega()
        
    def __str__(self):
        return f"{self.long_call.strike}-{self.short_call.strike} call spread"


class PutSpread:
    """Long a put, short a put with a lower strike."""
    def __init__(self, higher_strike, lower_strike):
        if higher_strike <= lower_strike:
            raise ValueError("First strike must be higher than second strike")
        self.long_put = Put(higher_strike)
        self.short_put = Put(lower_strike)
        
    def delta(self):
        """Delta of put spread is long put delta minus short put delta."""
        return self.long_put.delta() - self.short_put.delta()
        
    def vega(self):
        """Vega of put spread is long put vega plus short put vega."""
        return self.long_put.vega() + self.short_put.vega()
        
    def __str__(self):
        return f"{self.long_put.strike}-{self.short_put.strike} put spread"


class Straddle:
    """Long a call and a put with the same strike."""
    def __init__(self, strike):
        self.call = Call(strike)
        self.put = Put(strike)
        
    def delta(self):
        """Delta of straddle is call delta plus put delta."""
        return self.call.delta() + self.put.delta() - 1  # Adjust for overlap
        
    def vega(self):
        """Vega of straddle is call vega plus put vega."""
        return self.call.vega() + self.put.vega()
        
    def __str__(self):
        return f"{self.call.strike} straddle"


class Strangle:
    """Long a call and a put with different strikes."""
    def __init__(self, put_strike, call_strike):
        if put_strike >= call_strike:
            raise ValueError("Put strike must be lower than call strike")
        self.put = Put(put_strike)
        self.call = Call(call_strike)
        
    def delta(self):
        """Delta of strangle is call delta plus put delta."""
        return self.call.delta() + self.put.delta() - 1  # Adjust for overlap
        
    def vega(self):
        """Vega of strangle is call vega plus put vega."""
        return self.call.vega() + self.put.vega()
        
    def __str__(self):
        return f"{self.put.strike}-{self.call.strike} strangle"


class Portfolio:
    """A portfolio of options and option strategies."""
    def __init__(self):
        self.positions = []
        
    def add_position(self, position, quantity=1):
        """
        Add a position to the portfolio.
        
        Args:
            position: An option or option strategy
            quantity: Number of contracts (positive for long, negative for short)
        """
        self.positions.append((position, quantity))
        
    def delta(self):
        """Calculate the total delta of the portfolio."""
        return sum(position.delta() * quantity for position, quantity in self.positions)
        
    def vega(self):
        """Calculate the total vega of the portfolio."""
        return sum(position.vega() * quantity for position, quantity in self.positions)
    
    def __str__(self):
        """String representation of the portfolio."""
        if not self.positions:
            return "Empty portfolio"
        
        result = "Portfolio:\n"
        for position, quantity in self.positions:
            prefix = "+" if quantity > 0 else ""
            result += f"  {prefix}{quantity} x {position}\n"
        return result.strip()


class DiceSimulator:
    """Simulates dice rolls and tracks option portfolio performance."""
    def __init__(self):
        self.rolls = []
        self.portfolio = Portfolio()
        self.bid_ask_spread = 0.05  # 5% spread by default
        
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
    
    def get_market_prices(self, option):
        """
        Get the bid and ask prices for an option.
        
        Returns:
            tuple: (bid, ask) prices in pennies
        """
        fair_value = self.calculate_option_value(option)
        bid = round((fair_value - self.bid_ask_spread/2) * 100)
        ask = round((fair_value + self.bid_ask_spread/2) * 100)
        return bid, ask
    
    def set_spread(self, spread):
        """Set the bid-ask spread."""
        self.bid_ask_spread = spread
        print(f"Bid-ask spread set to {spread:.2%}")
    
    def add_to_portfolio(self, option, quantity=1):
        """Add an option position to the portfolio."""
        self.portfolio.add_position(option, quantity)
        print(f"Added {quantity} x {option} to portfolio")
    
    def calculate_portfolio_value(self):
        """Calculate the current value of the entire portfolio."""
        total_value = 0
        for position, quantity in self.portfolio.positions:
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


